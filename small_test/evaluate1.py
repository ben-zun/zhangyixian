# import os
# import json
# import time
# import csv
# import re
# import gc
# from typing import List, Dict, Any, Optional, Tuple

# import requests
# import torch
# import torch.nn.functional as F
# from transformers import (
#     AutoModelForCausalLM, AutoTokenizer,
#     AutoTokenizer as HFTokenizer,
#     AutoModel, AutoModelForSequenceClassification
# )
# from peft import PeftModel
# from sentence_transformers import SparseEncoder
# from pymilvus import MilvusClient

# # =========================
# # 0) ENV & CONFIG
# # =========================
# os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
# os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "1")
# os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "1")
# os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf_cache")

# BASE_MODEL_PATH = "./qwen_model/Qwen/Qwen2___5-7B-Instruct"
# LORA_PATH = "./lora_stage2"

# MILVUS_URI = "./milvus_lite.db"
# COLLECTION = "materials_hybrid"

# # ✅ 必须改成你真实存在的 snapshots 目录
# BGE_DIR = "./hf_cache/transformers/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
# SPLADE_DIR = "./hf_cache/transformers/models--naver--splade-cocondenser-ensembledistil/snapshots/49cf4c7b0db5b870a401ddf5e2669993ef3699c7"
# RERANK_DIR = "./hf_cache/transformers/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70"

# # ====== ARK (Planner/Filter/Judge/QGen) ======
# ARK_API_KEY = os.getenv("ARK_API_KEY", "1550c65b-2643-4c98-9c89-e63c2762cbe8")  # ✅ 不写死
# ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
# ARK_JUDGE_MODEL = os.getenv("ARK_JUDGE_MODEL", "ep-20260110133352-bczxh")
# ARK_QGEN_MODEL = os.getenv("ARK_QGEN_MODEL", ARK_JUDGE_MODEL)
# ROUTER_MODEL = os.getenv("ROUTER_MODEL", ARK_JUDGE_MODEL)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


# # =========================
# # 1) Utils
# # =========================
# def flush_gpu():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()
#     print("🧹 GPU Memory Flushed.")

# def _assert_exists(path: str, name: str):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"[{name}] Path not found: {path}")

# def ark_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 512) -> str:
#     """
#     同步 ARK 调用（planner/filter 用）
#     - timeout 拉长（避免你之前 5s 超时）
#     - 失败重试
#     """
#     if not ARK_API_KEY:
#         raise RuntimeError("Missing env ARK_API_KEY")
#     if not model:
#         raise RuntimeError("Missing ARK model endpoint (ep-xxxx)")

#     headers = {"Authorization": f"Bearer {ARK_API_KEY}", "Content-Type": "application/json"}
#     payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

#     last_err = None
#     for attempt in range(3):
#         try:
#             r = requests.post(ARK_BASE_URL, headers=headers, json=payload, timeout=120)
#             r.raise_for_status()
#             data = r.json()
#             return data["choices"][0]["message"]["content"]
#         except Exception as e:
#             last_err = e
#             print(f"ARK API Error (attempt {attempt+1}/3): {e}")
#             time.sleep(1.5)

#     print(f"ARK API Failed: {last_err}")
#     return "{}"

# def _extract_json_array(text: str) -> str:
#     m = re.search(r"\[[\s\S]*\]", text or "")
#     return m.group(0) if m else ""

# def _extract_json_obj_or_array(text: str):
#     m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text or "")
#     if not m:
#         return None
#     raw = m.group(1)
#     raw = raw.replace("“", "\"").replace("”", "\"").replace("，", ",")
#     raw = re.sub(r",\s*}", "}", raw)
#     raw = re.sub(r",\s*]", "]", raw)
#     try:
#         return json.loads(raw)
#     except:
#         return None

# # --- 化学式规范化（把 nacl -> NaCl 纠正） ---
# _FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
# _ELEMENT_SYMBOLS = {
#     "H","He","Li","Be","B","C","N","O","F","Ne",
#     "Na","Mg","Al","Si","P","S","Cl","Ar",
#     "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
#     "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
#     "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
#     "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
#     "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
# }

# def normalize_formula_guess(token: str) -> Optional[str]:
#     if not token:
#         return None
#     s = re.sub(r"[^A-Za-z0-9]", "", token.strip())
#     if not s:
#         return None
#     if _FORMULA_RE.fullmatch(s):
#         return s
#     if s.isdigit() or len(s) > 32:
#         return None

#     i = 0
#     out = []
#     while i < len(s):
#         if s[i].isdigit():
#             j = i
#             while j < len(s) and s[j].isdigit():
#                 j += 1
#             out.append(s[i:j])
#             i = j
#             continue

#         sym = None
#         if i + 2 <= len(s):
#             cand2 = s[i:i+2]
#             c2 = cand2[0].upper() + cand2[1].lower()
#             if c2 in _ELEMENT_SYMBOLS:
#                 sym = c2
#                 i += 2
#         if sym is None:
#             cand1 = s[i:i+1]
#             c1 = cand1[0].upper()
#             if c1 in _ELEMENT_SYMBOLS:
#                 sym = c1
#                 i += 1
#         if sym is None:
#             return None
#         out.append(sym)

#     f = "".join(out)
#     return f if _FORMULA_RE.fullmatch(f) else None


# # =========================
# # 2) Offline Dense/Sparse/Rerank
# # =========================
# def build_dense_encoder():
#     tok = HFTokenizer.from_pretrained(BGE_DIR, local_files_only=True)
#     mdl = AutoModel.from_pretrained(BGE_DIR, local_files_only=True).to("cpu").eval()

#     @torch.no_grad()
#     def encode(text: str) -> List[float]:
#         inp = tok([text], padding=True, truncation=True, return_tensors="pt").to("cpu")
#         out = mdl(**inp)
#         last = out.last_hidden_state
#         mask = inp["attention_mask"].unsqueeze(-1).float()
#         emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # mean pooling
#         emb = F.normalize(emb, p=2, dim=1)
#         return emb[0].cpu().tolist()

#     return encode

# def build_reranker():
#     tok = HFTokenizer.from_pretrained(RERANK_DIR, local_files_only=True)
#     mdl = AutoModelForSequenceClassification.from_pretrained(RERANK_DIR, local_files_only=True).to("cpu").eval()

#     @torch.no_grad()
#     def predict(pairs: List[List[str]], batch_size: int = 16) -> List[float]:
#         scores: List[float] = []
#         for i in range(0, len(pairs), batch_size):
#             batch = pairs[i:i+batch_size]
#             qs = [x[0] for x in batch]
#             ds = [x[1] for x in batch]
#             inp = tok(qs, ds, padding=True, truncation=True, return_tensors="pt").to("cpu")
#             out = mdl(**inp)
#             logits = out.logits
#             if logits.shape[-1] == 1:
#                 s = logits.squeeze(-1)
#             else:
#                 s = logits[:, -1]
#             scores.extend(s.detach().cpu().tolist())
#         return scores

#     return predict

# def to_sparse_list(vec):
#     vec = vec.cpu()
#     if vec.is_sparse:
#         vec = vec.coalesce()
#         idx = vec.indices()[0].tolist()
#         val = vec.values().tolist()
#     else:
#         nz = (vec != 0).nonzero(as_tuple=False).flatten()
#         idx = nz.tolist()
#         val = vec[nz].tolist()
#     return [(int(i), float(x)) for i, x in zip(idx, val)]


# # =========================
# # 3) Model Loaders
# # =========================
# def load_base_model_only():
#     print("🚀 Loading BASE Model...")
#     _assert_exists(BASE_MODEL_PATH, "BASE_MODEL_PATH")
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, local_files_only=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL_PATH,
#         device_map="auto",
#         dtype=DTYPE,                    # ✅ dtype, 不用 torch_dtype
#         trust_remote_code=True,
#         local_files_only=True,
#     ).eval()
#     return tokenizer, model

# def load_lora_model_only():
#     print("🚀 Loading LoRA Model (Base + Adapter)...")
#     _assert_exists(LORA_PATH, "LORA_PATH")
#     tokenizer, base_model = load_base_model_only()
#     print(f"   + Attaching LoRA adapter from {LORA_PATH}...")
#     model = PeftModel.from_pretrained(base_model, LORA_PATH, local_files_only=True).eval()
#     return tokenizer, model

# def load_rag_components():
#     print("📚 Loading RAG Retrievers (CPU Mode)...")
#     try:
#         client = MilvusClient(uri=MILVUS_URI)
#     except Exception as e:
#         print(f"⚠️ Milvus Load Error: {e}. RAG disabled.")
#         client = None

#     dense_encode = build_dense_encoder()

#     try:
#         sparse_encoder = SparseEncoder(SPLADE_DIR, device="cpu", trust_remote_code=True, local_files_only=True)
#     except Exception as e:
#         print(f"⚠️ SPLADE Load Error: {e}. Dense-only.")
#         sparse_encoder = None

#     rerank_predict = build_reranker()

#     return client, dense_encode, sparse_encoder, rerank_predict


# # =========================
# # 4) Generation (Local LLM)
# # =========================
# @torch.no_grad()
# def local_generate(tokenizer, model, messages, max_new_tokens=1024, temperature=0.1):
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
#     out = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=(temperature > 0),
#         temperature=temperature,
#         repetition_penalty=1.1,
#     )
#     text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
#     return text.strip()


# # =========================
# # 5) B(Chainlit) RAG 路由逻辑 —— 移植到 A (Planner/Filter 用 ARK)
# # =========================
# def parse_filter_expression_ark(query: str) -> str:
#     """
#     对齐 B：只有用户明确筛选时才返回表达式，否则返回空字符串。
#     """
#     sys_prompt = (
#         "你是一个 Milvus 数据库查询专家。请将用户的自然语言筛选要求转换为 Milvus 的 filter 表达式。\n"
#         "### 数据库 Schema 定义\n"
#         "- band_gap (float): 带隙 eV\n"
#         "- ehull (float): eV/atom\n"
#         "- is_stable (bool): 是否稳定\n"
#         "- formula (str): 化学式\n\n"
#         "### 转换规则\n"
#         "1) 只输出表达式字符串，不要 markdown，不要解释。\n"
#         "2) 允许 && || (Milvus expr 支持)。\n"
#         "3) 若没有明确筛选要求（只是问概念/单材料属性），输出空字符串。\n"
#         "示例：\n"
#         "- 带隙大于1.5 -> band_gap > 1.5\n"
#         "- 稳定的且带隙小于1 -> is_stable == true && band_gap < 1.0\n"
#     )
#     resp = ark_chat(
#         ROUTER_MODEL,
#         [{"role": "system", "content": sys_prompt},
#          {"role": "user", "content": query}],
#         temperature=0.0,
#         max_tokens=128
#     ).strip()

#     # 清理
#     expr = resp.replace("`", "").replace('"', "").replace("'", "").strip()

#     # 快速拦截明显不是 expr 的输出
#     if (not expr) or expr.startswith("{") or expr.startswith("[") or "（" in expr or "）" in expr:
#         return ""

#     # 只允许表达式字符
#     if not re.fullmatch(r"[a-zA-Z0-9_ \.\<\>\=\!\&\|\(\)]+", expr):
#         return ""

#     # 必须包含字段名才算筛选
#     if ("band_gap" not in expr) and ("ehull" not in expr) and ("is_stable" not in expr) and ("formula" not in expr):
#         return ""

#     return expr


# def _is_valid_formula_token(s: str) -> bool:
#     if not s or len(s) > 32:
#         return False
#     return re.fullmatch(r"[A-Za-z0-9\(\)\.\-\u00B7]+", s) is not None


# def global_search_planner_ark(query: str) -> List[str]:
#     """
#     对齐 B：纯 LLM（ARK）提取/标准化：
#     输出 JSON array: [{"formula":"...","confidence":0~1}, ...]
#     """
#     sys_prompt = (
#         "你是材料实体标准化器。任务：从用户问题中提取“需要去材料数据库检索的材料化学式”。\n"
#         "\n"
#         "## 重要约束（必须遵守）\n"
#         "1) 只输出一个 JSON 数组（array），不要输出任何解释、markdown、额外文字。\n"
#         "2) 数组元素为对象：{\"formula\": \"...\", \"confidence\": 0~1}\n"
#         "3) 只在你有把握时才填 formula；不确定就 formula:\"\"，confidence<=0.4。禁止瞎编。\n"
#         "4) 支持中文材料名、俗名、缩写：\n"
#         "   - 例：砷化镓->GaAs；氧化钇->Y2O3；氧化锌->ZnO；磷酸铁锂/LFP->LiFePO4。\n"
#         "5) 若问题是概念/原理/闲聊，输出空数组 []。\n"
#         "6) 最多输出 4 个对象。\n"
#         "\n"
#         "输出示例：[{\"formula\":\"GaAs\",\"confidence\":0.95},{\"formula\":\"\",\"confidence\":0.2}]"
#     )

#     user_prompt = f"用户问题：{query}\n请输出 JSON 数组："

#     for attempt in range(3):
#         try:
#             resp = ark_chat(
#                 ROUTER_MODEL,
#                 [{"role": "system", "content": sys_prompt},
#                  {"role": "user", "content": user_prompt}],
#                 temperature=0.0,
#                 max_tokens=256
#             )

#             arr_text = _extract_json_array(resp)
#             if not arr_text:
#                 raise ValueError(f"LLM 输出没有 JSON array: {resp[:200]}")

#             items = json.loads(arr_text)

#             formulas: List[str] = []
#             for it in items[:6]:
#                 f = (it.get("formula") or "").strip()
#                 try:
#                     conf = float(it.get("confidence", 0.0) or 0.0)
#                 except:
#                     conf = 0.0

#                 if (not f) or (conf < 0.55):
#                     continue

#                 # 先规范化大小写（nacl->NaCl）
#                 f2 = normalize_formula_guess(f) or f
#                 if not _is_valid_formula_token(f2):
#                     continue
#                 formulas.append(f2)

#             # 去重保序
#             uniq = []
#             for x in formulas:
#                 if x not in uniq:
#                     uniq.append(x)

#             return uniq[:4]

#         except Exception as e:
#             print(f"Planner Error attempt {attempt+1}/3: {e}")
#             time.sleep(0.8)

#     return []


# def search_pipeline_B(
#     client: MilvusClient,
#     dense_encode_fn,
#     sparse_encoder,
#     rerank_predict_fn,
#     query: str,
#     filter_expr: str = "",
#     top_n: int = 3,
# ) -> List[Dict[str, Any]]:
#     """
#     对齐 B：Dense + (Sparse) -> 合并候选 -> Rerank -> B 的排序逻辑：
#     - filter: stable > score > exact
#     - non-filter: exact > stable > score
#     """
#     if client is None:
#         return []

#     target_formula = normalize_formula_guess(query.strip())  # exact match 用
#     search_params = {
#         "collection_name": COLLECTION,
#         "limit": 60,
#         "output_fields": ["pk", "formula", "text", "band_gap", "ehull", "is_stable"]
#     }
#     if filter_expr:
#         print(f"🔍 [Filter Active]: {filter_expr}")
#         search_params["filter"] = filter_expr

#     candidates: Dict[str, Dict[str, Any]] = {}

#     # Dense
#     try:
#         dvec = dense_encode_fn(query)  # list[float]
#         res_d = client.search(data=[dvec], anns_field="dense_vec", **search_params)
#         for hit in (res_d[0] if res_d else []):
#             ent = hit.get("entity", {}) or {}
#             pk = ent.get("pk")
#             if pk is not None:
#                 candidates[str(pk)] = ent
#     except Exception as e:
#         print(f"Search Error(Dense): {e}")

#     # Sparse
#     if sparse_encoder is not None:
#         try:
#             st = sparse_encoder.encode(query, convert_to_tensor=True)
#             svec = to_sparse_list(st)
#             res_s = client.search(data=[svec], anns_field="sparse_vec", **search_params)
#             for hit in (res_s[0] if res_s else []):
#                 ent = hit.get("entity", {}) or {}
#                 pk = ent.get("pk")
#                 if pk is not None:
#                     candidates[str(pk)] = ent
#         except Exception as e:
#             print(f"Search Error(Sparse): {e}")

#     cand_list = list(candidates.values())
#     if not cand_list:
#         return []

#     # Rerank
#     pairs = [[query, (d.get("text", "") or "")[:800]] for d in cand_list]
#     scores = rerank_predict_fn(pairs, batch_size=32)

#     final_results = []
#     for doc, score in zip(cand_list, scores):
#         doc["score"] = float(score)
#         doc_formula = str(doc.get("formula", ""))
#         doc["is_exact_match"] = (not filter_expr) and target_formula and (doc_formula.lower() == target_formula.lower())
#         final_results.append(doc)

#     def smart_sort_key(doc):
#         is_exact = 1 if doc.get("is_exact_match") else 0
#         stable = 1 if doc.get("is_stable") is True else 0
#         score = float(doc.get("score", 0.0))

#         if filter_expr:
#             return (stable, score, is_exact)
#         return (is_exact, stable, score)

#     final_results.sort(key=smart_sort_key, reverse=True)
#     return final_results[:top_n]


# def run_retrieval_B_ark(
#     query: str,
#     client: MilvusClient,
#     dense_encode_fn,
#     sparse_encoder,
#     rerank_predict_fn,
#     top_k: int = 3
# ) -> List[Dict[str, Any]]:
#     """
#     完整对齐 B 的路由：
#     1) filter_expr -> 原问题 + filter 检索 top=5
#     2) 否则 formulas -> 每个 formula 检索 top=1
#     3) 否则 docs=[]
#     """
#     filter_expr = parse_filter_expression_ark(query)
#     target_formulas = global_search_planner_ark(query)

#     docs: List[Dict[str, Any]] = []
#     if filter_expr:
#         docs = search_pipeline_B(
#             client, dense_encode_fn, sparse_encoder, rerank_predict_fn,
#             query, filter_expr=filter_expr, top_n=max(5, top_k)
#         )
#     elif target_formulas:
#         for f in target_formulas:
#             docs.extend(
#                 search_pipeline_B(
#                     client, dense_encode_fn, sparse_encoder, rerank_predict_fn,
#                     f, filter_expr="", top_n=1
#                 )
#             )
#     else:
#         docs = []

#     # 去重保序
#     uniq = {}
#     for d in docs:
#         pk = d.get("pk")
#         if pk is not None and str(pk) not in uniq:
#             uniq[str(pk)] = d
#     return list(uniq.values())[:top_k]


# def format_context(docs):
#     if not docs: return "（无检索证据）"
#     parts = []
#     for i, d in enumerate(docs):
#         # 🔥 关键修改：在这里明确标注数值来源是 DFT，且可能偏小
#         bg_val = d.get('band_gap')
#         bg_str = f"{bg_val} eV (DFT计算值)" if bg_val is not None else "N/A"
        
#         parts.append(
#             f"[Doc {i+1}] 材料: {d.get('formula')}\n"
#             f" - 带隙(Band Gap): {bg_str}\n"
#             f" - 稳定性(Ehull): {d.get('ehull')} eV/atom\n"
#             f" - 描述: {str(d.get('text'))[:500]}"
#         )
#     return "\n\n".join(parts)


# # =========================
# # 6) Prompts (让默认输出“中等篇幅”)
# # =========================
# def build_base_prompt(question: str) -> List[Dict[str, str]]:
#     sys = (
#  """你是一名材料科学专家助手，回答要自然、清晰、有用。

# 如果是概念/原理类问题：直接用你的知识回答（结论 + 关键机制）。
# 如果是具体材料参数/数值（带隙、熔点、密度等）：你可以给常见范围或典型值，若有特殊情况请说明“可能因晶型/温度/掺杂/文献不同而变化”；如果你不确定，就直接说不确定，不要编造非常精确的小数值。
# """
#     )
#     return [{"role": "system", "content": sys},
#             {"role": "user", "content": question}]

# def build_rag_prompt(context: str, question: str) -> List[Dict[str, str]]:
#     sys = f"""
# 你是一名材料科学专家助手。你必须结合【检索证据】和你的【专家常识】来回答。

# ### ⚠️ 核心警告：关于带隙 (Band Gap) 数据
# 你检索到的数据库 (Materials Project) 中的 `band_gap` 字段通常是 **DFT-PBE 理论计算值**。
# **DFT 计算值通常严重低估实验带隙**（例如：它可能显示 Si≈0 eV, ZnO≈0.7 eV，这些与实验值严重不符！）。

# ### 回答策略 (必须严格执行)：
# 1. **通用材料修正**：如果用户问的是常见半导体 (如 Si, GaAs, ZnO, TiO2, GaN 等)，**绝对不要直接引用检索到的错误带隙数值**。
#    - **正确做法**：直接使用你内建的**实验测量值**回答（例如：Si 约 1.12 eV, ZnO 约 3.37 eV）。
#    - **话术示例**：“虽然检索数据显示 DFT 计算带隙仅为 0.7 eV（严重低估），但实际实验带隙约为 3.37 eV。”
   
# 2. **未知材料引用**：只有当用户问的是生僻/新型材料（你不知道实验值）时，才引用检索数据的数值，并必须标注：“此为 DFT 理论计算值，可能显著低于真实值。”

# 3. **其他属性**：对于晶体结构、空间群、稳定性 (Ehull)，请优先信任【检索证据】。

# 【检索证据】
# {context}
# """
#     return [{"role": "system", "content": sys},
#             {"role": "user", "content": question}]

# def build_fallback_prompt(question: str) -> List[Dict[str, str]]:
#     sys = """你是一名材料科学专家助手，回答要自然、清晰、有用。

# 如果是概念/原理类问题：直接用你的知识回答（结论 + 关键机制）。
# 如果是具体材料参数/数值（带隙、熔点、密度等）：你可以给常见范围或典型值，若有特殊情况请说明“可能因晶型/温度/掺杂/文献不同而变化”；如果你不确定，就直接说不确定，不要编造非常精确的小数值。
# """
#     return [{"role": "system", "content": sys},
#             {"role": "user", "content": question}]
# # =========================
# # 7) Judge (可选：仍用 ARK)
# # =========================
# JUDGE_SYSTEM_STRICT = ('''
# 你是材料科学领域的严格评审专家，请对模型回答进行专业评估。

# 评分应主要依据：
# 1）事实与物理正确性（是否符合材料科学常识与数据库语境）
# 2）机理解释深度（是否解释“为什么”，而非仅给结论）
# 3）参数与工程意义关联能力
# 4）是否避免编造精确数值并合理说明不确定性
# 5）结构清晰、专业表达自然

# 特别扣分项（严重时可直接低于5分）：
# - 将DFT带隙当作实验真实值使用
# - 错误解释形成能或凸包稳定性
# - 编造看似精确却无依据的材料参数
# - 物理概念混淆（如稳定性、带隙类型、缺陷机制）

# 请输出严格JSON：
# {
#  "accuracy":x,
#  "mechanism_depth":x,
#  "engineering_relevance":x,
#  "completeness":x,
#  "clarity":x,
#  "overall":x,
#  "comment":"一句话指出核心优缺点"
# }

# 评分区间说明：
# 9-10：专业接近教材或综述水平  
# 7-8.9：正确但深度一般  
# 5-6.9：部分正确但解释浅或有模糊  
# <5：存在明显概念或事实错误
# '''
# )
# def normalize_judge(j: Dict[str, Any]) -> Dict[str, Any]:
#     base = {
#         "accuracy": 0.0,
#         "professionalism": 0.0,
#         "logic": 0.0,
#         "completeness": 0.0,
#         "clarity": 0.0,
#         "overall": 0.0,
#         "notes": {
#             "accuracy": "",
#             "professionalism": "",
#             "logic": "",
#             "completeness": "",
#             "clarity": "",
#         },
#         "comment": ""
#     }
#     if not isinstance(j, dict):
#         return base

#     for k in ["accuracy", "professionalism", "logic", "completeness", "clarity", "overall"]:
#         if k in j:
#             try:
#                 base[k] = float(j[k])
#             except:
#                 pass

#     if isinstance(j.get("notes"), dict):
#         for k in base["notes"]:
#             if k in j["notes"]:
#                 base["notes"][k] = str(j["notes"][k])[:80]

#     if "comment" in j:
#         base["comment"] = str(j["comment"])[:240]

#     return base

# def judge_answer(question: str, requirements: str, answer: str) -> Dict[str, Any]:
#     user = (
#         f"【问题】{question}\n"
#         f"【要求】{requirements}\n"
#         f"【模型回答】{answer}\n\n"
#         "请按规则评分并输出JSON。"
#     )
#     out = ark_chat(
#         model=ARK_JUDGE_MODEL,
#         messages=[{"role": "system", "content": JUDGE_SYSTEM_STRICT},
#                   {"role": "user", "content": user}],
#         temperature=0.0,
#         max_tokens=900
#     )
#     obj = _extract_json_obj_or_array(out)
#     if not isinstance(obj, dict):
#         obj = {}
#     return normalize_judge(obj)


# # =========================
# # 8) Question Generation (可选：用 ARK)
# # =========================
# def generate_50_questions() -> List[Dict[str, Any]]:
#     system_prompt = (
#         "你是材料科学评测题库生成器。请生成50个用于评测材料领域LLM的高质量问题，"
#         "覆盖：概念理解、单材料属性分析、多约束筛选、证据驱动推理、工程可行性讨论。"
#         "每题必须可独立回答，不依赖外部链接。"
#         "只输出严格JSON数组，每个元素包含："
#         "{\"id\":\"Q001\",\"category\":\"concept|property|multi_constraint|evidence|engineering\","
#         "\"difficulty\":1-3,\"prompt\":\"...\",\"requirements\":\"...\"}"
#     )
#     user = "生成题库。注意题目要有一定区分度，difficulty=1(20题),2(20题),3(10题)。"
#     out = ark_chat(
#         model=ARK_QGEN_MODEL,
#         messages=[{"role": "system", "content": system_prompt},
#                   {"role": "user", "content": user}],
#         temperature=0.6,
#         max_tokens=4096
#     )
#     obj = _extract_json_obj_or_array(out)
#     if isinstance(obj, list) and len(obj) >= 40:
#         qs = obj[:50]
#         for i, q in enumerate(qs):
#             if "id" not in q:
#                 q["id"] = f"Q{i+1:03d}"
#         return qs
#     raise ValueError("Question generation failed (not a JSON list / too few).")


# # =========================
# # 9) Main Pipeline (A评测脚本，RAG路由换成B逻辑)
# # =========================
# def main():
#     print("🚀 Starting Evaluation Pipeline (A with B-logic RAG)...")
#     os.makedirs("eval_out1", exist_ok=True)

#     # ---- paths check ----
#     _assert_exists(BASE_MODEL_PATH, "BASE_MODEL_PATH")
#     _assert_exists(LORA_PATH, "LORA_PATH")
#     _assert_exists(BGE_DIR, "BGE_DIR")
#     _assert_exists(RERANK_DIR, "RERANK_DIR")
#     if not os.path.exists(SPLADE_DIR):
#         print(f"⚠️ SPLADE_DIR not found: {SPLADE_DIR} (will run dense-only)")

#     # Step1: questions
#     q_file = "eval_out1/questions_50.json"
#     if not os.path.exists(q_file):
#         questions = generate_50_questions()
#         with open(q_file, "w", encoding="utf-8") as f:
#             json.dump(questions, f, ensure_ascii=False, indent=2)
#     else:
#         print(f"📂 Loading existing questions from {q_file}")
#         with open(q_file, "r", encoding="utf-8") as f:
#             questions = json.load(f)

#     # results map
#     results_map = {
#         q["id"]: {
#             "question": q["prompt"],
#             "req": q.get("requirements", ""),
#             "category": q.get("category", "unknown"),
#             "difficulty": q.get("difficulty", 1),
#         } for q in questions
#     }

#     # ==========================================
#     # Phase 1: Base Inference (cache)
#     # ==========================================
#     print("\n" + "=" * 40)
#     print("PHASE 1: Base Model Inference")
#     print("=" * 40)

#     base_cache = "eval_out1/phase1_base.json"
#     if os.path.exists(base_cache):
#         print("⏭️  Skipping Phase 1 (Cache found).")
#         with open(base_cache, "r", encoding="utf-8") as f:
#             phase1_res = json.load(f)
#     else:
#         tok_base, mdl_base = load_base_model_only()
#         phase1_res = {}
#         for i, q in enumerate(questions):
#             print(f"[{i+1}/{len(questions)}] Base -> {q['id']}")
#             ans = local_generate(tok_base, mdl_base, build_base_prompt(q["prompt"]), max_new_tokens=1100, temperature=0.15)
#             phase1_res[q["id"]] = ans

#         with open(base_cache, "w", encoding="utf-8") as f:
#             json.dump(phase1_res, f, ensure_ascii=False, indent=2)

#         del mdl_base, tok_base
#         flush_gpu()

#     for qid, ans in phase1_res.items():
#         if qid in results_map:
#             results_map[qid]["base"] = ans

#     # ==========================================
#     # Phase 2: LoRA & RAG Inference (cache)
#     # ==========================================
#     print("\n" + "=" * 40)
#     print("PHASE 2: LoRA & RAG Inference (RAG uses B logic)")
#     print("=" * 40)

#     lora_cache = "eval_out1/phase2_lora.json"
#     if os.path.exists(lora_cache):
#         print("⏭️  Skipping Phase 2 (Cache found).")
#         with open(lora_cache, "r", encoding="utf-8") as f:
#             phase2_res = json.load(f)
#     else:
#         tok_lora, mdl_lora = load_lora_model_only()
#         client, dense_encode_fn, sparse_encoder, rerank_predict_fn = load_rag_components()

#         phase2_res = {}
#         for i, q in enumerate(questions):
#             qid = q["id"]
#             prompt = q["prompt"]
#             print(f"[{i+1}/{len(questions)}] LoRA/RAG -> {qid}")

#             # A) pure LoRA
#             ans_lora = local_generate(tok_lora, mdl_lora, build_base_prompt(prompt), max_new_tokens=1100, temperature=0.15)

#             docs = run_retrieval_B_ark(
#                 query=prompt,
#                 client=client,
#                 dense_encode_fn=dense_encode_fn,
#                 sparse_encoder=sparse_encoder,
#                 rerank_predict_fn=rerank_predict_fn,
#                 top_k=3
#             )
            
#             # 🔥 核心修改：如果有文档，走 RAG 模式；没文档，走兜底模式
#             if docs:
#                 ctx = format_context(docs)
#                 rag_msgs = build_rag_prompt(ctx, prompt)
#             else:
#                 # 没搜到 -> 启用兜底 Prompt
#                 rag_msgs = build_fallback_prompt(prompt)
            
#             ans_rag = local_generate(tok_lora, mdl_lora, rag_msgs, max_new_tokens=1400, temperature=0.15)
#             phase2_res[qid] = {"lora": ans_lora, "rag": ans_rag}

#         with open(lora_cache, "w", encoding="utf-8") as f:
#             json.dump(phase2_res, f, ensure_ascii=False, indent=2)

#         del mdl_lora, tok_lora
#         flush_gpu()

#     for qid, res in phase2_res.items():
#         if qid in results_map:
#             results_map[qid]["lora"] = res["lora"]
#             results_map[qid]["lora_rag"] = res["rag"]

#     # ==========================================
#     # Phase 3: Judge (resume + clean)
#     # ==========================================
#     print("\n" + "=" * 40)
#     print("PHASE 3: AI Judge Grading (With Resume)")
#     print("=" * 40)

#     csv_path = "eval_out1" \
#     "/final_report.csv"
#     headers = [
#         "id", "category", "difficulty", "model",
#         "accuracy", "professionalism", "logic", "completeness", "clarity", "overall",
#         "n_accuracy", "n_professionalism", "n_logic", "n_completeness", "n_clarity",
#         "comment"
#     ]

#     completed_keys = set()
#     valid_rows = []
#     if os.path.exists(csv_path):
#         print(f"🔄 Found existing report: {csv_path}, loading progress...")
#         with open(csv_path, "r", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 try:
#                     score = float(row.get("overall", 0))
#                 except:
#                     score = 0.0
#                 if score > 0.1:
#                     key = (row["id"], row["model"])
#                     completed_keys.add(key)
#                     valid_rows.append(row)
#         print(f"✅ Valid graded rows: {len(completed_keys)}")

#     with open(csv_path, "w", encoding="utf-8", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(headers)
#         for row in valid_rows:
#             w.writerow([row.get(h, "") for h in headers])
#         f.flush()

#         for qid, data in results_map.items():
#             q_prompt = data["question"]
#             q_req = data["req"]
#             q_cat = data.get("category", "unknown")
#             q_diff = data.get("difficulty", 1)

#             for m_name in ["base", "lora", "lora_rag"]:
#                 if (qid, m_name) in completed_keys:
#                     print(f"⏩ Skip {qid} [{m_name}] (Already Valid)")
#                     continue

#                 ans = data.get(m_name, "")
#                 if not ans or len(ans) < 5:
#                     print(f"⚠️ Empty answer for {qid} [{m_name}] -> skip")
#                     continue

#                 print(f"⚖️ Judging {qid} [{m_name}] ...")
#                 judged = judge_answer(q_prompt, q_req, ans)

#                 w.writerow([
#                     qid, q_cat, q_diff, m_name,
#                     judged["accuracy"],
#                     judged["professionalism"],
#                     judged["logic"],
#                     judged["completeness"],
#                     judged["clarity"],
#                     judged["overall"],
#                     judged["notes"]["accuracy"],
#                     judged["notes"]["professionalism"],
#                     judged["notes"]["logic"],
#                     judged["notes"]["completeness"],
#                     judged["notes"]["clarity"],
#                     judged["comment"],
#                 ])
#                 f.flush()
#                 time.sleep(1.2)

#     print(f"\n🎉 Done! Final report: {csv_path}")


# if __name__ == "__main__":
#     main()
import os
import json
import time
import csv
import re
import gc
from typing import List, Dict, Any, Optional

import requests
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoTokenizer as HFTokenizer,
    AutoModel, AutoModelForSequenceClassification
)
from peft import PeftModel
from sentence_transformers import SparseEncoder
from pymilvus import MilvusClient

# =========================
# 0) ENV & CONFIG
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "1")
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "1")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf_cache")

BASE_MODEL_PATH = "./qwen_model/Qwen/Qwen2___5-7B-Instruct"
LORA_PATH = "./lora_stage1"

MILVUS_URI = "./milvus_lite.db"
COLLECTION = "materials_hybrid"

BGE_DIR = "./hf_cache/transformers/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
SPLADE_DIR = "./hf_cache/transformers/models--naver--splade-cocondenser-ensembledistil/snapshots/49cf4c7b0db5b870a401ddf5e2669993ef3699c7"
RERANK_DIR = "./hf_cache/transformers/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70"

ARK_API_KEY = os.getenv("ARK_API_KEY", "1550c65b-2643-4c98-9c89-e63c2762cbe8")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
ARK_JUDGE_MODEL = os.getenv("ARK_JUDGE_MODEL", "ep-20260110133352-bczxh")
ARK_QGEN_MODEL = os.getenv("ARK_QGEN_MODEL", ARK_JUDGE_MODEL)
ROUTER_MODEL = os.getenv("ROUTER_MODEL", ARK_JUDGE_MODEL)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# =========================
# 1) Utils
# =========================
def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 GPU Memory Flushed.")

def _assert_exists(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{name}] Path not found: {path}")

def ark_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 512) -> str:
    if not ARK_API_KEY:
        raise RuntimeError("Missing env ARK_API_KEY")
    if not model:
        raise RuntimeError("Missing ARK model endpoint (ep-xxxx)")

    headers = {"Authorization": f"Bearer {ARK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(ARK_BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            print(f"ARK API Error (attempt {attempt+1}/3): {e}")
            time.sleep(1.5)

    print(f"ARK API Failed: {last_err}")
    return "{}"

def _extract_json_obj(text: str) -> Optional[dict]:
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    raw = m.group(0)
    raw = raw.replace("“", "\"").replace("”", "\"").replace("，", ",")
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except:
        return None

# --- 化学式规范化（把 nacl -> NaCl 纠正） ---
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
_ELEMENT_SYMBOLS = {
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
}

def normalize_formula_guess(token: str) -> Optional[str]:
    if not token:
        return None
    s = re.sub(r"[^A-Za-z0-9]", "", token.strip())
    if not s:
        return None
    if _FORMULA_RE.fullmatch(s):
        return s
    if s.isdigit() or len(s) > 32:
        return None

    i = 0
    out = []
    while i < len(s):
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            out.append(s[i:j])
            i = j
            continue

        sym = None
        if i + 2 <= len(s):
            cand2 = s[i:i+2]
            c2 = cand2[0].upper() + cand2[1].lower()
            if c2 in _ELEMENT_SYMBOLS:
                sym = c2
                i += 2
        if sym is None:
            cand1 = s[i:i+1]
            c1 = cand1[0].upper()
            if c1 in _ELEMENT_SYMBOLS:
                sym = c1
                i += 1
        if sym is None:
            return None
        out.append(sym)

    f = "".join(out)
    return f if _FORMULA_RE.fullmatch(f) else None

# =========================
# 2) Offline Dense/Sparse/Rerank
# =========================
def build_dense_encoder():
    tok = HFTokenizer.from_pretrained(BGE_DIR, local_files_only=True)
    mdl = AutoModel.from_pretrained(BGE_DIR, local_files_only=True).to("cpu").eval()

    @torch.no_grad()
    def encode(text: str) -> List[float]:
        inp = tok([text], padding=True, truncation=True, return_tensors="pt").to("cpu")
        out = mdl(**inp)
        last = out.last_hidden_state
        mask = inp["attention_mask"].unsqueeze(-1).float()
        emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = F.normalize(emb, p=2, dim=1)
        return emb[0].cpu().tolist()

    return encode

def build_reranker():
    tok = HFTokenizer.from_pretrained(RERANK_DIR, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(RERANK_DIR, local_files_only=True).to("cpu").eval()

    @torch.no_grad()
    def predict(pairs: List[List[str]], batch_size: int = 16) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            qs = [x[0] for x in batch]
            ds = [x[1] for x in batch]
            inp = tok(qs, ds, padding=True, truncation=True, return_tensors="pt").to("cpu")
            out = mdl(**inp)
            logits = out.logits
            if logits.shape[-1] == 1:
                s = logits.squeeze(-1)
            else:
                s = logits[:, -1]
            scores.extend(s.detach().cpu().tolist())
        return scores

    return predict

def to_sparse_list(vec):
    vec = vec.cpu()
    if vec.is_sparse:
        vec = vec.coalesce()
        idx = vec.indices()[0].tolist()
        val = vec.values().tolist()
    else:
        nz = (vec != 0).nonzero(as_tuple=False).flatten()
        idx = nz.tolist()
        val = vec[nz].tolist()
    return [(int(i), float(x)) for i, x in zip(idx, val)]

# =========================
# 3) Model Loaders
# =========================
def load_base_model_only():
    print("🚀 Loading BASE Model...")
    _assert_exists(BASE_MODEL_PATH, "BASE_MODEL_PATH")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    return tokenizer, model

def load_lora_model_only():
    print("🚀 Loading LoRA Model (Base + Adapter)...")
    _assert_exists(LORA_PATH, "LORA_PATH")
    tokenizer, base_model = load_base_model_only()
    print(f"   + Attaching LoRA adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH, local_files_only=True).eval()
    return tokenizer, model

def load_rag_components():
    print("📚 Loading RAG Retrievers (CPU Mode)...")
    try:
        client = MilvusClient(uri=MILVUS_URI)
    except Exception as e:
        print(f"⚠️ Milvus Load Error: {e}. RAG disabled.")
        client = None

    dense_encode = build_dense_encoder()

    try:
        sparse_encoder = SparseEncoder(SPLADE_DIR, device="cpu", trust_remote_code=True, local_files_only=True)
    except Exception as e:
        print(f"⚠️ SPLADE Load Error: {e}. Dense-only.")
        sparse_encoder = None

    rerank_predict = build_reranker()
    return client, dense_encode, sparse_encoder, rerank_predict

# =========================
# 4) Generation (Local LLM)
# =========================
@torch.no_grad()
def local_generate(tokenizer, model, messages, max_new_tokens=1024, temperature=0.1):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        repetition_penalty=1.1,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()

# =========================
# 5) (你的检索逻辑保持不变：略)
#     —— 这里保留你原来的 parse_filter_expression_ark / global_search_planner_ark
#     —— search_pipeline_B / run_retrieval_B_ark / format_context
# =========================

def parse_filter_expression_ark(query: str) -> str:
    sys_prompt = (
        "你是一个 Milvus 数据库查询专家。请将用户的自然语言筛选要求转换为 Milvus 的 filter 表达式。\n"
        "### 数据库 Schema 定义\n"
        "- band_gap (float): 带隙 eV\n"
        "- ehull (float): eV/atom\n"
        "- is_stable (bool): 是否稳定\n"
        "- formula (str): 化学式\n\n"
        "### 转换规则\n"
        "1) 只输出表达式字符串，不要 markdown，不要解释。\n"
        "2) 允许 && || (Milvus expr 支持)。\n"
        "3) 若没有明确筛选要求（只是问概念/单材料属性），输出空字符串。\n"
        "示例：\n"
        "- 带隙大于1.5 -> band_gap > 1.5\n"
        "- 稳定的且带隙小于1 -> is_stable == true && band_gap < 1.0\n"
    )
    resp = ark_chat(
        ROUTER_MODEL,
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": query}],
        temperature=0.0,
        max_tokens=128
    ).strip()

    expr = resp.replace("`", "").replace('"', "").replace("'", "").strip()
    if (not expr) or expr.startswith("{") or expr.startswith("[") or "（" in expr or "）" in expr:
        return ""
    if not re.fullmatch(r"[a-zA-Z0-9_ \.\<\>\=\!\&\|\(\)]+", expr):
        return ""
    if ("band_gap" not in expr) and ("ehull" not in expr) and ("is_stable" not in expr) and ("formula" not in expr):
        return ""
    return expr

def _is_valid_formula_token(s: str) -> bool:
    if not s or len(s) > 32:
        return False
    return re.fullmatch(r"[A-Za-z0-9\(\)\.\-\u00B7]+", s) is not None

def global_search_planner_ark(query: str) -> List[str]:
    sys_prompt = (
        "你是材料实体标准化器。任务：从用户问题中提取“需要去材料数据库检索的材料化学式”。\n"
        "只输出一个 JSON 数组（array），不要输出任何解释。\n"
        "数组元素为对象：{\"formula\": \"...\", \"confidence\": 0~1}\n"
        "只在你有把握时才填 formula；不确定就 formula:\"\"，confidence<=0.4。禁止瞎编。\n"
        "若问题是概念/原理/闲聊，输出空数组 []。\n"
        "最多输出 4 个对象。\n"
        "输出示例：[{\"formula\":\"GaAs\",\"confidence\":0.95}]"
    )
    user_prompt = f"用户问题：{query}\n请输出 JSON 数组："
    for attempt in range(3):
        try:
            resp = ark_chat(
                ROUTER_MODEL,
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=256
            )
            m = re.search(r"\[[\s\S]*\]", resp or "")
            if not m:
                raise ValueError("no json array")
            items = json.loads(m.group(0))

            formulas: List[str] = []
            for it in items[:6]:
                f = (it.get("formula") or "").strip()
                try:
                    conf = float(it.get("confidence", 0.0) or 0.0)
                except:
                    conf = 0.0
                if (not f) or (conf < 0.55):
                    continue
                f2 = normalize_formula_guess(f) or f
                if not _is_valid_formula_token(f2):
                    continue
                formulas.append(f2)

            uniq = []
            for x in formulas:
                if x not in uniq:
                    uniq.append(x)
            return uniq[:4]
        except Exception as e:
            print(f"Planner Error attempt {attempt+1}/3: {e}")
            time.sleep(0.8)
    return []

def search_pipeline_B(
    client: MilvusClient,
    dense_encode_fn,
    sparse_encoder,
    rerank_predict_fn,
    query: str,
    filter_expr: str = "",
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    if client is None:
        return []

    target_formula = normalize_formula_guess(query.strip())
    search_params = {
        "collection_name": COLLECTION,
        "limit": 60,
        "output_fields": ["pk", "formula", "text", "band_gap", "ehull", "is_stable"]
    }
    if filter_expr:
        print(f"🔍 [Filter Active]: {filter_expr}")
        search_params["filter"] = filter_expr

    candidates: Dict[str, Dict[str, Any]] = {}

    try:
        dvec = dense_encode_fn(query)
        res_d = client.search(data=[dvec], anns_field="dense_vec", **search_params)
        for hit in (res_d[0] if res_d else []):
            ent = hit.get("entity", {}) or {}
            pk = ent.get("pk")
            if pk is not None:
                candidates[str(pk)] = ent
    except Exception as e:
        print(f"Search Error(Dense): {e}")

    if sparse_encoder is not None:
        try:
            st = sparse_encoder.encode(query, convert_to_tensor=True)
            svec = to_sparse_list(st)
            res_s = client.search(data=[svec], anns_field="sparse_vec", **search_params)
            for hit in (res_s[0] if res_s else []):
                ent = hit.get("entity", {}) or {}
                pk = ent.get("pk")
                if pk is not None:
                    candidates[str(pk)] = ent
        except Exception as e:
            print(f"Search Error(Sparse): {e}")

    cand_list = list(candidates.values())
    if not cand_list:
        return []

    pairs = [[query, (d.get("text", "") or "")[:800]] for d in cand_list]
    scores = rerank_predict_fn(pairs, batch_size=32)

    final_results = []
    for doc, score in zip(cand_list, scores):
        doc["score"] = float(score)
        doc_formula = str(doc.get("formula", ""))
        doc["is_exact_match"] = (not filter_expr) and target_formula and (doc_formula.lower() == target_formula.lower())
        final_results.append(doc)

    def smart_sort_key(doc):
        is_exact = 1 if doc.get("is_exact_match") else 0
        stable = 1 if doc.get("is_stable") is True else 0
        score = float(doc.get("score", 0.0))
        if filter_expr:
            return (stable, score, is_exact)
        return (is_exact, stable, score)

    final_results.sort(key=smart_sort_key, reverse=True)
    return final_results[:top_n]

def run_retrieval_B_ark(
    query: str,
    client: MilvusClient,
    dense_encode_fn,
    sparse_encoder,
    rerank_predict_fn,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    filter_expr = parse_filter_expression_ark(query)
    target_formulas = global_search_planner_ark(query)

    docs: List[Dict[str, Any]] = []
    if filter_expr:
        docs = search_pipeline_B(
            client, dense_encode_fn, sparse_encoder, rerank_predict_fn,
            query, filter_expr=filter_expr, top_n=max(5, top_k)
        )
    elif target_formulas:
        for f in target_formulas:
            docs.extend(
                search_pipeline_B(
                    client, dense_encode_fn, sparse_encoder, rerank_predict_fn,
                    f, filter_expr="", top_n=1
                )
            )
    else:
        docs = []

    uniq = {}
    for d in docs:
        pk = d.get("pk")
        if pk is not None and str(pk) not in uniq:
            uniq[str(pk)] = d
    return list(uniq.values())[:top_k]

def format_context(docs):
    if not docs:
        return "（无检索证据）"
    parts = []
    for i, d in enumerate(docs):
        bg_val = d.get('band_gap')
        bg_str = f"{bg_val} eV (DFT计算值)" if bg_val is not None else "N/A"
        parts.append(
            f"[Doc {i+1}] 材料: {d.get('formula')}\n"
            f" - 带隙(Band Gap): {bg_str}\n"
            f" - 稳定性(Ehull): {d.get('ehull')} eV/atom\n"
            f" - 描述: {str(d.get('text'))[:500]}"
        )
    return "\n\n".join(parts)

# =========================
# 6) Prompts
# =========================
def build_base_prompt(question: str) -> List[Dict[str, str]]:
    sys = (
        "你是一名材料科学专家助手，回答要自然、清晰、有用。\n\n"
        "如果是概念/原理类问题：直接用你的知识回答（结论 + 关键机制）。\n"
        "如果是具体材料参数/数值（晶格常数、空间群、稳定性等）：\n"
        " - 能确定就给出典型值或范围，并说明可能因晶型/温度/文献不同而变化；\n"
        " - 不确定就明确说不确定，不要编造非常精确的小数值。\n"
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": question}]

def build_rag_prompt(context: str, question: str) -> List[Dict[str, str]]:
    sys = f"""
你是一名材料科学专家助手。你必须结合【检索证据】和你的【专家常识】来回答。
回答中需要明确哪些结论来自证据（例如引用 Doc 编号），哪些是常识推断。

【检索证据】
{context}
"""
    return [{"role": "system", "content": sys},
            {"role": "user", "content": question}]

def build_fallback_prompt(question: str) -> List[Dict[str, str]]:
    return build_base_prompt(question)

# =========================
# 7) Judge（改为三答案对比式）
# =========================
JUDGE_SYSTEM_COMPARE = r"""
你是材料科学领域的严格评审专家。你将同时看到同一道题的三个回答：BASE、LORA、LORA_RAG。
请用“同一把尺子”横向对比评分，并强制拉开差距：有证据且正确的回答应明显更高。

【核心评分维度】每项 0-10：
1) accuracy：事实与物理正确性（最重要）
2) evidence_use：证据使用质量（是否引用/对齐证据；没有证据却给精确数值要重罚）
3) mechanism_depth：机理/原因解释深度（不是堆话）
4) completeness：是否满足题目要求（是否给全字段/结构化输出）
5) clarity：结构清晰、表述专业自然

【硬性扣分规则（非常重要）】
A. 如果回答给出晶格常数/空间群/ehull等“精确数值”，但回答中没有任何可追溯依据（如证据片段/Doc引用/明确来源），accuracy 必须大幅扣分（通常 <=5）。
B. 如果回答明显编造精确数值（看起来很像真的，但无依据），accuracy 可直接 <=4。
C. 如果回答承认“不确定/需查证”，不算幻觉，但 completeness 会扣分（通常 5-7）。
D. LORA_RAG 若能引用证据（如 Doc 1/2/3）并与字段一致，应显著加分。

【输出格式】
只输出严格 JSON 对象（不要 markdown/解释）：
{
  "base":    {"accuracy":x,"evidence_use":x,"mechanism_depth":x,"completeness":x,"clarity":x,"overall":x,"comment":"一句话点评"},
  "lora":    {"accuracy":x,"evidence_use":x,"mechanism_depth":x,"completeness":x,"clarity":x,"overall":x,"comment":"一句话点评"},
  "lora_rag":{"accuracy":x,"evidence_use":x,"mechanism_depth":x,"completeness":x,"clarity":x,"overall":x,"comment":"一句话点评"},
  "rank": ["lora_rag","lora","base"],
  "reason": "一句话说明排序依据"
}

overall 不是平均分，而是综合印象分：以 accuracy 为主，其次 evidence_use / completeness。
"""

def _clamp01_10(x: Any) -> float:
    try:
        v = float(x)
    except:
        return 0.0
    if v < 0: v = 0.0
    if v > 10: v = 10.0
    return v

def normalize_model_score(obj: Any) -> Dict[str, Any]:
    base = {
        "accuracy": 0.0,
        "evidence_use": 0.0,
        "mechanism_depth": 0.0,
        "completeness": 0.0,
        "clarity": 0.0,
        "overall": 0.0,
        "comment": ""
    }
    if not isinstance(obj, dict):
        return base
    for k in ["accuracy","evidence_use","mechanism_depth","completeness","clarity","overall"]:
        if k in obj:
            base[k] = _clamp01_10(obj[k])
    if "comment" in obj:
        base["comment"] = str(obj["comment"])[:240]
    return base

def judge_triplet(question: str, requirements: str, ans_base: str, ans_lora: str, ans_rag: str) -> Dict[str, Any]:
    user = (
        f"【问题】{question}\n"
        f"【要求】{requirements}\n\n"
        f"【BASE回答】\n{ans_base}\n\n"
        f"【LORA回答】\n{ans_lora}\n\n"
        f"【LORA_RAG回答】\n{ans_rag}\n\n"
        "请按规则输出JSON。"
    )
    out = ark_chat(
        model=ARK_JUDGE_MODEL,
        messages=[{"role": "system", "content": JUDGE_SYSTEM_COMPARE},
                  {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=1200
    )
    obj = _extract_json_obj(out) or {}
    res = {
        "base": normalize_model_score(obj.get("base")),
        "lora": normalize_model_score(obj.get("lora")),
        "lora_rag": normalize_model_score(obj.get("lora_rag")),
        "rank": obj.get("rank", []),
        "reason": str(obj.get("reason", ""))[:240],
    }
    return res

# =========================
# 8) Question Generation（你说不用管，这里保留原函数但不会强依赖）
# =========================
def generate_50_questions() -> List[Dict[str, Any]]:
    system_prompt = (
        "你是材料科学评测题库生成器。请生成50个用于评测材料领域LLM的高质量问题，"
        "覆盖：概念理解、单材料属性分析、多约束筛选、证据驱动推理、工程可行性讨论。"
        "每题必须可独立回答，不依赖外部链接。"
        "只输出严格JSON数组，每个元素包含："
        "{\"id\":\"Q001\",\"category\":\"concept|property|multi_constraint|evidence|engineering\","
        "\"difficulty\":1-3,\"prompt\":\"...\",\"requirements\":\"...\"}"
    )
    user = "生成题库。注意题目要有一定区分度，difficulty=1(20题),2(20题),3(10题)。"
    out = ark_chat(
        model=ARK_QGEN_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user}],
        temperature=0.6,
        max_tokens=4096
    )
    obj = _extract_json_obj(out)
    if isinstance(obj, list) and len(obj) >= 40:
        qs = obj[:50]
        for i, q in enumerate(qs):
            if "id" not in q:
                q["id"] = f"Q{i+1:03d}"
        return qs
    raise ValueError("Question generation failed (not a JSON list / too few).")

# =========================
# 9) Main Pipeline
# =========================
def main():
    print("🚀 Starting Evaluation Pipeline (Compare-Judge)...")
    os.makedirs("eval_out1", exist_ok=True)

    _assert_exists(BASE_MODEL_PATH, "BASE_MODEL_PATH")
    _assert_exists(LORA_PATH, "LORA_PATH")
    _assert_exists(BGE_DIR, "BGE_DIR")
    _assert_exists(RERANK_DIR, "RERANK_DIR")
    if not os.path.exists(SPLADE_DIR):
        print(f"⚠️ SPLADE_DIR not found: {SPLADE_DIR} (will run dense-only)")

    # Step1: questions
    q_file = "eval_out1/questions_50.json"
    if not os.path.exists(q_file):
        questions = generate_50_questions()
        with open(q_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
    else:
        print(f"📂 Loading existing questions from {q_file}")
        with open(q_file, "r", encoding="utf-8") as f:
            questions = json.load(f)

    results_map = {
        q["id"]: {
            "question": q["prompt"],
            "req": q.get("requirements", ""),
            "category": q.get("category", "unknown"),
            "difficulty": q.get("difficulty", 1),
        } for q in questions
    }

    # ==========================================
    # Phase 1: Base Inference (cache)
    # ==========================================
    print("\n" + "=" * 40)
    print("PHASE 1: Base Model Inference")
    print("=" * 40)

    base_cache = "eval_out1/phase1_base.json"
    if os.path.exists(base_cache):
        print("⏭️  Skipping Phase 1 (Cache found).")
        with open(base_cache, "r", encoding="utf-8") as f:
            phase1_res = json.load(f)
    else:
        tok_base, mdl_base = load_base_model_only()
        phase1_res = {}
        for i, q in enumerate(questions):
            print(f"[{i+1}/{len(questions)}] Base -> {q['id']}")
            ans = local_generate(tok_base, mdl_base, build_base_prompt(q["prompt"]), max_new_tokens=1100, temperature=0.15)
            phase1_res[q["id"]] = ans
        with open(base_cache, "w", encoding="utf-8") as f:
            json.dump(phase1_res, f, ensure_ascii=False, indent=2)
        del mdl_base, tok_base
        flush_gpu()

    for qid, ans in phase1_res.items():
        if qid in results_map:
            results_map[qid]["base"] = ans

    # ==========================================
    # Phase 2: LoRA & RAG Inference (cache)
    # ==========================================
    print("\n" + "=" * 40)
    print("PHASE 2: LoRA & RAG Inference")
    print("=" * 40)

    lora_cache = "eval_out1/phase2_lora.json"
    if os.path.exists(lora_cache):
        print("⏭️  Skipping Phase 2 (Cache found).")
        with open(lora_cache, "r", encoding="utf-8") as f:
            phase2_res = json.load(f)
    else:
        tok_lora, mdl_lora = load_lora_model_only()
        client, dense_encode_fn, sparse_encoder, rerank_predict_fn = load_rag_components()

        phase2_res = {}
        for i, q in enumerate(questions):
            qid = q["id"]
            prompt = q["prompt"]
            print(f"[{i+1}/{len(questions)}] LoRA/RAG -> {qid}")

            ans_lora = local_generate(tok_lora, mdl_lora, build_base_prompt(prompt), max_new_tokens=1100, temperature=0.15)

            docs = run_retrieval_B_ark(
                query=prompt,
                client=client,
                dense_encode_fn=dense_encode_fn,
                sparse_encoder=sparse_encoder,
                rerank_predict_fn=rerank_predict_fn,
                top_k=3
            )

            if docs:
                ctx = format_context(docs)
                rag_msgs = build_rag_prompt(ctx, prompt)
            else:
                rag_msgs = build_fallback_prompt(prompt)

            ans_rag = local_generate(tok_lora, mdl_lora, rag_msgs, max_new_tokens=1400, temperature=0.15)
            phase2_res[qid] = {"lora": ans_lora, "rag": ans_rag}

        with open(lora_cache, "w", encoding="utf-8") as f:
            json.dump(phase2_res, f, ensure_ascii=False, indent=2)

        del mdl_lora, tok_lora
        flush_gpu()

    for qid, res in phase2_res.items():
        if qid in results_map:
            results_map[qid]["lora"] = res["lora"]
            results_map[qid]["lora_rag"] = res["rag"]

    # ==========================================
    # Phase 3: Compare-Judge (resume)
    # ==========================================
    print("\n" + "=" * 40)
    print("PHASE 3: Compare-Judge Grading (One call per question)")
    print("=" * 40)

    csv_path = "eval_out1/final_report.csv"
    headers = [
        "id", "category", "difficulty", "model",
        "accuracy", "evidence_use", "mechanism_depth", "completeness", "clarity", "overall",
        "comment",
        "rank_list", "rank_reason"
    ]

    completed_qids = set()
    kept_rows = []
    if os.path.exists(csv_path):
        print(f"🔄 Found existing report: {csv_path}, loading progress...")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 只要这个题至少有一行 overall>0.1，就认为该题完成（因为我们现在是一题三行）
                try:
                    ov = float(row.get("overall", 0))
                except:
                    ov = 0.0
                if ov > 0.1:
                    completed_qids.add(row["id"])
                kept_rows.append(row)
        print(f"✅ Completed questions (detected): {len(completed_qids)}")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        # 先写回旧行（可选：你也可以不保留旧行，直接注释掉这段）
        for row in kept_rows:
            w.writerow([row.get(h, "") for h in headers])
        f.flush()

        for qid, data in results_map.items():
            if qid in completed_qids:
                print(f"⏩ Skip {qid} (Already Completed)")
                continue

            q_prompt = data["question"]
            q_req = data["req"]
            q_cat = data.get("category", "unknown")
            q_diff = data.get("difficulty", 1)

            ans_base = data.get("base", "")
            ans_lora = data.get("lora", "")
            ans_rag = data.get("lora_rag", "")

            if min(len(ans_base), len(ans_lora), len(ans_rag)) < 5:
                print(f"⚠️ Missing answer(s) for {qid} -> skip")
                continue

            print(f"⚖️ Compare-Judging {qid} ...")
            judged = judge_triplet(q_prompt, q_req, ans_base, ans_lora, ans_rag)

            rank_list = judged.get("rank", [])
            rank_reason = judged.get("reason", "")

            for model_key in ["base", "lora", "lora_rag"]:
                sc = judged[model_key]
                w.writerow([
                    qid, q_cat, q_diff, model_key,
                    sc["accuracy"],
                    sc["evidence_use"],
                    sc["mechanism_depth"],
                    sc["completeness"],
                    sc["clarity"],
                    sc["overall"],
                    sc["comment"],
                    json.dumps(rank_list, ensure_ascii=False),
                    rank_reason
                ])
                f.flush()

            time.sleep(1.2)

    print(f"\n🎉 Done! Final report: {csv_path}")


if __name__ == "__main__":
    main()
