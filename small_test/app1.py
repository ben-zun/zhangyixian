# import os
# import re
# import json
# import torch
# import chainlit as cl
# from typing import List, Dict, Any, Optional
# from threading import Thread
# import requests

# # ================= 🚑 环境变量 =================
# os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "./hf_cache"

# from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, AutoModel, AutoModelForSequenceClassification
# from peft import PeftModel
# from sentence_transformers import SparseEncoder
# from transformers import AutoTokenizer as HFTokenizer
# import torch.nn.functional as F
# from pymilvus import MilvusClient

# # ================= ⚙️ 配置区 =================
# # 本地模型路径
# BASE_MODEL_PATH = "./qwen_model/Qwen/Qwen2___5-7B-Instruct"
# LORA_PATH = "./lora_test"

# # RAG 模型路径
# BGE_DIR = "./hf_cache/transformers/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
# SPLADE_DIR = "./hf_cache/transformers/models--naver--splade-cocondenser-ensembledistil/snapshots/49cf4c7b0db5b870a401ddf5e2669993ef3699c7"
# RERANK_DIR = "./hf_cache/transformers/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70"

# # 数据库
# MILVUS_URI = "./milvus_lite.db"
# COLLECTION = "materials_hybrid"

# # API 配置 (用于意图识别)
# ARK_API_KEY = "1550c65b-2643-4c98-9c89-e63c2762cbe8"
# ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
# ROUTER_MODEL = "ep-20260110133352-bczxh" 

# # ================= 🧠 模型加载 (Global) =================
# print("🚀 初始化: 正在加载大模型...")
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, local_files_only=True)
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL_PATH,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     local_files_only=True,
# )
# model = PeftModel.from_pretrained(base_model, LORA_PATH, local_files_only=True)
# model.eval()

# print("📚 初始化: 正在加载 RAG 组件 (CPU)...")
# # Milvus
# try:
#     client = MilvusClient(uri=MILVUS_URI)
# except Exception as e:
#     print(f"⚠️ Milvus 警告: {e}")
#     client = None

# # Dense (BGE)
# bge_tok = HFTokenizer.from_pretrained(BGE_DIR, local_files_only=True)
# bge_model = AutoModel.from_pretrained(BGE_DIR, local_files_only=True).to("cpu").eval()

# # Sparse (SPLADE)
# try:
#     sparse_encoder = SparseEncoder(SPLADE_DIR, device="cpu", trust_remote_code=True, local_files_only=True)
# except:
#     print("⚠️ SPLADE 加载失败，将降级为仅 Dense 检索")
#     sparse_encoder = None

# # Rerank
# rerank_tok = HFTokenizer.from_pretrained(RERANK_DIR, local_files_only=True)
# rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_DIR, local_files_only=True).to("cpu").eval()

# print("✅ 所有模型加载完毕！")

# # ================= 🛠️ 核心工具函数 =================

# def ark_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
#     """调用豆包 API 进行意图识别"""
#     headers = {"Authorization": f"Bearer {ARK_API_KEY}", "Content-Type": "application/json"}
#     payload = {"model": model, "messages": messages, "temperature": temperature}
#     try:
#         r = requests.post(ARK_BASE_URL, headers=headers, json=payload, timeout=5)
#         r.raise_for_status()
#         return r.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         print(f"API Error: {e}")
#         return "{}"

# def extract_json(text: str) -> Dict:
#     try:
#         m = re.search(r"\{.*\}", text, re.DOTALL)
#         if m: return json.loads(m.group(0))
#         return json.loads(text)
#     except:
#         return {}

# @torch.no_grad()
# def dense_encode(texts):
#     if isinstance(texts, str): texts = [texts]
#     inp = bge_tok(texts, padding=True, truncation=True, return_tensors="pt").to("cpu")
#     out = bge_model(**inp)
#     emb = (out.last_hidden_state * inp["attention_mask"].unsqueeze(-1)).sum(1) / inp["attention_mask"].sum(1).clamp(min=1e-9)
#     return F.normalize(emb, p=2, dim=1).cpu().numpy()

# def to_sparse_list(vec):
#     vec = vec.cpu()
#     if vec.is_sparse: vec = vec.coalesce()
#     idx = vec.indices()[0].tolist()
#     val = vec.values().tolist()
#     return [(int(i), float(x)) for i, x in zip(idx, val)]

# @torch.no_grad()
# def rerank_predict(pairs, batch_size=16):
#     scores = []
#     for i in range(0, len(pairs), batch_size):
#         batch = pairs[i:i+batch_size]
#         qs = [x[0] for x in batch]
#         ds = [x[1] for x in batch]
#         inp = rerank_tok(qs, ds, padding=True, truncation=True, return_tensors="pt").to("cpu")
#         out = rerank_model(**inp)
#         logits = out.logits
#         s = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, -1]
#         scores.extend(s.detach().cpu().tolist())
#     return scores

# def clean_latex(text: str) -> str:
#     text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
#     return re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

# # ================= 🧭 核心逻辑：意图识别 + 智能检索 =================

# async def analyze_intent(query: str) -> Dict[str, Any]:
#     """
#     判断是“查数据”还是“问概念”。如果是查数据，提取化学式。
#     """
#     # 极简 Prompt，速度快
#     sys_prompt = (
#         "你是材料科学助手。判断用户意图：\n"
#         "1. intent: 'data' (查具体材料参数/结构)，'concept' (问定义/原理/闲聊)。\n"
#         "2. formula: 提取目标材料的化学式（如 NaCl, Si, GaAs）。若无具体材料或为纯概念问题，返回 null。\n"
#         "返回JSON: {\"intent\": \"...\", \"formula\": \"...\"}"
#     )
    
#     # 异步调用防止阻塞
#     resp = await cl.make_async(ark_chat)(ROUTER_MODEL, [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}])
#     res_json = extract_json(resp)
    
#     # 兜底：解析失败默认走检索，但没有 formula
#     return {
#         "intent": res_json.get("intent", "data"), 
#         "formula": res_json.get("formula")
#     }

# @cl.step(type="tool", name="Smart Retrieval")
# async def search_pipeline(query: str, top_n: int = 3):
#     if not client: return []

#     # 1. 意图识别
#     analysis = await analyze_intent(query)
#     intent = analysis["intent"]
#     target_formula = analysis["formula"]
    
#     print(f"🤖 意图分析: Intent={intent}, Formula={target_formula}")

#     # 🌟 策略 A：如果是纯概念问题（且没提到具体材料），直接跳过 RAG
#     if intent == "concept" and not target_formula:
#         print("   -> 概念类问题，跳过检索")
#         return []

#     # 🌟 策略 B：构造检索词
#     # 如果提取到了 formula，把 formula 加权进 query
#     search_query = f"{target_formula} {query}" if target_formula else query
    
#     # 2. 检索 (Dense + Sparse)
#     dense_vec = (await cl.make_async(dense_encode)(search_query))[0].tolist()
    
#     candidates = {} # pk -> doc
    
#     # Dense Search
#     res_d = await cl.make_async(client.search)(
#         collection_name=COLLECTION, data=[dense_vec], anns_field="dense_vec",
#         limit=60, output_fields=["pk", "formula", "text", "band_gap", "ehull"]
#     )
#     for hit in res_d[0]:
#         candidates[hit["entity"]["pk"]] = hit["entity"]

#     # Sparse Search (如果有)
#     if sparse_encoder:
#         sparse_vec = to_sparse_list(await cl.make_async(sparse_encoder.encode)(search_query, convert_to_tensor=True))
#         res_s = await cl.make_async(client.search)(
#             collection_name=COLLECTION, data=[sparse_vec], anns_field="sparse_vec",
#             limit=60, output_fields=["pk", "formula", "text", "band_gap", "ehull"]
#         )
#         for hit in res_s[0]:
#             candidates[hit["entity"]["pk"]] = hit["entity"] # 简单的并集

#     cand_list = list(candidates.values())
#     if not cand_list: return []

#     # 3. Rerank
#     pairs = [[query, (d.get("text", "") or "")[:500]] for d in cand_list]
#     scores = await cl.make_async(rerank_predict)(pairs)
    
#     # 4. 🌟 策略 C：强制置顶逻辑 (Hard Boosting)
#     final_results = []
#     for doc, score in zip(cand_list, scores):
#         doc["score"] = score
#         # 如果提取到了 formula，且文档里的 formula 和它完全一致（忽略大小写）
#         # 直接给巨额加分，保证排第一
#         if target_formula and doc.get("formula", "").lower() == target_formula.lower():
#             doc["score"] += 100.0  # 🚀 暴力置顶
#             doc["is_exact_match"] = True
#         else:
#             doc["is_exact_match"] = False
#         final_results.append(doc)
#     # ==========================
#     # 5. 🌟 策略 D: 稳定性终极重排 (Stability Re-ranking) [新增]
#     # ==========================
#     def smart_sort_key(doc):
#         # 第一优先级: 是否是精确匹配的化学式? (1=是, 0=否)
#         # 是精确匹配的，永远排在非精确匹配的前面
#         is_exact = 1 if doc.get("is_exact_match") else 0
        
#         # 第二优先级: 热力学稳定性 (Ehull)
#         # 我们希望 Ehull 越小越好 (0 是最好的)
#         # 所以用负数 -ehull 来排序 (因为 Python sort 默认 reverse=True 是大数在前)
#         ehull = doc.get("ehull", 10.0)
#         if ehull is None: ehull = 10.0
        
#         # 只有在“精确匹配”的圈子里，才严格按稳定性排位
#         # 如果是模糊搜索，可能相关性更重要，所以权重给低一点
#         stability_score = -ehull if is_exact else 0
        
#         # 第三优先级: 原始检索分 (Vector Similarity)
#         # 当稳定性一样时 (比如都有两个 Ehull=0 的相)，才看谁的文本更相关
#         raw_score = doc.get("score", 0)
        
#         return (is_exact, stability_score, raw_score)

#     # 执行多级排序
#     final_results.sort(key=smart_sort_key, reverse=True)

#     # Debug: 打印一下最终谁是老大
#     if final_results:
#         top = final_results[0]
#         print(f"🥇 最终Top1: {top.get('formula')} (Ehull={top.get('ehull')}, Exact={top.get('is_exact_match')})")

#     return final_results[:top_n]

# # ================= 🎨 Prompt 工程 =================

# def build_prompt(context_str: str, query: str) -> str:
#     if not context_str:
#         # 无证据时的 Prompt
#         return f"""
# 你是一名材料科学专家。用户问了一个物理概念问题："{query}"。
# 请用通俗易懂但专业的语言解释该概念。
# **注意**：不要编造具体材料的数据。如果需要举例，请使用经典的教科书材料（如 Si, GaAs, TiO2）。
# """
    
#     # 有证据时的 Prompt
#     return f"""
# 你是一名严谨的计算材料科学家。请基于以下【检索到的数据库证据】回答问题。

# ### 🚫 严格约束
# 1. **数据优先**：必须优先引用证据中的数值（Band Gap, Ehull）。
# 2. **DFT修正**：证据中的 Band Gap 通常是 DFT 计算值（偏小）。引用时请说明：“数据库显示计算带隙为 X eV（PBE值，通常被低估），实验值可能更高。”
# 3. **稳定性判断**：如果 Ehull = 0，说明它是热力学稳定相；如果 Ehull > 0.1 eV/atom，它是亚稳或不稳定相。
# 4. **拒绝幻觉**：如果证据里没有某个参数（如导电率），直接说“数据库中未包含此数据”，严禁瞎编。

# ### 📚 数据库证据
# {context_str}

# 用户问题：{query}
# """

# # ================= 🚀 Chainlit 主流程 =================

# @cl.on_chat_start
# async def start():
#     await cl.Message(content="👋 **Materials AI Pro** 已启动。\n已启用：意图路由、强制置顶、DFT 纠错。").send()

# @cl.on_message
# async def main(message: cl.Message):
#     # 1. 检索
#     docs = await search_pipeline(message.content, top_n=3)
    
#     # 2. 构建 Context
#     context_str = ""
#     elements = []
    
#     if docs:
#         parts = []
#         for i, d in enumerate(docs):
#             # 标记是否是精确匹配
#             tag = "🎯精确匹配" if d.get("is_exact_match") else f"参考 #{i+1}"
            
#             content = (
#                 f"Formula: {d.get('formula')}\n"
#                 f"Band Gap (DFT): {d.get('band_gap')} eV\n"
#                 f"Ehull: {d.get('ehull')} eV/atom\n"
#                 f"Details: {d.get('text')[:300]}..."
#             )
#             parts.append(f"[{tag}] {content}")
#             # 侧边栏
#             elements.append(cl.Text(name=f"{tag}: {d.get('formula')}", content=content, display="side"))
#         context_str = "\n\n".join(parts)
    
#     # 3. 构造 Prompt
#     sys_prompt = build_prompt(context_str, message.content)
    
#     # 4. 生成
#     msg = cl.Message(content="", elements=elements)
#     await msg.send()
    
#     # 历史记录管理
#     history = cl.user_session.get("history", [])
#     messages = [{"role": "system", "content": sys_prompt}] + history[-4:] + [{"role": "user", "content": message.content}]
    
#     streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
#     text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer([text_input], return_tensors="pt").to("cuda")
    
#     thread = Thread(target=model.generate, kwargs={
#         **inputs, "streamer": streamer, "max_new_tokens": 4096, "temperature": 0.1
#     })
#     thread.start()
    
#     final_text = ""
#     for token in streamer:
#         final_text += token
#         await msg.stream_token(token)
    
#     # 5. 收尾
#     final_text = clean_latex(final_text)
#     msg.content = final_text
#     await msg.update()
    
#     history.append({"role": "user", "content": message.content})
#     history.append({"role": "assistant", "content": final_text})
#     cl.user_session.set("history", history)
# import os
# import re
# import json
# import torch
# import chainlit as cl
# from typing import List, Dict, Any, Optional
# from threading import Thread
# import requests

# # ================= 🚑 环境变量 =================
# os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "./hf_cache"

# from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, AutoModel, AutoModelForSequenceClassification
# from peft import PeftModel
# from sentence_transformers import SparseEncoder
# from transformers import AutoTokenizer as HFTokenizer
# import torch.nn.functional as F
# from pymilvus import MilvusClient

# # ================= ⚙️ 配置区 =================
# BASE_MODEL_PATH = "./qwen_model/Qwen/Qwen2___5-7B-Instruct"
# LORA_PATH = "./lora_test"

# # RAG 模型路径
# BGE_DIR = "./hf_cache/transformers/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
# SPLADE_DIR = "./hf_cache/transformers/models--naver--splade-cocondenser-ensembledistil/snapshots/49cf4c7b0db5b870a401ddf5e2669993ef3699c7"
# RERANK_DIR = "./hf_cache/transformers/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70"

# # 数据库
# MILVUS_URI = "./milvus_lite.db"
# COLLECTION = "materials_hybrid"

# # API 配置
# ARK_API_KEY = "1550c65b-2643-4c98-9c89-e63c2762cbe8"
# ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
# ROUTER_MODEL = "ep-20260110133352-bczxh" 

# # ================= 🧠 模型加载 (Global) =================
# print("🚀 初始化: 正在加载大模型...")
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, local_files_only=True)
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL_PATH,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     local_files_only=True,
# )
# model = PeftModel.from_pretrained(base_model, LORA_PATH, local_files_only=True)
# model.eval()

# print("📚 初始化: 正在加载 RAG 组件 (CPU)...")
# try:
#     client = MilvusClient(uri=MILVUS_URI)
# except Exception as e:
#     print(f"⚠️ Milvus 警告: {e}")
#     client = None

# bge_tok = HFTokenizer.from_pretrained(BGE_DIR, local_files_only=True)
# bge_model = AutoModel.from_pretrained(BGE_DIR, local_files_only=True).to("cpu").eval()

# try:
#     sparse_encoder = SparseEncoder(SPLADE_DIR, device="cpu", trust_remote_code=True, local_files_only=True)
# except:
#     print("⚠️ SPLADE 加载失败，将降级为仅 Dense 检索")
#     sparse_encoder = None

# rerank_tok = HFTokenizer.from_pretrained(RERANK_DIR, local_files_only=True)
# rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_DIR, local_files_only=True).to("cpu").eval()

# print("✅ 所有模型加载完毕！")

# # ================= 🛠️ 核心工具函数 =================

# def ark_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
#     headers = {"Authorization": f"Bearer {ARK_API_KEY}", "Content-Type": "application/json"}
#     payload = {"model": model, "messages": messages, "temperature": temperature}
#     try:
#         r = requests.post(ARK_BASE_URL, headers=headers, json=payload, timeout=5)
#         r.raise_for_status()
#         return r.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         print(f"API Error: {e}")
#         return "{}"

# @torch.no_grad()
# def dense_encode(texts):
#     if isinstance(texts, str): texts = [texts]
#     inp = bge_tok(texts, padding=True, truncation=True, return_tensors="pt").to("cpu")
#     out = bge_model(**inp)
#     emb = (out.last_hidden_state * inp["attention_mask"].unsqueeze(-1)).sum(1) / inp["attention_mask"].sum(1).clamp(min=1e-9)
#     return F.normalize(emb, p=2, dim=1).cpu().numpy()

# def to_sparse_list(vec):
#     vec = vec.cpu()
#     if vec.is_sparse: vec = vec.coalesce()
#     idx = vec.indices()[0].tolist()
#     val = vec.values().tolist()
#     return [(int(i), float(x)) for i, x in zip(idx, val)]

# @torch.no_grad()
# def rerank_predict(pairs, batch_size=16):
#     scores = []
#     for i in range(0, len(pairs), batch_size):
#         batch = pairs[i:i+batch_size]
#         qs = [x[0] for x in batch]
#         ds = [x[1] for x in batch]
#         inp = rerank_tok(qs, ds, padding=True, truncation=True, return_tensors="pt").to("cpu")
#         out = rerank_model(**inp)
#         logits = out.logits
#         s = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, -1]
#         scores.extend(s.detach().cpu().tolist())
#     return scores

# def clean_latex(text: str) -> str:
#     text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
#     return re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

# def format_rag_data(doc: Dict[str, Any]) -> str:
#     """把数据库的 Raw Data 清洗成人话"""
#     formula = doc.get('formula', 'Unknown')
#     bg = doc.get('band_gap')
#     ehull = doc.get('ehull')
#     text = doc.get('text', '')
    
#     # 清洗 Text 里的干扰字符
#     clean_desc = text.replace("[RETRIEVAL_HEADER]", "").replace("[CONTENT]", "").replace("\n", "; ")
    
#     # 稳定性描述
#     if ehull is None:
#         stab_str = "❓ 未知"
#     elif ehull <= 0.001:
#         stab_str = "✅ 热力学稳定基态 (Stable)"
#     elif ehull < 0.1:
#         stab_str = "⚠️ 亚稳态 (Metastable)"
#     else:
#         stab_str = "❌ 不稳定 (Unstable)"
        
#     # 带隙描述
#     bg_str = f"{bg:.3f} eV" if isinstance(bg, (int, float)) else "N/A"
    
#     return (
#         f"### 材料: {formula}\n"
#         f"- **稳定性**: {stab_str} (Ehull={ehull} eV/atom)\n"
#         f"- **带隙 (DFT)**: {bg_str}\n"
#         f"- **详情**: {clean_desc[:300]}..."
#     )

# # ================= 🧭 核心逻辑：全局规划 + 智能检索 =================

# async def global_search_planner(query: str) -> List[str]:
#     """全局规划器"""
#     sys_prompt = (
#         "你是材料科学检索规划师。请分析用户问题，列出回答该问题所需的所有关键材料化学式。\n"
#         "策略：\n"
#         "1. **具体查询**（如'查一下Si，查一下氯化钠'）：直接提取化学式。\n"
#         "2. **概念**（如'什么是带隙？'）：返回空列表 []\n"
#         "3. **闲聊**（如'你好'）：返回空列表 []。\n"
#         "输出：只输出一个 JSON 字符串列表，例如 [\"Si\", \"GaAs\"]。不要包含 markdown 或其他文字。"
#     )
    
#     try:
#         resp = await cl.make_async(ark_chat)(ROUTER_MODEL, [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}], temperature=0.1)
#         m = re.search(r"\[.*\]", resp, re.DOTALL)
#         if m:
#             formulas = json.loads(m.group(0))
#             # 这里的正则要把化学式常见的符号都包进去
#             valid_formulas = [f for f in formulas if re.match(r"^[A-Za-z0-9\(\)]+$", str(f))]
#             return list(set(valid_formulas))[:4]
#     except Exception as e:
#         print(f"Planner Error: {e}")
    
#     return []

# async def search_pipeline(query: str, top_n: int = 3):
#     if not client: return []

#     # Planner 传进来的 query 本身就是 formula
#     target_formula = query.strip()
    
#     # 2. 检索
#     dense_vec = (await cl.make_async(dense_encode)(query))[0].tolist()
    
#     candidates = {}
    
#     res_d = await cl.make_async(client.search)(
#         collection_name=COLLECTION, data=[dense_vec], anns_field="dense_vec",
#         limit=60, output_fields=["pk", "formula", "text", "band_gap", "ehull"]
#     )
#     for hit in res_d[0]:
#         candidates[hit["entity"]["pk"]] = hit["entity"]

#     if sparse_encoder:
#         sparse_vec = to_sparse_list(await cl.make_async(sparse_encoder.encode)(query, convert_to_tensor=True))
#         res_s = await cl.make_async(client.search)(
#             collection_name=COLLECTION, data=[sparse_vec], anns_field="sparse_vec",
#             limit=60, output_fields=["pk", "formula", "text", "band_gap", "ehull"]
#         )
#         for hit in res_s[0]:
#             candidates[hit["entity"]["pk"]] = hit["entity"]

#     cand_list = list(candidates.values())
#     if not cand_list: return []

#     # 3. Rerank
#     pairs = [[query, (d.get("text", "") or "")[:500]] for d in cand_list]
#     scores = await cl.make_async(rerank_predict)(pairs)
    
#     # 4. 赋值 Score 并标记精确匹配
#     final_results = []
#     for doc, score in zip(cand_list, scores):
#         doc["score"] = score
#         if target_formula.lower() == doc.get("formula", "").lower():
#             doc["is_exact_match"] = True
#         else:
#             doc["is_exact_match"] = False
#         final_results.append(doc)

#     # 5. 🌟 稳定性终极重排
#     def smart_sort_key(doc):
#         is_exact = 1 if doc.get("is_exact_match") else 0
#         ehull = doc.get("ehull")
#         if ehull is None: ehull = 10.0
        
#         # 优先精确匹配，其次稳定性（越小越好）
#         stability_score = -ehull if is_exact else 0
#         raw_score = doc.get("score", 0)
        
#         return (is_exact, stability_score, raw_score)

#     final_results.sort(key=smart_sort_key, reverse=True)
#     return final_results[:top_n]

# # ================= 🚀 Chainlit 主流程 (完全重写版) =================

# @cl.on_chat_start
# async def start():
#     await cl.Message(content="👋 **Materials AI Pro (Final)** 已启动。\n已启用：Planner规划 + 稳定性重排 + 自然语言Context。").send()

# @cl.on_message
# async def main(message: cl.Message):
#     # ================= 🧠 Step 1: 全局规划 (Plan) =================
#     msg_thinking = cl.Message(content=f"🤔 正在规划...", author="System")
#     await msg_thinking.send()
    
#     target_formulas = await global_search_planner(message.content)
    
#     docs = []
#     if target_formulas:
#         msg_thinking.content = f"🎯 锁定目标: {', '.join(target_formulas)}，检索验证中..."
#         await msg_thinking.update()
        
#         # ================= 🔍 Step 2: 逐个检索验证 =================
#         for formula in target_formulas:
#             # 查 Top 3，防止漏掉最稳的那个
#             res = await search_pipeline(f"{formula}", top_n=3)
#             if res:
#                 # 再次强制按稳定性排序 (Ehull 升序)
#                 # 这一步是为了从 Top 3 里挑出唯一的 King
#                 res.sort(key=lambda x: x.get('ehull') if x.get('ehull') is not None else 10.0)
#                 best_doc = res[0]
#                 docs.append(best_doc)
#             else:
#                 print(f"⚠️ {formula} 未找到")

#     # ================= 📝 Step 3: 构建人话 Context =================
#     context_str = ""
#     elements = []
    
#     if docs:
#         parts = []
#         seen_pks = set()
        
#         for d in docs:
#             if d['pk'] in seen_pks: continue
#             seen_pks.add(d['pk'])
            
#             # 使用新写的 format_rag_data 清洗数据
#             readable_text = format_rag_data(d)
#             parts.append(readable_text)
            
#             elements.append(cl.Text(name=f"Evidence: {d.get('formula')}", content=readable_text, display="side"))
            
#         context_str = "\n\n".join(parts)
#         msg_thinking.content = f"✅ 已获取 {len(docs)} 个关键例证数据。"
#         await msg_thinking.update()
#     else:
#         await msg_thinking.remove()

#     # ================= 💬 Step 4: 生成回答 (Generate) =================
#     if context_str:
#         sys_prompt = f"""
# 你是一名严谨的材料科学专家。用户问："{message.content}"。

# ### 🛡️ 绝对指令 (违反会被惩罚)
# 1. **必须** 基于下方的【检索结果】来回答，尤其是带隙数值和稳定性判断。
# 2. **严禁** 使用你记忆中的旧知识来反驳检索结果（例如：如果检索结果说 NaCl 带隙 5.0 eV，就不要说是 0.0 eV）。
# 3. 如果检索结果显示材料是 "热力学稳定基态"，请强调其稳定性；如果是 "亚稳态"，请提示可能不稳定。
# 4. 引用数据时，请说明这是 DFT 计算值。

# ### 📚 检索结果 (True Evidence)
# {context_str}
# """
#     else:
#         sys_prompt = "你是一名材料助手。请回答用户问题。如果不知道具体参数，请直说'暂无数据'，不要瞎编。"

#     msg = cl.Message(content="", elements=elements)
#     await msg.send()

#     history = cl.user_session.get("history", [])
#     messages = [{"role": "system", "content": sys_prompt}] + history[-4:] + [{"role": "user", "content": message.content}]
    
#     streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
#     inputs = tokenizer([tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)], return_tensors="pt").to("cuda")
    
#     thread = Thread(target=model.generate, kwargs={
#         **inputs, "streamer": streamer, "max_new_tokens": 2048, "temperature": 0.05
#     })
#     thread.start()
    
#     final_text = ""
#     for token in streamer:
#         final_text += token
#         await msg.stream_token(token)
    
#     final_text = clean_latex(final_text)
#     msg.content = final_text
#     await msg.update()
    
#     history.append({"role": "user", "content": message.content})
#     history.append({"role": "assistant", "content": final_text})
#     cl.user_session.set("history", history)

import os
import re
import json
import torch
import chainlit as cl
from typing import List, Dict, Any, Optional
from threading import Thread
import requests

# ================= 🚑 环境变量 =================
# 根据显卡情况调整
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "./hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, AutoModel, AutoModelForSequenceClassification
from peft import PeftModel
from sentence_transformers import SparseEncoder
from transformers import AutoTokenizer as HFTokenizer
import torch.nn.functional as F
from pymilvus import MilvusClient

# ================= ⚙️ 配置区 =================
BASE_MODEL_PATH = "./qwen_model/Qwen/Qwen2___5-7B-Instruct"
LORA_PATH = "./lora_stage1"

# RAG 模型路径
BGE_DIR = "./hf_cache/transformers/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
SPLADE_DIR = "./hf_cache/transformers/models--naver--splade-cocondenser-ensembledistil/snapshots/49cf4c7b0db5b870a401ddf5e2669993ef3699c7"
RERANK_DIR = "./hf_cache/transformers/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70"

# 数据库
MILVUS_URI = "./milvus_lite.db"
COLLECTION = "materials_hybrid"

# API 配置 (用于意图识别和Text2SQL)
ARK_API_KEY = "1550c65b-2643-4c98-9c89-e63c2762cbe8"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
ROUTER_MODEL = "ep-20260110133352-bczxh" 

# ================= 🧠 模型加载 (Global) =================
print("🚀 初始化: 正在加载大模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH, local_files_only=True)
model.eval()

print("📚 初始化: 正在加载 RAG 组件 (CPU)...")
try:
    client = MilvusClient(uri=MILVUS_URI)
except Exception as e:
    print(f"⚠️ Milvus 警告: {e}")
    client = None

bge_tok = HFTokenizer.from_pretrained(BGE_DIR, local_files_only=True)
bge_model = AutoModel.from_pretrained(BGE_DIR, local_files_only=True).to("cpu").eval()

try:
    sparse_encoder = SparseEncoder(SPLADE_DIR, device="cpu", trust_remote_code=True, local_files_only=True)
except:
    print("⚠️ SPLADE 加载失败，将降级为仅 Dense 检索")
    sparse_encoder = None

rerank_tok = HFTokenizer.from_pretrained(RERANK_DIR, local_files_only=True)
rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_DIR, local_files_only=True).to("cpu").eval()

print("✅ 所有模型加载完毕！")

# ================= 🛠️ 核心工具函数 =================

def ark_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
    headers = {"Authorization": f"Bearer {ARK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    try:
        r = requests.post(ARK_BASE_URL, headers=headers, json=payload, timeout=5)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API Error: {e}")
        return "{}"

@torch.no_grad()
def dense_encode(texts):
    if isinstance(texts, str): texts = [texts]
    inp = bge_tok(texts, padding=True, truncation=True, return_tensors="pt").to("cpu")
    out = bge_model(**inp)
    emb = (out.last_hidden_state * inp["attention_mask"].unsqueeze(-1)).sum(1) / inp["attention_mask"].sum(1).clamp(min=1e-9)
    return F.normalize(emb, p=2, dim=1).cpu().numpy()

def to_sparse_list(vec):
    vec = vec.cpu()
    if vec.is_sparse: vec = vec.coalesce()
    idx = vec.indices()[0].tolist()
    val = vec.values().tolist()
    return [(int(i), float(x)) for i, x in zip(idx, val)]

@torch.no_grad()
def rerank_predict(pairs, batch_size=16):
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        qs = [x[0] for x in batch]
        ds = [x[1] for x in batch]
        inp = rerank_tok(qs, ds, padding=True, truncation=True, return_tensors="pt").to("cpu")
        out = rerank_model(**inp)
        logits = out.logits
        s = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, -1]
        scores.extend(s.detach().cpu().tolist())
    return scores

def clean_latex(text: str) -> str:
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    return re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

def format_rag_data(doc: Dict[str, Any]) -> str:
    """把数据库的 Raw Data 清洗成人话 (带 DFT 警告版)"""
    formula = doc.get('formula', 'Unknown')
    bg = doc.get('band_gap')
    ehull = doc.get('ehull')
    is_stable = doc.get('is_stable')
    text = doc.get('text', '')
    
    # 清洗 Text 里的干扰字符
    clean_desc = text.replace("[RETRIEVAL_HEADER]", "").replace("[CONTENT]", "").replace("\n", "; ")
    
    # 稳定性描述
    if ehull is not None:
        if ehull <= 0.001:
            stab_str = "✅ 稳定 (Stable)"
        elif ehull < 0.1:
            stab_str = "⚠️ 亚稳 (Metastable)"
        else:
            stab_str = "❌ 不稳 (Unstable)"
        stab_detail = f"(Ehull={ehull:.3f})"
    elif is_stable is not None:
        stab_str = "✅ 稳定" if is_stable else "⚠️ 可能不稳"
        stab_detail = ""
    else:
        stab_str = "❓ 未知"
        stab_detail = ""
        
    # 🔥 核心修改：带隙描述 (加上 DFT 标签)
    if isinstance(bg, (int, float)):
        # 如果带隙很小 (<0.5) 但又不是金属，加个显眼的警告
        if bg < 0.5 and bg > 0:
            bg_str = f"{bg:.3f} eV (⚠️DFT计算值，严重偏小)"
        else:
            bg_str = f"{bg:.3f} eV (DFT计算值)"
    else:
        bg_str = "N/A"
    
    return (
        f"### 材料: {formula}\n"
        f"- **稳定性**: {stab_str} {stab_detail}\n"
        f"- **带隙**: {bg_str}\n"
        f"- **详情**: {clean_desc[:300]}..."
    )

# ================= 🧭 核心逻辑：意图解析 + 规划 + 检索 =================

async def parse_filter_expression(query: str) -> str:
    """
    Text-to-Filter: 将自然语言转为 Milvus filter 表达式。
    只有当用户明确要求筛选数值时，这里才会返回非空字符串。
    """
    sys_prompt = (
        "你是一个 Milvus 数据库查询专家。请将用户的自然语言筛选要求转换为 Milvus 的 filter 表达式。\n"
        "### 数据库 Schema 定义\n"
        "- band_gap (float): 带隙，单位 eV。\n"
        "- ehull (float): 形成能/稳定性，单位 eV/atom。0表示最稳定。\n"
        "- formula (str): 化学式。\n\n"
        "### 转换规则\n"
        "1. **大于/小于**: '带隙大于1.5' -> `band_gap > 1.5`\n"
        "2. **范围**: '带隙在1到2之间' -> `band_gap >= 1.0 && band_gap <= 2.0`\n"
        "3. **稳定性**: '稳定的' -> `ehull <= 0.05`\n"
        "4. **逻辑组合**: '稳定的且带隙小于1' -> `ehull <= 0.05 && band_gap < 1.0`\n"
        "5. **无条件**: 如果用户没有提出数值筛选要求，或者只是问概念，输出空字符串 ``。\n"
        "6. **输出格式**: 只输出表达式字符串，不要 markdown。\n"
    )
    
    try:
        # 温度设为 0，保证逻辑转换的绝对准确
        resp = await cl.make_async(ark_chat)(ROUTER_MODEL, [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}], temperature=0.0)
        expr = resp.strip().replace("`", "").replace('"', "").replace("'", "")
        # 简单校验一下
        if "band_gap" in expr or "ehull" in expr:
            return expr
    except Exception as e:
        print(f"Filter Parse Error: {e}")
    
    return ""
def _extract_json_array(text: str) -> str:
    """从 LLM 输出中截取第一个 [...] 作为 JSON array"""
    m = re.search(r"\[[\s\S]*\]", text)
    return m.group(0) if m else ""

def _is_valid_formula(s: str) -> bool:
    """化学式粗校验：允许 LiFePO4 / TiO2 / Al2O3 / (NH4)2SO4 这种"""
    if not s or len(s) > 32:
        return False
    # 允许括号、点号（如 CuSO4·5H2O）、连字符（少见但容错）、数字
    return re.fullmatch(r"[A-Za-z0-9\(\)\.\-\u00B7]+", s) is not None
async def global_search_planner(query: str) -> List[str]:
    """
    纯 LLM 材料实体标准化：中文材料名/俗名/缩写 -> 化学式 formula
    返回：化学式列表（只返回 formula，不返回别的）
    """

    sys_prompt = (
        "你是材料实体标准化器。任务：从用户问题中提取“需要去材料数据库检索的材料化学式”。\n"
        "\n"
        "## 重要约束（必须遵守）\n"
        "1) 只输出一个 JSON 数组（array），不要输出任何解释、markdown、额外文字。\n"
        "2) 数组元素为对象：{\"formula\": \"...\", \"confidence\": 0~1}\n"
        "3) 只在你有把握时才填 formula；不确定就把 formula 设为 \"\"，confidence 给低分（<=0.4）。禁止瞎编化学式。\n"
        "4) 支持中文材料名、俗名、缩写：\n"
        "   - 例：砷化镓 -> GaAs；氧化锌 -> ZnO；磷酸铁锂/LFP -> LiFePO4；三元/NCM -> 可能无法唯一确定 -> 留空。\n"
        "   - 如果用户说“三元材料(NCM811)”这类：可以输出近似式 LiNi0.8Co0.1Mn0.1O2（有把握才给），否则留空。\n"
        "5) 如果问题是概念/原理/闲聊，不涉及具体材料检索，输出空数组 []。\n"
        "6) 最多输出 4 个对象。\n"
        "\n"
        "## 输出示例\n"
        "[{\"formula\":\"GaAs\",\"confidence\":0.95},{\"formula\":\"\",\"confidence\":0.2}]\n"
    )

    user_prompt = (
        f"用户问题：{query}\n"
        "请输出 JSON 数组："
    )

    # --- 重试：一次失败就再来（减少解析失败） ---
    for attempt in range(3):
        try:
            resp = await cl.make_async(ark_chat)(
                ROUTER_MODEL,
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": user_prompt}],
                temperature=0.0  # 提取任务强烈建议 0
            )

            arr_text = _extract_json_array(resp)
            if not arr_text:
                raise ValueError(f"LLM 输出没有 JSON array: {resp[:200]}")

            items = json.loads(arr_text)

            formulas: List[str] = []
            for it in items[:6]:
                f = (it.get("formula") or "").strip()
                conf = float(it.get("confidence", 0.0) or 0.0)

                # 置信度低 or 空 -> 丢弃
                if (not f) or (conf < 0.55):
                    continue

                # 化学式合法性校验（防止乱写）
                if not _is_valid_formula(f):
                    continue

                formulas.append(f)

            # 去重保序
            uniq = []
            for x in formulas:
                if x not in uniq:
                    uniq.append(x)

            return uniq[:4]

        except Exception as e:
            print(f"Planner Error attempt {attempt+1}/3: {e}")

    return []



async def search_pipeline(query: str, filter_expr: str = "", top_n: int = 3):
    if not client: return []

    target_formula = query.strip()
    
    # 构造检索参数
    search_params = {
        "collection_name": COLLECTION,
        "limit": 60,
        "output_fields": ["pk", "formula", "text", "band_gap", "ehull","is_stable"]
    }
    
    # 🔥 如果 filter_expr 有内容，这里才会生效
    if filter_expr:
        print(f"🔍 [Filter Active]: {filter_expr}")
        search_params["filter"] = filter_expr

    # 1. 检索 (Dense)
    dense_vec = (await cl.make_async(dense_encode)(query))[0].tolist()
    candidates = {}
    
    try:
        res_d = await cl.make_async(client.search)(
            data=[dense_vec], anns_field="dense_vec", **search_params
        )
        for hit in res_d[0]:
            candidates[hit["entity"]["pk"]] = hit["entity"]
    except Exception as e:
        print(f"Search Error: {e}")

    # 2. 检索 (Sparse)
    if sparse_encoder:
        try:
            sparse_vec = to_sparse_list(await cl.make_async(sparse_encoder.encode)(query, convert_to_tensor=True))
            res_s = await cl.make_async(client.search)(
                data=[sparse_vec], anns_field="sparse_vec", **search_params
            )
            for hit in res_s[0]:
                candidates[hit["entity"]["pk"]] = hit["entity"]
        except Exception as e:
            pass

    cand_list = list(candidates.values())
    # === Silicon Incident 修复：同素异形体干扰时，强制“基态优先” ===
    # 如果用户 query 能解析成一个合法化学式，且当前是“查具体材料”场景（无 filter），
    # 额外用结构化 filter 拉取该化学式的更多候选，再做 (stable + low Ehull) 排序。
    if (not filter_expr) and _is_valid_formula(target_formula):
        try:
            extra_rows = await cl.make_async(client.query)(
                collection_name=COLLECTION,
                filter=f"formula == '{target_formula}'",
                output_fields=[
                    "pk", "formula", "space_group", "crystal_system",
                    "is_stable", "ehull", "energy_above_hull",
                    "formation_energy_per_atom", "band_gap", "is_gap_direct",
                    "is_metal", "density", "volume", "elements", "text"
                ],
                limit=200,
            )
            for r in (extra_rows or []):
                candidates[r.get("pk")] = r
            cand_list = list(candidates.values())
        except Exception as e:
            print("⚠️ extra formula query failed:", e)

    if not cand_list: return []

    # 3. Rerank
    # 如果有 filter，我们不过度依赖精确匹配，而是依赖 filter 筛选后的结果
    pairs = [[query, (d.get("text", "") or "")[:500]] for d in cand_list]
    scores = await cl.make_async(rerank_predict)(pairs)
    
    final_results = []
    for doc, score in zip(cand_list, scores):
        doc["score"] = score
        # 标记精确匹配 (仅当没有 filter 时，精确匹配才最重要；有 filter 时，满足条件的都重要)
        if not filter_expr and target_formula.lower() == doc.get("formula", "").lower():
            doc["is_exact_match"] = True
        else:
            doc["is_exact_match"] = False
        final_results.append(doc)

    # 4. 排序逻辑
    def smart_sort_key(doc):
        # 1) 精确化学式/化学计量匹配（强约束）
        is_exact = 1 if (str(doc.get("formula", "")).strip().lower() == target_formula.strip().lower()) else 0

        # 2) 稳定性（字段类型可能是 bool/int/str，做鲁棒解析）
        def _to_bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return v != 0
            if isinstance(v, str):
                return v.strip().lower() in ("true", "1", "yes", "y", "stable")
            return False
        stable = 1 if _to_bool(doc.get("is_stable")) else 0

        # 3) Ehull（越小越接近基态/凸包；0 通常代表热力学稳定相）
        def _ehull_val(d):
            for k in ("ehull", "energy_above_hull", "e_above_hull", "energyAboveHull"):
                if k in d and d[k] is not None:
                    try:
                        return float(d[k])
                    except Exception:
                        pass
            return 1e9
        ehull = _ehull_val(doc)

        # 4) 相关性分数（越大越相关）
        score = float(doc.get("score", 0.0))

        # filter 场景：更偏“筛选”（稳定 + 低Ehull 优先，其次相关性）
        if filter_expr:
            return (stable, -ehull, score, is_exact)

        # 非 filter：查具体材料，先 exact，再稳定，再低Ehull（基态优先），再相关性
        return (is_exact, stable, -ehull, score)

    final_results.sort(key=smart_sort_key, reverse=True)
    return final_results[:top_n]

# ================= 🚀 Chainlit 主流程 =================

@cl.on_chat_start
async def start():
    await cl.Message(content="👋 **Materials AI Pro (Full Version)** 已启动。\n集成：Text-to-Filter筛选 + 全局规划 + 稳定性重排。").send()

@cl.on_message
async def main(message: cl.Message):
    msg_thinking = cl.Message(content=f"🤔 正在解析意图...", author="System")
    await msg_thinking.send()
    
    # 1. 解析硬性过滤条件 (如: band_gap < 0.5)
    # 只有用户问到数值筛选时，这里才会有值
    filter_expr = await parse_filter_expression(message.content)
    
    # 2. 全局规划 (提取化学式)
    target_formulas = await global_search_planner(message.content)
    
    docs = []
    
    # ================= 分支 A: 启用数值筛选 (Filter Mode) =================
    if filter_expr:
        msg_thinking.content = f"🔍 启用条件筛选: `{filter_expr}`"
        await msg_thinking.update()
        
        # 用原话去搜，但带上 filter 约束
        # top_n 设为 5，方便展示多个结果
        res = await search_pipeline(message.content, filter_expr=filter_expr, top_n=5)
        if res:
            docs.extend(res)
            
    # ================= 分支 B: 查具体材料/概念 (Lookup Mode) =================
    elif target_formulas:
        msg_thinking.content = f"🎯 锁定目标: {', '.join(target_formulas)}..."
        await msg_thinking.update()
        
        for formula in target_formulas:
            # 没 filter，就查最稳的那个 (top_n=1)
            res = await search_pipeline(f"{formula}", top_n=1)
            if res:
                docs.extend(res)
    
    # ================= 构建 Context =================
    context_str = ""
    elements = []
    
    if docs:
        parts = []
        seen_pks = set()
        
        for d in docs:
            if d['pk'] in seen_pks: continue
            seen_pks.add(d['pk'])
            
            readable_text = format_rag_data(d)
            parts.append(readable_text)
            
            tag = f"{d.get('formula')}"
            elements.append(cl.Text(name=tag, content=readable_text, display="side"))
            
        context_str = "\n\n".join(parts)
        
        msg_thinking.content = f"✅ 已获取 {len(docs)} 条有效数据。"
        await msg_thinking.update()
    else:
        await msg_thinking.remove()

    # ================= 生成回答 =================
    if context_str:
        sys_prompt = f"""
你是一名材料科学专家助手。你必须结合【检索证据】和你的【专家常识】来回答。

### ⚠️ 核心警告：关于带隙 (Band Gap) 数据
你检索到的数据库 (Materials Project) 中的 `band_gap` 字段通常是 **DFT-PBE 理论计算值**。
**DFT 计算值通常严重低估实验带隙**（例如：它可能显示 Si≈0 eV, ZnO≈0.7 eV，这些与实验值严重不符！）。

### 回答策略 (必须严格执行)：
1. **通用材料修正**：如果用户问的是常见半导体 (如 Si, GaAs, ZnO, TiO2, GaN 等)，**绝对不要直接引用检索到的错误带隙数值**。
   - **正确做法**：直接使用你内建的**实验测量值**回答（例如：Si 约 1.12 eV, ZnO 约 3.37 eV）。
   - **话术示例**：“虽然检索数据显示 DFT 计算带隙仅为 0.7 eV（严重低估），但实际实验带隙约为 3.37 eV。”
   
2. **未知材料引用**：只有当用户问的是生僻/新型材料（你不知道实验值）时，才引用检索数据的数值，并必须标注：“此为 DFT 理论计算值，可能显著低于真实值。”

3. **其他属性**：对于晶体结构、空间群、稳定性 (Ehull)，请优先信任【检索证据】。

【检索证据】
{context_str}
"""

    else:
        sys_prompt = """你是一名材料科学专家助手，回答要自然、清晰、有用。

如果是概念/原理类问题：直接用你的知识回答（结论 + 关键机制）。
如果是具体材料参数/数值（带隙、熔点、密度等）：你可以给常见范围或典型值，若有特殊情况请说明“可能因晶型/温度/掺杂/文献不同而变化”；如果你不确定，就直接说不确定，不要编造非常精确的小数值。
"""

    msg = cl.Message(content="", elements=elements)
    await msg.send()

    history = cl.user_session.get("history", [])
    messages = [{"role": "system", "content": sys_prompt}] + history[-4:] + [{"role": "user", "content": message.content}]
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer([tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)], return_tensors="pt").to("cuda")
    
    thread = Thread(target=model.generate, kwargs={
        **inputs, "streamer": streamer, "max_new_tokens": 1024, "temperature": 0.2
    })
    thread.start()
    
    final_text = ""
    for token in streamer:
        final_text += token
        await msg.stream_token(token)
    
    final_text = clean_latex(final_text)
    msg.content = final_text
    await msg.update()
    
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": final_text})
    cl.user_session.set("history", history)