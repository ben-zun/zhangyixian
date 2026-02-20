# # build_sft_from_md_jsonl.py
# # 将一堆长 MD 知识文档切块 -> 生成可用于指令微调的 JSONL（messages 格式）
# # 目标：不丢失知识（默认 MODE="copy" 最稳）

# import os
# import glob
# import json
# import re
# from typing import List, Dict, Any, Optional, Tuple

# # ================= 🚑 必须最前：离线环境变量（按你工程习惯）=================
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# os.environ["HF_HOME"] = "./hf_cache"
# os.environ["HF_HUB_CACHE"] = "./hf_cache/hub"
# os.environ["TRANSFORMERS_CACHE"] = "./hf_cache/transformers"
# # ===========================================================================

# # ================= ✅ 你要改的路径都写死在这里 =================
# MD_DIR = "./my_md_knowledge"          # <<< 放你的 md 文件目录（会递归找 *.md）
# OUT_JSONL = "./sft_md_out/train.jsonl"  # <<< 输出 jsonl
# OUT_META = "./sft_md_out/meta.json"     # <<< 输出每条样本来自哪个文件/哪一块（方便追踪）
# # ===============================================================

# # ============== ✅ 模式选择 ==============
# # "copy"   : 不调用模型，assistant 直接复述块内容（最保真，不丢知识）
# # "rewrite": 调用本地大模型对块做“忠实改写（不删减）”
# MODE = "copy"

# # ============== 如果用 rewrite 模式，需要你的本地模型路径 ==============
# BASE_MODEL_PATH = "./qwen_model/Qwen/Qwen2___5-7B-Instruct"
# LORA_PATH = None  # 例如 "./lora_lcot_out"；不想用 LoRA 就 None
# # ========================================================================

# # ================= 切块参数（非常关键） =================
# # 注意：这是 tokenizer token 上限，不是字符数
# CHUNK_MAX_TOKENS = 1800     # 每块最大 token（留一些空间给 system/user 提示词）
# CHUNK_OVERLAP_TOKENS = 120  # 块之间重叠，防止跨段落信息断裂
# MIN_CHUNK_TOKENS = 80       # 太短的块直接丢弃（避免垃圾样本）
# # =======================================================

# # ================= 样本提示词（你可按口味改）=================
# SYSTEM_PROMPT = (
#     "你是材料科学知识库助手。你必须输出中文，且内容必须忠实、完整、可追溯。"
# )

# # copy 模式：用户给“资料片段”，assistant 要“完整复述 + 保留公式/符号/引用标记”
# USER_TEMPLATE_COPY = (
#     "请你学习并记住下面这段材料知识。要求：\n"
#     "1) 不要丢失任何知识点、数值、公式、符号、引用标记（如 [@xxx]）。\n"
#     "2) 允许排版更清晰，但不能删减。\n"
#     "3) 输出必须为中文。\n\n"
#     "【资料片段】\n{chunk}\n"
# )

# # rewrite 模式：让模型“忠实改写但不删减”
# USER_TEMPLATE_REWRITE = (
#     "你在做“知识保真转写”。请将下面资料片段转写成更适合指令微调的知识回答：\n"
#     "- 必须保留全部知识点、数值、公式、符号、引用标记（如 [@xxx]），不能删减；\n"
#     "- 可以重排结构、加小标题、列表，让它更像一段高质量中文讲解；\n"
#     "- 不要加入资料之外的新事实；\n"
#     "- 最终输出就是你的回答正文，不要输出任何额外说明。\n\n"
#     "【资料片段】\n{chunk}\n"
# )
# # =========================================================

# def normalize_md_text(s: str) -> str:
#     # 轻量清洗：不改语义，不做“总结”
#     s = s.replace("\r\n", "\n").replace("\r", "\n")
#     s = re.sub(r"\n{3,}", "\n\n", s).strip()
#     return s

# def split_by_token_window(tokenizer, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
#     """
#     纯按 token 滑窗切块（最稳：不会因为 markdown 结构而超长）
#     """
#     ids = tokenizer.encode(text, add_special_tokens=False)
#     n = len(ids)
#     chunks = []
#     start = 0
#     while start < n:
#         end = min(start + max_tokens, n)
#         piece_ids = ids[start:end]
#         chunk = tokenizer.decode(piece_ids, skip_special_tokens=True)
#         chunk = chunk.strip()
#         if chunk:
#             chunks.append(chunk)
#         if end >= n:
#             break
#         start = max(0, end - overlap_tokens)
#     return chunks

# def ensure_dir(p: str):
#     d = os.path.dirname(os.path.abspath(p))
#     if d:
#         os.makedirs(d, exist_ok=True)

# def load_local_model_if_needed(mode: str):
#     if mode != "rewrite":
#         return None, None

#     import torch
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     from peft import PeftModel

#     print("🚀 Loading local model for rewrite...")
#     tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
#     tok.pad_token_id = tok.eos_token_id

#     base = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     base.eval()

#     if LORA_PATH:
#         mdl = PeftModel.from_pretrained(base, LORA_PATH)
#         mdl.eval()
#     else:
#         mdl = base

#     return tok, mdl

# def llm_rewrite(tok, mdl, chunk: str) -> str:
#     import torch

#     user = USER_TEMPLATE_REWRITE.format(chunk=chunk)
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": user},
#     ]
#     text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     enc = tok([text], return_tensors="pt", padding=True)
#     enc = {k: v.to("cuda") for k, v in enc.items()}

#     with torch.no_grad():
#         out_ids = mdl.generate(
#             input_ids=enc["input_ids"],
#             attention_mask=enc["attention_mask"],
#             max_new_tokens=1200,
#             do_sample=False,       # rewrite 要稳定，不采样
#             temperature=0.1,
#             top_p=0.9,
#         )
#     gen = out_ids[0][enc["input_ids"].shape[1]:]
#     ans = tok.decode(gen, skip_special_tokens=True).strip()
#     return ans

# def build_samples_for_chunk(chunk: str, mode: str, tok=None, mdl=None) -> Tuple[Dict[str, Any], str]:
#     """
#     返回：jsonl样本 + 实际assistant内容（便于做 meta 记录）
#     """
#     if mode == "copy":
#         user = USER_TEMPLATE_COPY.format(chunk=chunk)
#         assistant = chunk  # ✅ 绝对保真：不丢知识
#     elif mode == "rewrite":
#         user = USER_TEMPLATE_REWRITE.format(chunk=chunk)
#         assistant = llm_rewrite(tok, mdl, chunk)
#         if not assistant:
#             # 兜底：防止模型抽风输出空
#             assistant = chunk
#     else:
#         raise ValueError(f"Unknown MODE: {mode}")

#     sample = {
#         "messages": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user},
#             {"role": "assistant", "content": assistant},
#         ]
#     }
#     return sample, assistant

# def main():
#     # tokenizer：切块必须用 tokenizer（copy 模式也需要 tokenizer）
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     tok, mdl = load_local_model_if_needed(MODE)

#     ensure_dir(OUT_JSONL)
#     ensure_dir(OUT_META)

#     md_files = sorted(glob.glob(os.path.join(MD_DIR, "**", "*.md"), recursive=True))
#     if not md_files:
#         raise RuntimeError(f"没有在目录里找到 md：{MD_DIR}")

#     samples = []
#     meta = []
#     total_chunks = 0

#     for fp in md_files:
#         with open(fp, "r", encoding="utf-8", errors="ignore") as f:
#             raw = f.read()
#         text = normalize_md_text(raw)

#         chunks = split_by_token_window(
#             tokenizer,
#             text=text,
#             max_tokens=CHUNK_MAX_TOKENS,
#             overlap_tokens=CHUNK_OVERLAP_TOKENS,
#         )

#         # 过滤太短块
#         keep = []
#         for c in chunks:
#             if len(tokenizer.encode(c, add_special_tokens=False)) >= MIN_CHUNK_TOKENS:
#                 keep.append(c)

#         for i, chunk in enumerate(keep):
#             sample, assistant = build_samples_for_chunk(chunk, MODE, tok, mdl)
#             samples.append(sample)
#             meta.append({
#                 "file": fp,
#                 "chunk_index": i,
#                 "mode": MODE,
#                 "chunk_tokens": len(tokenizer.encode(chunk, add_special_tokens=False)),
#             })
#             total_chunks += 1

#         print(f"✅ {fp} -> chunks kept: {len(keep)}")

#     # 写 jsonl
#     with open(OUT_JSONL, "w", encoding="utf-8") as w:
#         for s in samples:
#             w.write(json.dumps(s, ensure_ascii=False) + "\n")

#     with open(OUT_META, "w", encoding="utf-8") as w:
#         json.dump({
#             "md_dir": MD_DIR,
#             "out_jsonl": OUT_JSONL,
#             "mode": MODE,
#             "chunk_max_tokens": CHUNK_MAX_TOKENS,
#             "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
#             "min_chunk_tokens": MIN_CHUNK_TOKENS,
#             "num_files": len(md_files),
#             "num_samples": len(samples),
#             "files": md_files,
#             "samples_meta": meta,
#         }, w, ensure_ascii=False, indent=2)

#     print("\n================ DONE ================")
#     print(f"MD files: {len(md_files)}")
#     print(f"Samples : {len(samples)}")
#     print(f"JSONL   : {OUT_JSONL}")
#     print(f"META    : {OUT_META}")
#     print("======================================")

# if __name__ == "__main__":
#     main()
import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================== 路径配置（你之后自己改） ==================
BASE_MODEL_PATH = "./qwen_model/Qwen/Qwen2___5-7B-Instruct"
MD_DIR = "./library"              # 放你的 .md 文件
OUT_JSONL = "./sft_material_knowledge/train.jsonl"

MAX_NEW_TOKENS = 1400
TEMPERATURE = 0.3
TOP_P = 0.9
# ===============================================================

SYSTEM_PROMPT = (
    "你是一名材料知识助手，擅长用中文撰写系统、深入、结构清晰的材料科学长文。"
    "你的目标是把材料科学概念讲清楚，而不是简单复述。"
    "文章应具有学术风格、分章节结构，并适合研究生或科研人员阅读。"
)

USER_TEMPLATE = """请你系统性地讲解下面这篇材料科学内容对应的知识主题。
要求：
1. 用中文输出；
2. 输出为一篇完整的长文（不少于800字）；
3. 允许引言、分章节、小结；
4. 不要提及“原文”、“这篇文章”等字样；
5. 保留并正确阐释其中的材料科学概念。

材料内容如下：
----------------
{md_text}
"""

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

@torch.no_grad()
def generate_long_article(tokenizer, model, md_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(md_text=md_text)}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True
    )

    gen = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def main():
    os.makedirs(Path(OUT_JSONL).parent, exist_ok=True)

    tokenizer, model = load_model()

    md_files = sorted(Path(MD_DIR).glob("*.md"))
    print(f"📘 Found {len(md_files)} markdown files.")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for md_path in md_files:
            print(f"🔧 Processing: {md_path.name}")
            md_text = md_path.read_text(encoding="utf-8").strip()
            if len(md_text) < 300:
                print("⚠️ Skip (too short)")
                continue

            article = generate_long_article(tokenizer, model, md_text)

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(md_text=md_text)
                    },
                    {"role": "assistant", "content": article}
                ]
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ SFT jsonl saved to: {OUT_JSONL}")

if __name__ == "__main__":
    main()
