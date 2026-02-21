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
