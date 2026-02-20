# import os

# # ================= 环境变量 =================
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "./hf_cache"
# os.environ["TRANSFORMERS_CACHE"] = "./hf_cache/transformers"  # 有 warning 无所谓
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# # ================= 固定路径配置 =================
# BASE_MODEL = "./qwen_model/Qwen/Qwen2___5-7B-Instruct"

# STAGE1_TRAIN = "sft_material_knowledges1/train_500_doubao.jsonl"           # 长
# STAGE2_TRAIN = "sft_material_knowledges1/material_concise_qa_400.jsonl"    # 短

# OUT_STAGE1 = "lora_stage1"
# OUT_STAGE2 = "lora_stage2"

# # ================= 训练超参 =================
# STAGE1_EPOCHS = 7      # 长报告
# STAGE2_EPOCHS = 5      # 短问答

# LR_STAGE1 = 1e-4
# LR_STAGE2 = 5e-5       # 第二阶段小一点，避免冲掉长文能力

# BATCH_SIZE = 1
# GRAD_ACCUM = 8
# MAX_LEN = 3072

# LOGGING_STEPS = 5
# SAVE_STEPS = 100
# SEED = 42

# # ================= SwanLab（可选） =================
# USE_SWANLAB = True
# SWAN_PROJECT = "Qwen2.5-LoRA-SFT"
# SWAN_MODE_LOCAL = False  # 没网就 True
# # ==================================================

# from typing import Any
# from dataclasses import dataclass

# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     set_seed,
# )

# from peft import LoraConfig, get_peft_model, PeftModel

# if USE_SWANLAB:
#     import swanlab
#     # ✅ 用新路径，避免 deprecated warning
#     from swanlab.integration.transformers import SwanLabCallback


# def build_text(tokenizer, messages):
#     return tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=False
#     )


# @dataclass
# class Collator:
#     tokenizer: Any
#     max_length: int

#     def __call__(self, feats):
#         texts = [x["text"] for x in feats]
#         enc = self.tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )
#         labels = enc["input_ids"].clone()
#         labels[enc["attention_mask"] == 0] = -100
#         enc["labels"] = labels
#         return enc


# def build_lora_cfg():
#     return LoraConfig(
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=[
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj"
#         ],
#     )


# def _file_nonempty(path: str) -> bool:
#     try:
#         return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0
#     except Exception:
#         return False


# def _has_lora_adapter(out_dir: str) -> bool:
#     """判断某个 LoRA 目录是否已训练完成（至少有 adapter 权重 + config）"""
#     if not out_dir or not os.path.isdir(out_dir):
#         return False
#     cfg = os.path.join(out_dir, "adapter_config.json")
#     w1 = os.path.join(out_dir, "adapter_model.safetensors")
#     w2 = os.path.join(out_dir, "adapter_model.bin")
#     return os.path.exists(cfg) and (os.path.exists(w1) or os.path.exists(w2))


# def _load_train_dataset(train_file: str, tokenizer):
#     ds = load_dataset("json", data_files={"train": train_file})["train"]

#     def map_fn(ex):
#         return {"text": build_text(tokenizer, ex["messages"])}

#     ds = ds.map(map_fn, remove_columns=ds.column_names)
#     return ds


# def train_stage(train_file, out_dir, init_lora=None, epochs=4, lr=1e-4, swan_name="stage"):
#     print(f"\n🚀 Training on {train_file} -> {out_dir}")
#     os.makedirs(out_dir, exist_ok=True)

#     tokenizer = AutoTokenizer.from_pretrained(
#         BASE_MODEL,
#         trust_remote_code=True,
#         local_files_only=True,
#     )
#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     # ✅ 关键修复：强制整模型在单卡（避免 device_map=auto offload 到 CPU）
#     # 你设置了 CUDA_VISIBLE_DEVICES=3，所以 cuda:0 就是那张卡。
#     model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL,
#         torch_dtype=torch.bfloat16,
#         device_map={"": 0},   # ✅ 全部放到 cuda:0
#         trust_remote_code=True,
#         local_files_only=True,
#     )

#     # ✅ checkpointing 配套
#     model.config.use_cache = False
#     model.gradient_checkpointing_enable()

#     try:
#         model.enable_input_require_grads()
#     except Exception:
#         pass

#     lora_cfg = build_lora_cfg()

#     if init_lora:
#         # ✅ 从上阶段 LoRA 加载，并确保可训练
#         model = PeftModel.from_pretrained(
#             model,
#             init_lora,
#             is_trainable=True,
#         )
#         try:
#             model.set_adapter("default")
#         except Exception:
#             pass
#     else:
#         model = get_peft_model(model, lora_cfg)

#     # 看看是否真的有可训练参数
#     try:
#         model.print_trainable_parameters()
#     except Exception:
#         pass

#     if sum(p.requires_grad for p in model.parameters()) == 0:
#         raise RuntimeError("❌ 当前模型没有任何可训练参数（LoRA 可能被冻结或未正确加载）。")

#     ds = _load_train_dataset(train_file, tokenizer)
#     collator = Collator(tokenizer, MAX_LEN)

#     callbacks = []
#     if USE_SWANLAB:
#         cb = SwanLabCallback(
#             project=SWAN_PROJECT,
#             experiment_name=swan_name,
#             config={
#                 "train_file": train_file,
#                 "out_dir": out_dir,
#                 "epochs": epochs,
#                 "lr": lr,
#                 "batch_size": BATCH_SIZE,
#                 "grad_accum": GRAD_ACCUM,
#                 "max_len": MAX_LEN,
#             },
#             mode="local" if SWAN_MODE_LOCAL else None,
#         )
#         callbacks.append(cb)

#     args = TrainingArguments(
#         output_dir=out_dir,
#         num_train_epochs=epochs,
#         learning_rate=lr,
#         per_device_train_batch_size=BATCH_SIZE,
#         gradient_accumulation_steps=GRAD_ACCUM,
#         logging_steps=LOGGING_STEPS,
#         save_steps=SAVE_STEPS,
#         save_total_limit=2,
#         bf16=True,
#         fp16=False,
#         gradient_checkpointing=True,
#         report_to="none",
#         remove_unused_columns=False,
#         dataloader_num_workers=0,
#     )

#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=ds,
#         data_collator=collator,
#         callbacks=callbacks if callbacks else None,
#     )

#     trainer.train()

#     trainer.model.save_pretrained(out_dir)
#     tokenizer.save_pretrained(out_dir)
#     print(f"✅ Saved to: {out_dir}")


# def main():
#     set_seed(SEED)

#     # ================== 方案1：Stage1 已完成就跳过 ==================
#     if _has_lora_adapter(OUT_STAGE1):
#         print(f"✅ Detected existing Stage1 LoRA at: {OUT_STAGE1} -> skip Stage1")
#     else:
#         train_stage(
#             train_file=STAGE1_TRAIN,
#             out_dir=OUT_STAGE1,
#             init_lora=None,
#             epochs=STAGE1_EPOCHS,
#             lr=LR_STAGE1,
#             swan_name="stage1-long",
#         )

#     # Stage2：短（接着 Stage1 的 LoRA 继续训）
#     train_stage(
#         train_file=STAGE2_TRAIN,
#         out_dir=OUT_STAGE2,
#         init_lora=OUT_STAGE1,
#         epochs=STAGE2_EPOCHS,
#         lr=LR_STAGE2,
#         swan_name="stage2-short",
#     )

#     print("\n✅ Two-stage training finished!")


# if __name__ == "__main__":
#     main()
import os

# ================= 环境变量 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache/transformers"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ================= 固定路径配置 =================
BASE_MODEL = "./qwen_model/Qwen/Qwen2___5-7B-Instruct"

# 只保留 Stage 1 (长文本)
STAGE1_TRAIN = "sft_material_knowledges1/train_500_doubao.jsonl" 
OUT_STAGE1 = "lora_stage1"

# ================= 训练超参 =================
# 🔥 这里控制 Stage 1 的训练轮数，目前是 7
STAGE1_EPOCHS = 7      

LR_STAGE1 = 1e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_LEN = 3072

LOGGING_STEPS = 5
SAVE_STEPS = 100
SEED = 42

# ================= SwanLab（可选） =================
USE_SWANLAB = True
SWAN_PROJECT = "Qwen2.5-LoRA-SFT"
SWAN_MODE_LOCAL = False 
# ==================================================

from typing import Any
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import LoraConfig, get_peft_model, PeftModel

if USE_SWANLAB:
    import swanlab
    from swanlab.integration.transformers import SwanLabCallback


def build_text(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


@dataclass
class Collator:
    tokenizer: Any
    max_length: int

    def __call__(self, feats):
        texts = [x["text"] for x in feats]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def build_lora_cfg():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )


def _load_train_dataset(train_file: str, tokenizer):
    ds = load_dataset("json", data_files={"train": train_file})["train"]

    def map_fn(ex):
        return {"text": build_text(tokenizer, ex["messages"])}

    ds = ds.map(map_fn, remove_columns=ds.column_names)
    return ds


def train_stage(train_file, out_dir, init_lora=None, epochs=4, lr=1e-4, swan_name="stage"):
    print(f"\n🚀 Training on {train_file} -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 强制单卡加载 (cuda:0 对应 CUDA_VISIBLE_DEVICES 指定的那张卡)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},   
        trust_remote_code=True,
        local_files_only=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    lora_cfg = build_lora_cfg()

    if init_lora:
        # 加载现有 LoRA (用于 Stage 2，但在只跑 Stage 1 时通常用不到)
        model = PeftModel.from_pretrained(
            model,
            init_lora,
            is_trainable=True,
        )
        try:
            model.set_adapter("default")
        except Exception:
            pass
    else:
        # 新建 LoRA
        model = get_peft_model(model, lora_cfg)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    if sum(p.requires_grad for p in model.parameters()) == 0:
        raise RuntimeError("❌ 当前模型没有任何可训练参数。")

    ds = _load_train_dataset(train_file, tokenizer)
    collator = Collator(tokenizer, MAX_LEN)

    callbacks = []
    if USE_SWANLAB:
        cb = SwanLabCallback(
            project=SWAN_PROJECT,
            experiment_name=swan_name,
            config={
                "train_file": train_file,
                "out_dir": out_dir,
                "epochs": epochs,
                "lr": lr,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "max_len": MAX_LEN,
            },
            mode="local" if SWAN_MODE_LOCAL else None,
        )
        callbacks.append(cb)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()

    # 保存 LoRA 权重
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✅ Saved Stage 1 to: {out_dir}")


def main():
    set_seed(SEED)

    # 仅执行 Stage 1 训练
    # 这里去掉了检查 _has_lora_adapter 的逻辑，确保每次运行都会重新训练 Stage 1
    # 如果你想保留“已存在则跳过”的功能，可以把下面的 if 语句加回来
    
    print("🚀 Starting Stage 1 (Long Context) Training...")
    
    train_stage(
        train_file=STAGE1_TRAIN,
        out_dir=OUT_STAGE1,
        init_lora=None,       # Stage 1 从头开始，没有初始 LoRA
        epochs=STAGE1_EPOCHS, # 使用配置的 7 epochs
        lr=LR_STAGE1,
        swan_name="stage1-long-only",
    )

    print("\n✅ Stage 1 training finished and saved!")


if __name__ == "__main__":
    main()