import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"   # <- 你的卡号
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import json
import re
from typing import Dict, Any, List, Tuple, Optional

import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ========= 必须最前面配置环境 =========

# ===================================


# ========= 你需要改的配置 =========
INPUT_JSONL = "all_fields.jsonl"           # <- 你的原始数据（jsonl）
OUTPUT_INDEX_DIR = "./material_rag_index_bgem3"  # <- 输出索引目录

EMBEDDING_MODEL = "BAAI/bge-m3"  # 多语言检索强

BATCH_DOCS = 2048               # 每批加入向量库的文档数（可调）
EMBED_BATCH_SIZE = 64           # embedding 批大小（4090 可适当加到 96/128）
MAX_STRUCTURE_SITES = 0         # 0 表示完全不写 sites；建议保持 0（否则文本爆炸）
# ===================================


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalize_formula_like(s: str) -> str:
    """把用户常见的下标形式统一一下（用于写入索引文本头部）。"""
    if not s:
        return s
    sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return s.translate(sub_map).strip()


def build_retrieval_text(obj: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    把一条材料记录压缩为“检索友好文本 + metadata”。

    重点：
    - 文本：只保留对检索/问答最有用的字段（band_gap / e_form / spacegroup / lattice参数等）
    - metadata：保留 material_id / formula / 关键性质，方便前端引用
    """
    formula = normalize_formula_like(obj.get("formula_pretty", "Unknown"))
    mp_id = obj.get("material_id", obj.get("source", "Unknown"))

    band_gap = obj.get("band_gap", None)
    e_form = obj.get("formation_energy_per_atom", None)
    e_hull = obj.get("energy_above_hull", None)
    is_stable = obj.get("is_stable", None)
    is_metal = obj.get("is_metal", None)
    is_gap_direct = obj.get("is_gap_direct", None)
    efermi = obj.get("efermi", None)
    density = obj.get("density", None)
    volume = obj.get("volume", None)

    sym = obj.get("symmetry", {}) or {}
    crystal_system = sym.get("crystal_system", None)
    sg_symbol = sym.get("symbol", None)
    sg_number = sym.get("number", None)
    point_group = sym.get("point_group", None)

    lattice = safe_get(obj, ["structure", "lattice"], {}) or {}
    a = lattice.get("a", None)
    b = lattice.get("b", None)
    c = lattice.get("c", None)
    alpha = lattice.get("alpha", None)
    beta = lattice.get("beta", None)
    gamma = lattice.get("gamma", None)

    # 组装“检索头”——非常关键：能显著提高“直接命中材料”的概率
    header = (
        f"ID: {mp_id} | FORMULA: {formula} | "
        f"CrystalSystem: {crystal_system} | SpaceGroup: {sg_symbol} (No.{sg_number}) | "
        f"BandGap(eV): {band_gap} | Eform(eV/atom): {e_form} | Ehull(eV): {e_hull} | "
        f"Stable: {is_stable} | Metal: {is_metal} | DirectGap: {is_gap_direct}"
    )

    # 组装“正文”——给 LLM 展开讲原因用（但仍避免结构 sites 噪音）
    body_lines = [
        f"Material {formula} ({mp_id}) basic properties:",
        f"- band_gap (eV): {band_gap}",
        f"- formation_energy_per_atom (eV/atom): {e_form}",
        f"- energy_above_hull (eV): {e_hull}",
        f"- is_stable: {is_stable}",
        f"- is_metal: {is_metal}",
        f"- is_gap_direct: {is_gap_direct}",
        f"- efermi (eV): {efermi}",
        f"- density (g/cc): {density}",
        f"- volume (A^3): {volume}",
        "",
        "Symmetry:",
        f"- crystal_system: {crystal_system}",
        f"- spacegroup_symbol: {sg_symbol}",
        f"- spacegroup_number: {sg_number}",
        f"- point_group: {point_group}",
        "",
        "Lattice parameters:",
        f"- a,b,c (A): {a}, {b}, {c}",
        f"- alpha,beta,gamma (deg): {alpha}, {beta}, {gamma}",
    ]

    # （可选）结构sites千万别全写！如果你真想保留一点点，只保留元素列表或前几个site
    if MAX_STRUCTURE_SITES and MAX_STRUCTURE_SITES > 0:
        sites = safe_get(obj, ["structure", "sites"], []) or []
        sites = sites[:MAX_STRUCTURE_SITES]
        body_lines.append("")
        body_lines.append(f"Structure sites (first {len(sites)} shown):")
        for s in sites:
            label = s.get("label", "")
            abc = s.get("abc", None)
            xyz = s.get("xyz", None)
            species = s.get("species", [])
            elem = species[0].get("element", "") if species else ""
            body_lines.append(f"- {label}/{elem} abc={abc} xyz={xyz}")

    text = "[RETRIEVAL_HEADER]\n" + header + "\n\n" + "[CONTENT]\n" + "\n".join(body_lines)

    metadata = {
        "source": mp_id,
        "formula": formula,
        "band_gap": band_gap,
        "formation_energy_per_atom": e_form,
        "energy_above_hull": e_hull,
        "is_stable": is_stable,
        "is_metal": is_metal,
        "is_gap_direct": is_gap_direct,
        "efermi": efermi,
        "density": density,
        "volume": volume,
        "crystal_system": crystal_system,
        "spacegroup_symbol": sg_symbol,
        "spacegroup_number": sg_number,
        "point_group": point_group,
        "a": a, "b": b, "c": c,
        "alpha": alpha, "beta": beta, "gamma": gamma,
    }

    return text, metadata


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception as e:
                print(f"⚠️ JSON parse error at line {ln}: {e}")
                continue


def main():
    print("🚀 Building FAISS index from jsonl (150k materials) ...")
    print(f"📥 INPUT:  {INPUT_JSONL}")
    print(f"📦 OUTPUT: {OUTPUT_INDEX_DIR}")
    print(f"🧠 EMBEDDING: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBED_BATCH_SIZE,
        },
    )

    vector_db: Optional[FAISS] = None
    buffer_texts: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []

    total = 0
    kept = 0

    for ln, obj in iter_jsonl(INPUT_JSONL):
        total += 1

        # 基本字段检查
        if "material_id" not in obj and "source" not in obj:
            continue
        if "formula_pretty" not in obj:
            continue

        text, meta = build_retrieval_text(obj)

        buffer_texts.append(text)
        buffer_metas.append(meta)
        kept += 1

        # 分批写入向量库（避免一次性占用过大内存）
        if len(buffer_texts) >= BATCH_DOCS:
            if vector_db is None:
                vector_db = FAISS.from_texts(buffer_texts, embeddings, metadatas=buffer_metas)
            else:
                vector_db.add_texts(buffer_texts, metadatas=buffer_metas)

            print(f"✅ added {len(buffer_texts)} docs | total kept={kept} / read={total} (last line={ln})")

            buffer_texts.clear()
            buffer_metas.clear()

    # flush 最后一批
    if buffer_texts:
        if vector_db is None:
            vector_db = FAISS.from_texts(buffer_texts, embeddings, metadatas=buffer_metas)
        else:
            vector_db.add_texts(buffer_texts, metadatas=buffer_metas)

        print(f"✅ added last {len(buffer_texts)} docs | total kept={kept} / read={total}")

    if vector_db is None:
        raise RuntimeError("❌ 没有成功写入任何文档，请检查输入文件格式。")

    vector_db.save_local(OUTPUT_INDEX_DIR)
    print("🎉 Done. FAISS index saved.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
