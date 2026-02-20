import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# ===== 把 HF 模型缓存到当前工程目录 =====
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_CACHE"] = "./hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache/transformers"
# ======================================


import json
from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# SparseEncoder 是 sentence-transformers 的稀疏编码器接口（用于 SPLADE 等）
# https://sbert.net/docs/sparse_encoder/pretrained_models.html  (见 Milvus sparse doc 也提到可用 SPLADE) :contentReference[oaicite:3]{index=3}
try:
    from sentence_transformers import SparseEncoder
except Exception:
    SparseEncoder = None

client = MilvusClient(uri="./milvus_lite.db")

# ================= ⚙️ 配置区 =================
MILVUS_URI = "./milvus_lite.db"
COLLECTION = "materials_hybrid"

INPUT_JSONL = "all_fields.jsonl"

# Dense（你现在用 bge-m3）
DENSE_MODEL = "BAAI/bge-m3"

# Sparse（SPLADE）
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"

BATCH_DOCS = 512          # 建议 256~1024，取决于显存/内存
DENSE_BATCH = 64          # 4090 可 64~128
SPARSE_BATCH = 64

MAX_STRUCTURE_SITES = 0   # 保持 0，别把 sites 全塞进 text
# ============================================


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalize_formula_like(s: str) -> str:
    if not s:
        return s
    sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return s.translate(sub_map).strip()


def build_retrieval_text(obj: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    """
    返回: (text, metadata_dict, pk)
    pk 默认用 material_id（如果未来你要 chunk，可改成 f"{mp_id}#{chunk_idx}"）
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

    header = (
        f"ID: {mp_id} | FORMULA: {formula} | "
        f"CrystalSystem: {crystal_system} | SpaceGroup: {sg_symbol} (No.{sg_number}) | "
        f"BandGap(eV): {band_gap} | Eform(eV/atom): {e_form} | Ehull(eV): {e_hull} | "
        f"Stable: {is_stable} | Metal: {is_metal} | DirectGap: {is_gap_direct}"
    )

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
        "material_id": mp_id,
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
    pk = mp_id
    return text, metadata, pk


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
from typing import List, Tuple, Union, Any

def to_milvus_sparse(vec: Any) -> List[Tuple[int, float]]:
    """
    Convert a sparse vector into Milvus SPARSE_FLOAT_VECTOR format:
      List[(dim_index:int, value:float)]

    Supports:
      - torch sparse 1D tensor (preferred)
      - torch dense 1D tensor (will treat non-zeros as sparse)
      - scipy.sparse CSR 1-row matrix / 1D sparse row (optional)

    Notes:
      - Milvus expects indices to be strictly increasing (coalesce helps for torch).
      - Filters out exact 0.0 values.
    """
    # --- torch path ---
    try:
        import torch
        if isinstance(vec, torch.Tensor):
            if vec.dim() != 1:
                raise ValueError(f"Expected 1D tensor, got shape={tuple(vec.shape)}")

            if vec.is_sparse:
                v = vec.coalesce()
                idx = v.indices()[0]   # (nnz,)
                val = v.values()       # (nnz,)
                # Convert to python lists
                idx_list = idx.tolist()
                val_list = val.tolist()
                return [
                    (int(i), float(x))
                    for i, x in zip(idx_list, val_list)
                    if x != 0.0
                ]
            else:
                # dense tensor -> sparse list
                nz = (vec != 0).nonzero(as_tuple=False).flatten()
                if nz.numel() == 0:
                    return []
                idx_list = nz.tolist()
                val_list = vec[nz].tolist()
                return [
                    (int(i), float(x))
                    for i, x in zip(idx_list, val_list)
                    if x != 0.0
                ]
    except Exception:
        # if torch not available or not a torch tensor, continue
        pass

    # --- scipy path (optional) ---
    try:
        import numpy as np
        import scipy.sparse as sp
        if sp.issparse(vec):
            row = vec
            # If it's a matrix with shape (1, V) or (V, ) like sparse row
            if hasattr(row, "tocsr"):
                row = row.tocsr()
            # Flatten to one row if it's (1, V)
            if len(row.shape) == 2 and row.shape[0] == 1:
                indices = row.indices
                data = row.data
                return [
                    (int(i), float(x))
                    for i, x in zip(indices.tolist(), data.tolist())
                    if x != 0.0
                ]
            # If it's (V, 1) or multi-row, you should slice first outside
            raise ValueError(f"Expected a 1-row CSR matrix, got shape={row.shape}")
    except Exception:
        pass

    raise TypeError(f"Unsupported sparse vector type: {type(vec)}")


def ensure_collection(client: MilvusClient, dense_dim: int, recreate: bool = False):
    if client.has_collection(COLLECTION):
        if recreate:
            print(f"⚠️ collection {COLLECTION} exists, dropping and recreating ...")
            client.drop_collection(COLLECTION)
        else:
            return

    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,   # ✅ 注意：单数
    )
    schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="dense_vec", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
    schema.add_field(field_name="sparse_vec", datatype=DataType.SPARSE_FLOAT_VECTOR)

    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        consistency_level="Strong",
    )


    # client.create_index(
    #     collection_name=COLLECTION,
    #     field_name="dense_vec",
    #     index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
    # )
    # client.create_index(
    #     collection_name=COLLECTION,
    #     field_name="sparse_vec",
    #     index_params={"index_type": "AUTOINDEX", "metric_type": "IP"},
    # )


def main():
    print("🚀 Build Milvus Hybrid Index (Dense=bge-m3 + Sparse=SPLADE) ...")
    print(f"📥 INPUT: {INPUT_JSONL}")
    print(f"🌐 Milvus: {MILVUS_URI}")
    print(f"📚 Collection: {COLLECTION}")
    print(f"🧠 Dense: {DENSE_MODEL}")
    print(f"🧠 Sparse: {SPLADE_MODEL}")

    client = MilvusClient(uri=MILVUS_URI)

    # Dense encoder
    dense = SentenceTransformer(DENSE_MODEL, device="cuda")
    dense_dim = dense.get_sentence_embedding_dimension()

    # Sparse encoder
    if SparseEncoder is None:
        raise RuntimeError(
            "你的 sentence-transformers 版本不包含 SparseEncoder。\n"
            "请升级：pip install -U sentence-transformers"
        )
    sparse = SparseEncoder(SPLADE_MODEL, device="cuda")

    ensure_collection(client, dense_dim, recreate=False)

    buffer_texts: List[str] = []
    buffer_pks: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []
    total = 0
    kept = 0

    for ln, obj in tqdm(iter_jsonl(INPUT_JSONL), desc="Reading jsonl"):
        total += 1
        if "material_id" not in obj and "source" not in obj:
            continue
        if "formula_pretty" not in obj:
            continue

        text, meta, pk = build_retrieval_text(obj)
        buffer_texts.append(text)
        buffer_metas.append(meta)
        buffer_pks.append(pk)
        kept += 1

        if len(buffer_texts) >= BATCH_DOCS:
            # === encode dense
            dense_vecs = dense.encode(
                buffer_texts,
                batch_size=DENSE_BATCH,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            # === encode sparse (CSR matrix list)
            # SparseEncoder.encode 通常返回 scipy.sparse matrix（每行一个文档）
            sparse_mat = sparse.encode(
                buffer_texts,
                batch_size=SPARSE_BATCH,
                show_progress_bar=False,
                convert_to_tensor=True,   # ✅ 强制 torch.Tensor
            )

            # sparse_mat: torch.Tensor, shape (B, V)
            sparse_vecs = [to_milvus_sparse(sparse_mat[i]) for i in range(sparse_mat.shape[0])]


            data = []
            for pk, text, dv, sv, meta in zip(buffer_pks, buffer_texts, dense_vecs, sparse_vecs, buffer_metas):
                item = {
                    "pk": pk,
                    "text": text,
                    "dense_vec": dv,
                    "sparse_vec": sv,
                }
                # 动态字段：直接合并 meta（Milvus enable_dynamic_fields=True）
                item.update(meta)
                data.append(item)

            client.insert(collection_name=COLLECTION, data=data)
            print(f"✅ inserted {len(data)} | kept={kept} / read={total} (last line={ln})")

            buffer_texts.clear()
            buffer_pks.clear()
            buffer_metas.clear()

    # flush last
    if buffer_texts:
        dense_vecs = dense.encode(
            buffer_texts,
            batch_size=DENSE_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        sparse_mat = sparse.encode(
            buffer_texts,
            batch_size=SPARSE_BATCH,
            show_progress_bar=False,
            convert_to_tensor=True,   # ✅ 强制 torch.Tensor
        )

        # sparse_mat: torch.Tensor, shape (B, V)
        sparse_vecs = [to_milvus_sparse(sparse_mat[i]) for i in range(sparse_mat.shape[0])]

        data = []
        for pk, text, dv, sv, meta in zip(buffer_pks, buffer_texts, dense_vecs, sparse_vecs, buffer_metas):
            item = {"pk": pk, "text": text, "dense_vec": dv, "sparse_vec": sv}
            item.update(meta)
            data.append(item)

        client.insert(collection_name=COLLECTION, data=data)
        print(f"✅ inserted last {len(data)} | kept={kept} / read={total}")

    # load into memory (for search)
    client.load_collection(collection_name=COLLECTION)
    print("🎉 Done. Milvus hybrid collection is ready.")
    


if __name__ == "__main__":
    main()
