import json
import os
from mp_api.client import MPRester
from monty.json import MontyEncoder
from tqdm import tqdm

# === 配置区 ===
API_KEY = "Yw59l1WQM8bDghAXwi9eyvc3f689xkFx"
OUTPUT_FILE = "all_fields.jsonl"

def download_full_dataset():
    # 如果文件存在，先删除，防止追加写入导致混淆
    if os.path.exists(OUTPUT_FILE):
        print(f"⚠️ 检测到旧文件 {OUTPUT_FILE}，正在删除...")
        os.remove(OUTPUT_FILE)

    print("🚀 开始全量下载 Materials Project 所有材料数据...")
    print("⚠️ 注意：这将下载超过 150,000+ 个材料的所有摘要字段，耗时可能较长，请耐心等待。")
    
    with MPRester(API_KEY) as mpr:
        # --- 核心修改：无筛选、无字段限制 ---
        # 1. 不传 fields 参数 -> 获取所有字段
        # 2. 不传 band_gap/energy_above_hull 等参数 -> 获取所有材料
        # 3. mp-api 客户端会自动处理分页，但因为数据量巨大，这就这一步可能会卡很久
        print("\n[Phase 1] 正在向 MP 服务器请求所有材料数据 (Querying)...")
        print("    (这一步可能需要几分钟没有任何输出，因为客户端在后台拼命拉数据，请不要关闭程序)")
        
        try:
            # 使用 search 方法获取所有 summary 数据
            # deprecated=False 意味着只获取当前有效的材料（排除已被弃用的旧ID）
            docs = mpr.materials.summary.search(deprecated=False)
        except Exception as e:
            print(f"❌ 下载过程中出错: {e}")
            print("提示: 如果是超时错误，请检查网络；如果是内存错误，说明你的电脑内存不足以一次性加载所有数据。")
            return

        total_count = len(docs)
        print(f"\n✅ 成功获取数据对象！共检索到 {total_count} 个材料。")
        print(f"[Phase 2] 正在将数据写入文件: {OUTPUT_FILE} ...")

        # --- 写入文件 ---
        # 使用 MontyEncoder 确保 MP 的特殊对象（如 Structure）能被正确转为 JSON
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for doc in tqdm(docs, desc="写入进度"):
                # doc.dict() 会把所有字段转为字典
                # MontyEncoder 会处理 datetime, Structure 等复杂对象
                json_str = json.dumps(doc.dict(), cls=MontyEncoder) 
                f.write(json_str + "\n")

    print(f"\n🎉 完美！全量数据集已保存至: {OUTPUT_FILE}")
    print(f"📊 总计材料数: {total_count}")
    # 打印文件大小
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"💾 文件大小: {file_size:.2f} MB")

if __name__ == "__main__":
    download_full_dataset()