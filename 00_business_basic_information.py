import json
import os

# ================= 配置区域 =================
DATA_DIR = './data/'  # 请确认路径
BUSINESS_FILE = os.path.join(DATA_DIR, 'business.json')

# 请将上一轮找到的 Café Du Monde 的 ID 填入这里
# 例如: '4p5K8...'(这是示例，请填入真实的 ID)
TARGET_BUSINESS_ID = "FEXhWNCMkv22qG04E83Qjg" 
# ===========================================

def inspect_business_details():
    print(f"--- 正在搜索 ID: {TARGET_BUSINESS_ID} ---")
    found = False
    
    with open(BUSINESS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data['business_id'] == TARGET_BUSINESS_ID:
                    print("\n✅ 找到目标商家！详细信息如下：\n")
                    print("="*60)
                    
                    # 使用 json.dumps 漂亮地打印所有字段
                    print(json.dumps(data, indent=4, ensure_ascii=False))
                    
                    print("="*60)
                    found = True
                    break  # 找到后直接退出，节省时间
            except Exception as e:
                continue
    
    if not found:
        print(f"❌ 未找到 ID 为 {TARGET_BUSINESS_ID} 的商家，请检查 ID 是否正确。")

if __name__ == "__main__":
    inspect_business_details()