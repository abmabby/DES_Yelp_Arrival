import json
import pandas as pd
import os

# ================= 配置区域 =================
DATA_DIR = './data/'
BUSINESS_FILE = os.path.join(DATA_DIR, 'business.json')
CHECKIN_FILE = os.path.join(DATA_DIR, 'checkin.json')
TOP_N = 25  # 看前25个
# ===========================================

def scan_absolute_top():
    print("--- [Step 1] 正在全量扫描 checkin.json (这可能需要几十秒) ---")
    checkin_counts = {}
    
    # 1. 统计每个 ID 的签到数
    with open(CHECKIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                bid = data['business_id']
                # 计算逗号数量+1 = 签到次数
                count = data['date'].count(',') + 1
                checkin_counts[bid] = count
            except: continue
            
    print(f"扫描完成，共获取 {len(checkin_counts)} 个商家的统计数据。")
    
    # 2. 找出 Top N 的 ID
    # 按 value (数量) 倒序排序，取前 Top N
    top_ids = sorted(checkin_counts, key=checkin_counts.get, reverse=True)[:TOP_N]
    top_id_set = set(top_ids)
    
    print("--- [Step 2] 正在匹配商家详细信息 ---")
    results = []
    
    with open(BUSINESS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data['business_id'] in top_id_set:
                    results.append({
                        'name': data['name'],
                        'checkins': checkin_counts[data['business_id']],
                        'city': data['city'],
                        'state': data['state'],
                        'categories': data['categories']
                    })
            except: continue
            
    # 3. 展示结果
    df = pd.DataFrame(results)
    df = df.sort_values('checkins', ascending=False).reset_index(drop=True)
    
    print(f"\n🏆 Yelp 数据集 Check-in 总榜 Top {TOP_N} 🏆")
    # 设置显示格式，防止类别显示不全
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 50) # 限制类别列宽，防止换行太乱
    pd.set_option('display.width', 1000)
    
    print(df[['name', 'checkins', 'city']].to_string())

if __name__ == "__main__":
    scan_absolute_top()