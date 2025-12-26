import json
import pandas as pd
import os

# ================= 配置区域 =================
DATA_DIR = './data/'  # 请确保这里是你存放数据的路径
BUSINESS_FILE = os.path.join(DATA_DIR, 'business.json')
CHECKIN_FILE = os.path.join(DATA_DIR, 'checkin.json')

# 想要查看的 Top N 数量
TOP_N = 20

# 广泛的餐饮/排队相关类别列表
TARGET_CATEGORIES = [
    'Food', 'Restaurants', 'Coffee & Tea', 'Bubble Tea', 
    'Bakeries', 'Desserts', 'Sandwiches', 'Burgers', 
    'Breakfast & Brunch', 'Ice Cream & Frozen Yogurt',
    'Donuts', 'Cafes'
]
# ===========================================

def find_high_traffic_businesses():
    print("--- [Step 1] 正在统计 checkin.json 中的签到数量 ---")
    checkin_counts = {}
    
    count = 0
    with open(CHECKIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            if count % 100000 == 0:
                print(f"已扫描 {count} 条签到记录...", end='\r')
            
            try:
                data = json.loads(line)
                bid = data['business_id']
                # 计算这个商家的签到总数 (逗号分隔的时间戳)
                num_checkins = data['date'].count(',') + 1
                checkin_counts[bid] = num_checkins
            except Exception:
                continue
    
    print(f"\ncheckin.json 扫描完成，共获取 {len(checkin_counts)} 个商家的签到统计。")

    print("\n--- [Step 2] 正在匹配 business.json 信息 ---")
    results = []
    
    with open(BUSINESS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                bid = data['business_id']
                
                # 如果这个商家有签到数据
                if bid in checkin_counts:
                    # 检查是否还在营业 (可选，如果只做历史仿真建议注释掉这行，保留更多数据)
                    # if data['is_open'] == 0: continue
                    
                    # 检查类别是否相关
                    cats = data.get('categories')
                    if cats and any(c in cats for c in TARGET_CATEGORIES):
                        results.append({
                            'name': data['name'],
                            'business_id': bid,  # 这里是你需要的 ID
                            'city': data['city'],
                            'total_checkins': checkin_counts[bid],
                            'categories': data['categories'] # 保留类别辅助判断，不打印也可以
                        })
            except Exception:
                continue

    # 3. 排序并展示
    df = pd.DataFrame(results)
    if df.empty:
        print("未找到匹配的商家。")
        return

    # 按 check-in 数量倒序排列，取前 TOP_N
    df_sorted = df.sort_values(by='total_checkins', ascending=False).head(TOP_N)
    
    print(f"\n--- Check-in 数量最多的 Top {TOP_N} 商家 ---")
    
    # 设置显示选项，防止列被折叠
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # 打印你指定的列：Name, ID, City, Total Checkins
    # 我保留了 index (最左边的排名)，方便你对应
    print(df_sorted[['name', 'business_id', 'city', 'total_checkins']].to_string())

if __name__ == "__main__":
    find_high_traffic_businesses()