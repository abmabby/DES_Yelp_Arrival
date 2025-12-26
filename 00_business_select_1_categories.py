# 找特定类别的商家, 并提取其签到数据, 找出最多的一个分析看看是否合适
# 结论:发现一个符合要求的面包店, 但是它尽管是checkin数据最多的2015年也只有一千多条, 感觉还是不够用

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os


# ================= 配置区域 =================
# 数据文件路径 (请修改为你本地的路径)
DATA_DIR = './data/'  # 假设文件在当前目录
BUSINESS_FILE = os.path.join(DATA_DIR, 'business.json')
CHECKIN_FILE = os.path.join(DATA_DIR, 'checkin.json')

# 目标类别 (Bubble Tea, 网红烘焙)
TARGET_CATEGORIES = ['Bubble Tea', 'Coffee & Tea', 'Bakeries', 'Desserts']

# 筛选门槛：评论数太少的店数据肯定不够做仿真
MIN_REVIEW_COUNT = 500 

# 仿真聚焦的年份 (建议选疫情前，数据最稳定)
TARGET_YEAR = 2015
# ===========================================

def step1_filter_businesses():
    """
    第一步：扫描 business.json，只保留符合条件的目标商家ID
    """
    print(f"--- [Step 1] 正在流式读取 {BUSINESS_FILE} 筛选商家 ---")
    candidates = []
    count = 0
    
    with open(BUSINESS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            if count % 100000 == 0:
                print(f"已扫描 {count} 个商家...", end='\r')
                
            try:
                data = json.loads(line)
                # 过滤已关闭的店 (可选，如果只想做纯历史数据仿真，可注释掉此行)
                if data['is_open'] == 0: 
                    continue
                    
                cats = data.get('categories')
                if cats and any(c in cats for c in TARGET_CATEGORIES):
                    if data['review_count'] >= MIN_REVIEW_COUNT:
                        candidates.append({
                            'business_id': data['business_id'],
                            'name': data['name'],
                            'review_count': data['review_count'],
                            'stars': data['stars'],
                            'city': data['city'],
                            'state': data['state'],
                            'categories': data['categories']
                        })
            except Exception as e:
                continue
                
    print(f"\n筛选完成！共找到 {len(candidates)} 个候选商家。")
    return pd.DataFrame(candidates).sort_values('review_count', ascending=False)

def step2_extract_checkins(target_ids):
    """
    第二步：扫描 checkin.json，只提取候选名单中的商家数据
    为了节省内存，只提取特定ID的数据
    """
    print(f"--- [Step 2] 正在流式读取 {CHECKIN_FILE} 提取签到数据 ---")
    
    # 使用 Set 提高查询速度 (O(1))
    target_id_set = set(target_ids)
    extracted_data = {} # {business_id: [datetime_objects]}
    
    count = 0
    with open(CHECKIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            if count % 1000000 == 0: # 每100万行提示一次
                print(f"已处理 {count // 1000000} 百万条签到记录...", end='\r')
            
            try:
                data = json.loads(line)
                bid = data['business_id']
                
                if bid in target_id_set:
                    # 提取时间字符串
                    date_str = data['date']
                    # 解析为 datetime 对象列表
                    dates = [datetime.strptime(d.strip(), "%Y-%m-%d %H:%M:%S") 
                             for d in date_str.split(',')]
                    extracted_data[bid] = dates
            except Exception:
                continue
                
    print(f"\n提取完成！获取了 {len(extracted_data)} 个商家的签到详情。")
    return extracted_data

def step3_analyze_target(business_info, dates):
    """
    第三步：深入分析选中的 Top 1 商家，检查 2019 年的数据密度
    """
    name = business_info['name']
    print(f"\n--- [Step 3] 分析目标商家: {name} ---")
    
    df = pd.DataFrame({'dt': dates})
    
    # 1. 年度分布检查 (查看2010-2022的走势)
    df['year'] = df['dt'].dt.year
    year_counts = df['year'].value_counts().sort_index()
    
    print("\n年度签到分布:")
    print(year_counts.to_string())
    
    # 2. 聚焦目标年份 (比如 2019)
    df_target = df[df['year'] == TARGET_YEAR].copy()
    count_target_year = len(df_target)
    
    print(f"\n{TARGET_YEAR} 年总签到数: {count_target_year}")
    
    if count_target_year == 0:
        print(f"⚠️ 警告: 该商家在 {TARGET_YEAR} 年没有数据！")
        return

    # 3. 计算周均和日均 (用于后续仿真参数 lambda)
    # 简单估算：Yelp Check-in 只是真实客流的一小部分
    # 假设 Multiplier (真实客流系数) = 10 (参考经验值，或后续做敏感性分析)
    MULTIPLIER = 10 
    est_real_visits = count_target_year * MULTIPLIER
    avg_daily_visits = est_real_visits / 365
    print(f"估算真实年客流 (系数x{MULTIPLIER}): {est_real_visits}")
    print(f"估算日均客流: {avg_daily_visits:.1f} 人/天")

    # 4. 绘图
    plt.figure(figsize=(14, 6))
    
    # 子图1: 年度趋势
    plt.subplot(1, 2, 1)
    year_counts.plot(kind='bar', color='#1f77b4')
    plt.title(f'Yearly Check-ins: {name}')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    
    # 子图2: 目标年份的小时分布 (Peak Hour Identification)
    df_target['hour'] = df_target['dt'].dt.hour
    hour_counts = df_target['hour'].value_counts().sort_index()
    
    plt.subplot(1, 2, 2)
    hour_counts.plot(kind='bar', color='#ff7f0e')
    plt.title(f'Hourly Distribution in {TARGET_YEAR}')
    plt.xlabel('Hour of Day (0-23)')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 返回数据以供后续步骤使用
    return df_target

# ================= 主流程 =================
if __name__ == "__main__":
    # 1. 筛选商家
    df_candidates = step1_filter_businesses()
    
    if len(df_candidates) > 0:
        # 取 Top 5 看看
        print("\n候选商家 Top 5:")
        print(df_candidates[['name', 'city', 'review_count', 'stars']].head(5))
        
        # 2. 选择 Top 1 进行深入数据提取
        # 注意：这里我们提取 Top 5 的所有数据，防止 Top 1 数据质量不好需要换
        top_5_ids = df_candidates.head(5)['business_id'].tolist()
        all_checkins = step2_extract_checkins(top_5_ids)
        
        # 3. 分析 Top 1
        target_business = df_candidates.iloc[1]
        target_id = target_business['business_id']
        
        if target_id in all_checkins:
            target_dates = all_checkins[target_id]
            # 运行分析
            step3_analyze_target(target_business, target_dates)
        else:
            print("错误：未找到目标商家的签到数据。")
    else:
        print("未找到符合条件的商家。")