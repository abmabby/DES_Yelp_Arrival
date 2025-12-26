import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import warnings
import pytz  # <--- 新增：处理时区

warnings.filterwarnings('ignore')

# ==================== 核心配置区域 ====================
TARGET_BUSINESS_ID = "FEXhWNCMkv22qG04E83Qjg"
ESTIMATED_DAILY_VISITORS = 3000 
DATA_DIR = './data/'
BUSINESS_FILE = os.path.join(DATA_DIR, 'business.json')
CHECKIN_FILE = os.path.join(DATA_DIR, 'checkin.json')

# [新增] 目标商家的时区 (新奥尔良属于中部时间)
TARGET_TIMEZONE = 'America/Chicago' 
# ====================================================

class YelpDESGenerator:
    def __init__(self, business_id):
        self.bid = business_id
        self.meta = {}      
        self.df_raw = None  
        self.df_clean = None 
        self.target_year = None
        self.operating_hours = {} 
    
    def step1_load_metadata(self):
        """阶段一：加载商家元数据与营业时间解析"""
        print(f"--- [Step 1] 加载商家元数据 (ID: {self.bid}) ---")
        found = False
        with open(BUSINESS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data['business_id'] == self.bid:
                        self.meta = data
                        found = True
                        break
                except: continue
        
        if not found: raise ValueError("未找到商家ID！")
            
        print(f"   商家名称: {self.meta.get('name')}")
        print(f"   所在城市: {self.meta.get('city')}")
        print(f"   [设定时区]: {TARGET_TIMEZONE}")
        
        raw_hours = self.meta.get('hours', {})
        days_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
        
        print("   [营业时间解析]:")
        for day_str, time_range in raw_hours.items():
            if day_str in days_map:
                try:
                    start_str, end_str = time_range.split('-')
                    s_h = int(start_str.split(':')[0])
                    e_h = int(end_str.split(':')[0])
                    self.operating_hours[days_map[day_str]] = (s_h, e_h)
                    print(f"     - {day_str}: {s_h}:00 - {e_h}:00")
                except: pass
    
    def step2_load_and_filter_data(self):
        """阶段二：加载 Check-in 并转换时区"""
        print("\n--- [Step 2] 加载 Check-in 数据并转换时区 ---")
        dates = []
        with open(CHECKIN_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['business_id'] == self.bid:
                    # 读取原始字符串
                    raw_dates = [datetime.strptime(d.strip(), "%Y-%m-%d %H:%M:%S") for d in data['date'].split(',')]
                    # [关键修正] UTC 转 Local Time
                    utc = pytz.utc
                    local_tz = pytz.timezone(TARGET_TIMEZONE)
                    
                    for dt in raw_dates:
                        # 1. 设为 UTC
                        dt_utc = utc.localize(dt)
                        # 2. 转为当地时间
                        dt_local = dt_utc.astimezone(local_tz)
                        # 3. 转回无时区的 datetime (方便 pandas 处理)
                        dates.append(dt_local.replace(tzinfo=None))
                    break
        
        self.df_raw = pd.DataFrame({'dt': dates})
        self.df_raw['year'] = self.df_raw['dt'].dt.year
        
        yearly_counts = self.df_raw['year'].value_counts()
        best_year = yearly_counts.idxmax()
        count = yearly_counts.max()
        
        self.target_year = best_year
        print(f"   自动锁定最佳年份: {best_year} (数据量: {count} 条)")
        
        self.df_clean = self.df_raw[self.df_raw['year'] == best_year].copy()
        
    def step3_noise_removal(self):
        """阶段三：基于营业时间的噪点剔除"""
        print("\n--- [Step 3] 噪点剔除 (Noise Removal - Local Time) ---")
        original_count = len(self.df_clean)
        
        self.df_clean['weekday'] = self.df_clean['dt'].dt.dayofweek
        self.df_clean['hour'] = self.df_clean['dt'].dt.hour
        
        def is_open(row):
            day = row['weekday']
            h = row['hour']
            if day in self.operating_hours:
                start, end = self.operating_hours[day]
                # Cafe Du Monde 这种 8-20 的简单逻辑
                if start <= h < end:
                    return True
            return False

        self.df_clean = self.df_clean[self.df_clean.apply(is_open, axis=1)]
        new_count = len(self.df_clean)
        
        print(f"   剔除非营业时间数据: {original_count - new_count} 条")
        print(f"   有效仿真样本数: {new_count} 条 (保留率: {new_count/original_count:.1%})")

    def step4_statistical_validation(self):
        """阶段四：统计检验"""
        print("\n--- [Step 4] 分布拟合检验 (Statistical Validation) ---")
        
        # 寻找高峰小时
        hourly_counts = self.df_clean['hour'].value_counts()
        peak_hour = hourly_counts.idxmax()
        
        print(f"   识别到年度高峰时段: {peak_hour}:00 - {peak_hour+1}:00 (当地时间)")
        
        peak_data = self.df_clean[self.df_clean['hour'] == peak_hour].sort_values('dt')
        
        inter_arrivals = peak_data['dt'].diff().dropna().dt.total_seconds() / 60.0
        # 稍微放宽一点间隔过滤，防止误删
        inter_arrivals = inter_arrivals[inter_arrivals <= 60]
        
        loc, scale = stats.expon.fit(inter_arrivals, floc=0)
        ks_stat, p_value = stats.kstest(inter_arrivals, 'expon', args=(loc, scale))
        
        print(f"   [K-S Test 结果]")
        print(f"   - P-value: {p_value:.4e}")
        if p_value > 0.05:
            print("   - 结论: ✅ 完美符合指数分布")
        else:
            print("   - 结论: ⚠️ P值显著 (Visual Check Required)")
            
        plt.figure(figsize=(10, 5))
        plt.hist(inter_arrivals, bins=30, density=True, alpha=0.6, color='skyblue', label='Real Data')
        x = np.linspace(0, inter_arrivals.max(), 100)
        plt.plot(x, stats.expon.pdf(x, loc, scale), 'r-', lw=2, label=f'Exponential Fit')
        plt.title(f"Model Validation: Peak Hour ({peak_hour}:00) Inter-arrival Times")
        plt.legend()
        plt.show()

    def step5_generate_parameters(self):
        """阶段五：参数生成"""
        print("\n--- [Step 5] 生成仿真参数 (Scaling & Export) ---")
        
        hourly_distribution = self.df_clean['hour'].value_counts(normalize=True).sort_index()
        real_lambda = hourly_distribution * ESTIMATED_DAILY_VISITORS
        
        plt.figure(figsize=(10, 5))
        plt.bar(real_lambda.index, real_lambda.values, color='orange', alpha=0.7)
        plt.title(f"Generated Arrival Pattern ({self.target_year} - Local Time)")
        plt.xlabel("Hour of Day")
        plt.xticks(range(8, 21))
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        
        print("\n✅ [最终产出] Python DES 参数字典：")
        print("="*60)
        print("# Arrival Rates (lambda) - Corrected for Timezone")
        print(f"# Source: {self.meta['name']} ({self.target_year})")
        print("arrival_rate_schedule = {")
        for h, rate in real_lambda.items():
            print(f"    {int(h)}: {rate:.2f},  # {int(h)}:00 - {int(h)+1}:00")
        print("}")
        print("="*60)

if __name__ == "__main__":
    generator = YelpDESGenerator(TARGET_BUSINESS_ID)
    generator.step1_load_metadata()
    generator.step2_load_and_filter_data()
    generator.step3_noise_removal()
    generator.step4_statistical_validation()
    generator.step5_generate_parameters()