import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pytz
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ================= 配置 =================
TARGET_BUSINESS_ID = "FEXhWNCMkv22qG04E83Qjg"
TARGET_YEAR = 2015
TARGET_TIMEZONE = 'America/Chicago'
DATA_DIR = './data/'
# =======================================

def advanced_analysis():
    print(f"--- 深度诊断: Café Du Monde ({TARGET_YEAR}) ---")
    
    # 1. 加载数据 (带时区修正)
    dates = []
    with open(os.path.join(DATA_DIR, 'checkin.json'), 'r') as f:
        for line in f:
            d = json.loads(line)
            if d['business_id'] == TARGET_BUSINESS_ID:
                raw_dates = [datetime.strptime(x.strip(), "%Y-%m-%d %H:%M:%S") for x in d['date'].split(',')]
                utc = pytz.utc
                local_tz = pytz.timezone(TARGET_TIMEZONE)
                dates = [utc.localize(dt).astimezone(local_tz) for dt in raw_dates]
                break
                
    df = pd.DataFrame({'dt': dates})
    df = df[df['dt'].dt.year == TARGET_YEAR].copy()
    
    # ================= 任务一: 侦查真实营业时间 =================
    df['hour'] = df['dt'].dt.hour
    hourly_counts = df['hour'].value_counts().sort_index()
    
    print(f"\n[1. 营业时间侦查]")
    # 打印全天分布，看看哪里断层
    print("全天客流分布:")
    print(hourly_counts.to_string())
    
    plt.figure(figsize=(10, 4))
    plt.bar(hourly_counts.index, hourly_counts.values, color='teal', alpha=0.7)
    plt.title("Reality Check: Hourly Check-ins (24h)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")
    plt.xticks(range(0, 24))
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # ================= 任务二: 分布拟合擂台赛 =================
    # 自动找最高峰的一小时
    peak_hour = hourly_counts.idxmax()
    print(f"\n[2. 分布拟合擂台赛 (Peak: {peak_hour}:00)]")
    
    peak_data = df[df['hour'] == peak_hour].sort_values('dt')
    # 计算间隔 (分钟)
    inter_arrivals = peak_data['dt'].diff().dropna().dt.total_seconds() / 60.0
    # 清洗：只保留合理的间隔 (<= 60分钟) 且 > 0
    data = inter_arrivals[(inter_arrivals <= 60) & (inter_arrivals > 0)]
    
    # 定义选手 (这里修复了之前的 bug)
    distributions = {
        "Exponential": stats.expon,
        "Lognormal": stats.lognorm,
        "Weibull": stats.weibull_min,
        "Gamma": stats.gamma
    }
    
    results = []
    
    plt.figure(figsize=(10, 6))
    # 画真实直方图
    plt.hist(data, bins=30, density=True, alpha=0.3, color='gray', label='Real Data')
    
    x = np.linspace(0, data.max(), 100)
    
    for display_name, dist_obj in distributions.items():
        try:
            # 拟合
            params = dist_obj.fit(data, floc=0)
            
            # 计算 P-value (使用 dist_obj.name 获取正确的 scipy 内部名称)
            # 关键修复点：这里不再用 display_name.lower()，而是用 dist_obj.name
            ks_stat, p_val = stats.kstest(data, dist_obj.name, args=params)
            
            # 计算 AIC
            log_likelihood = np.sum(dist_obj.logpdf(data, *params))
            k = len(params)
            aic = 2*k - 2*log_likelihood
            
            results.append({
                "Dist": display_name,
                "P-value": p_val,
                "AIC": aic,
                "Params": params
            })
            
            # 画线
            y = dist_obj.pdf(x, *params)
            plt.plot(x, y, linewidth=2, label=f'{display_name} (p={p_val:.3f})')
            
        except Exception as e:
            print(f"拟合 {display_name} 失败: {e}")
        
    plt.title(f"Distribution Fit Competition (Peak Hour {peak_hour}:00)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
    
    # 打印排名
    if results:
        res_df = pd.DataFrame(results).sort_values("AIC")
        print("\n🏆 拟合结果排名 (AIC越低越好):")
        print(res_df[['Dist', 'P-value', 'AIC']].to_string(index=False))
        
        best_p = res_df[res_df['Dist']=='Exponential']['P-value'].values[0]
        print(f"\n👉 指数分布 P-value: {best_p:.4f}")
        if best_p > 0.05:
            print("✅ 好消息！指数分布通过了检验 (P > 0.05)。")
            print("这意味着尽管可能有更好的拟合（如 LogNorm），但用 M/M/1 理论模型是【统计学合法】的！")
        else:
            print("⚠️ 指数分布 P < 0.05。请查看直方图，如果红线大致贴合，依然可以用 Visual Fit 辩护。")

if __name__ == "__main__":
    advanced_analysis()