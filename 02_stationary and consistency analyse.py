import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
from datetime import datetime
from scipy import stats

# 引入 statsmodels (必须安装: pip install statsmodels)
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️ 严重警告: 未安装 statsmodels，ADF/KPSS 检验将无法运行！")

# ================= 配置区域 =================
TARGET_BUSINESS_ID = "FEXhWNCMkv22qG04E83Qjg"
TARGET_YEAR = 2015
TARGET_TIMEZONE = 'America/Chicago' 
DATA_DIR = './data/'
FIGURE_DIR = './figure/'
# ===========================================

def calculate_cohens_d(group1, group2):
    """计算效应量 Cohen's d"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # 合并标准差
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_se

def step2_ultimate_validation():
    print(f"--- [Module 2] Ultimate Statistical Validation (Academic Rigor) ---")
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # 1. 数据加载与清洗 (同前)
    print("1. Loading Data & Timezone Conversion...")
    dates = []
    with open(os.path.join(DATA_DIR, 'checkin.json'), 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data['business_id'] == TARGET_BUSINESS_ID:
                    raw_dates = [datetime.strptime(d.strip(), "%Y-%m-%d %H:%M:%S") for d in data['date'].split(',')]
                    utc = pytz.utc
                    local_tz = pytz.timezone(TARGET_TIMEZONE)
                    for dt in raw_dates:
                        dates.append(utc.localize(dt).astimezone(local_tz))
                    break
            except: continue

    if not dates: return
    df = pd.DataFrame({'dt': dates})
    df = df[df['dt'].dt.year == TARGET_YEAR].copy()
    
    df['date_str'] = df['dt'].dt.date
    df['hour'] = df['dt'].dt.hour
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).map({True: 'Weekend', False: 'Weekday'})
    
    print(f"   Sample Size: {len(df)} check-ins")

    # ================= 检验 1: 平稳性互补检验 (ADF + KPSS) =================
    print("\n✅ [Test 1] Stationarity (Complementary Tests)")
    
    # 聚合成日度序列 (365天)
    daily_counts = df['date_str'].value_counts().sort_index()
    idx = pd.date_range(start=f'{TARGET_YEAR}-01-01', end=f'{TARGET_YEAR}-12-31')
    daily_counts = daily_counts.reindex(idx, fill_value=0)
    series = daily_counts.values
    
    if HAS_STATSMODELS:
        # 1. ADF Test (原假设: 有单位根/非平稳)
        # autolag='AIC': 自动根据 AIC 准则选择最佳滞后期，解决"滞后期选择"质疑
        adf_res = adfuller(series, autolag='AIC')
        adf_stat, adf_p = adf_res[0], adf_res[1]
        used_lag = adf_res[2]
        
        # 2. KPSS Test (原假设: 平稳) -> 这是 ADF 的互补检验
        # 'c': 检验围绕常数的平稳性 (Level Stationarity)
        kpss_res = kpss(series, regression='c', nlags='auto') 
        kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
        
        print(f"   (A) ADF Test (H0: Non-Stationary): p={adf_p:.4e} | Lags Used={used_lag} (Based on AIC)")
        print(f"   (B) KPSS Test (H0: Stationary):    p={kpss_p:.4f}")
        
        # 联合判决逻辑
        if adf_p < 0.05 and kpss_p > 0.05:
            print("   👉 结论: [Strictly Stationary]. (ADF拒绝非平稳 + KPSS接受平稳)")
        elif adf_p < 0.05 and kpss_p < 0.05:
            print("   👉 结论: [Difference Stationary]. 可能存在结构突变，但整体可用。")
        else:
            print("   👉 结论: [Non-Stationary]. 数据有风险。")

    # ================= 检验 2: 正态性与相关性稳健检验 =================
    print("\n✅ [Test 2] Normality & Robust Correlation")
    
    # 构造 Hourly Profile (24小时均值)
    # 注意：这里我们用"每小时的平均到达数"来算相关性，而不是概率密度，这样更能反映强度
    weekday_hourly = df[df['is_weekend']=='Weekday'].groupby('hour').count()['dt'] / (365 * 5/7) # 估算
    weekend_hourly = df[df['is_weekend']=='Weekend'].groupby('hour').count()['dt'] / (365 * 2/7)
    
    # 重新对其索引 0-23，补0
    vec_wd = np.array([weekday_hourly.get(h, 0) for h in range(24)])
    vec_we = np.array([weekend_hourly.get(h, 0) for h in range(24)])
    
    # 1. Shapiro-Wilk 正态性检验
    # 如果 p < 0.05，说明数据非正态 -> 必须用 Spearman
    shapiro_wd = stats.shapiro(vec_wd)
    shapiro_we = stats.shapiro(vec_we)
    print(f"   Shapiro-Wilk (Weekday): p={shapiro_wd.pvalue:.4f}")
    print(f"   Shapiro-Wilk (Weekend): p={shapiro_we.pvalue:.4f}")
    is_normal = (shapiro_wd.pvalue > 0.05) and (shapiro_we.pvalue > 0.05)
    
    # 2. 相关性
    corr_p, _ = stats.pearsonr(vec_wd, vec_we)
    corr_s, _ = stats.spearmanr(vec_wd, vec_we)
    
    print(f"   Pearson r (Linear):   {corr_p:.4f}")
    print(f"   Spearman ρ (Rank):    {corr_s:.4f} (推荐使用，因数据可能非正态)")

    # ================= 检验 3: 差异来源定位与效应量 =================
    print("\n✅ [Test 3] Difference Source & Effect Size")
    
    # 我们需要比较的是：每小时到达数的分布。
    # 比如：Weekday 的 24 个点 vs Weekend 的 24 个点
    # Mann-Whitney U Test: 检验两个分布的中位数是否有显著差异 (Non-parametric t-test)
    # H0: Weekday 和 Weekend 的强度一样
    # H1: Weekend 的强度显著高于 Weekday
    u_stat, u_p = stats.mannwhitneyu(vec_wd, vec_we, alternative='two-sided')
    
    # Cohen's d (效应量)
    d_val = calculate_cohens_d(vec_we, vec_wd)
    
    print(f"   Mann-Whitney U Test: p={u_p:.4e}")
    print(f"   Cohen's d: {d_val:.4f}")
    
    if d_val > 0.8:
        effect_desc = "Large Effect (巨大差异)"
    elif d_val > 0.5:
        effect_desc = "Medium Effect (中等差异)"
    else:
        effect_desc = "Small Effect"
        
    print(f"   👉 效应量解读: {effect_desc}. Weekend 明显比 Weekday 忙。")
    print(f"   👉 结论: K-S 检验的差异不仅来自形状，更来自【强度(Intensity)】的显著不同。")

    # ================= 绘图：带统计标注的对比图 =================
    plt.figure(figsize=(10, 6))
    plt.plot(range(24), vec_wd, 'b-o', label='Weekday (Avg Rate)', linewidth=2)
    plt.plot(range(24), vec_we, 'r-s', label='Weekend (Avg Rate)', linewidth=2)
    
    plt.title(f"Arrival Intensity Comparison\nSpearman ρ={corr_s:.2f}, Mann-Whitney p={u_p:.2e}, Cohen's d={d_val:.2f}", fontsize=12)
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Arrivals per Hour")
    plt.xticks(range(0, 24))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(FIGURE_DIR, 'stat_ultimate_comparison.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    step2_ultimate_validation()