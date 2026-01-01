import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
from datetime import datetime
from scipy import stats

# ================= 🔬 实验参数配置 (Configuration) =================
TARGET_BUSINESS_ID = "FEXhWNCMkv22qG04E83Qjg" # Café Du Monde
TARGET_YEAR = 2015
TARGET_TIMEZONE = 'America/Chicago'

# [关键策略]: 只拟合高峰期数据，保证 lambda 相对恒定 (Stationary)
PEAK_START_HOUR = 11
PEAK_END_HOUR = 13 

DATA_DIR = './data/'
FIGURE_DIR = './figure/'
# ===================================================================

def calculate_aic(params, dist, data):
    """计算 AIC = 2k - 2*ln(L)"""
    log_likelihood = np.sum(dist.logpdf(data, *params))
    k = len(params)
    aic = 2 * k - 2 * log_likelihood
    print(f"    AIC计算细节: 参数个数k={k}, 对数似然={log_likelihood:.2f}, AIC={aic:.1f}")
    return aic

def load_and_preprocess_data():
    """加载并清洗数据"""
    print("="*50)
    print("1. 加载并预处理数据")
    print("="*50)
    dates = []
    # 确保文件存在
    file_path = os.path.join(DATA_DIR, 'checkin.json')
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return pd.DataFrame()

    line_count = 0
    target_line_found = False
    with open(file_path, 'r') as f:
        for line in f:
            line_count += 1
            try:
                data = json.loads(line)
                if data['business_id'] == TARGET_BUSINESS_ID:
                    target_line_found = True
                    raw_dates_str = data['date'].split(',')
                    print(f"    找到目标商家({TARGET_BUSINESS_ID})，原始签到时间数={len(raw_dates_str)}")
                    raw_dates = [datetime.strptime(d.strip(), "%Y-%m-%d %H:%M:%S") for d in raw_dates_str]
                    # 时区转换 (UTC -> Local)
                    utc = pytz.utc
                    local_tz = pytz.timezone(TARGET_TIMEZONE)
                    for dt in raw_dates:
                        local_dt = utc.localize(dt).astimezone(local_tz)
                        if local_dt.year == TARGET_YEAR:
                            dates.append(local_dt)
                    break
            except Exception as e:
                continue
    
    print(f"    扫描行数={line_count}, 是否找到目标商家={target_line_found}")
    print(f"    {TARGET_YEAR}年有效签到时间数={len(dates)}")
    
    df = pd.DataFrame({'dt': dates})
    if not df.empty:
        df['date_str'] = df['dt'].dt.date
        df['hour'] = df['dt'].dt.hour
        df['is_weekend'] = df['dt'].dt.dayofweek.isin([5, 6])
        # 打印基础统计
        print(f"    数据基础统计:")
        print(f"      - 日期范围: {df['dt'].min()} ~ {df['dt'].max()}")
        print(f"      - 工作日签到数: {len(df[~df['is_weekend']])}")
        print(f"      - 周末签到数: {len(df[df['is_weekend']])}")
        print(f"      - 小时分布(前5小时): {df['hour'].value_counts().head(5)}")
    else:
        print("    ❌ 无有效数据")
    return df

def get_pooled_inter_arrival_times(df_subset, label):
    """获取高峰期合并间隔数据（新增CV计算）"""
    print(f"\n    处理{label}间隔数据:")
    pooled_intervals = []
    if df_subset.empty: 
        print(f"      ❌ {label}数据为空")
        return np.array([])
    
    grouped = df_subset.groupby('date_str')
    print(f"      按日期分组数={len(grouped)}")
    
    daily_intervals_count = []
    for idx, (date, group) in enumerate(grouped):
        # 1. 筛选高峰期 (Peak Window Slicing)
        peak_data = group[(group['hour'] >= PEAK_START_HOUR) & (group['hour'] < PEAK_END_HOUR)]
        peak_count = len(peak_data)
        
        # 必须至少有2个点才能算间隔
        if peak_count < 2:
            if idx < 5: # 只打印前5天的空数据
                print(f"        日期{date}: 高峰期签到数={peak_count}，跳过")
            continue
            
        # 2. 排序
        sorted_times = peak_data['dt'].sort_values()
        
        # 3. 计算间隔 (Diff)，转换为分钟
        intervals = sorted_times.diff().dropna().dt.total_seconds() / 60.0
        
        # 4. 清洗逻辑
        valid = intervals[(intervals > 0) & (intervals < 60)]
        valid_count = len(valid)
        daily_intervals_count.append(valid_count)
        
        if idx < 5: # 只打印前5天的有效数据
            print(f"        日期{date}: 高峰期签到数={peak_count}，有效间隔数={valid_count}，间隔示例={valid.head(3).round(2).tolist()}")
        
        pooled_intervals.extend(valid.tolist())
    
    # 转换为数组并打印统计
    intervals_arr = np.array(pooled_intervals)
    print(f"      {label}间隔数据统计:")
    print(f"        - 总有效间隔数: {len(intervals_arr)}")
    if len(intervals_arr) > 0:
        # 核心修改：计算均值、标准差、变异系数(CV)
        mean_val = np.mean(intervals_arr)
        std_val = np.std(intervals_arr)
        # 处理均值接近0的情况，避免除零错误
        if mean_val < 1e-8:
            cv_val = np.nan
        else:
            cv_val = std_val / mean_val  # CV = 标准差 / 均值
        
        # 打印统计信息（新增CV）
        print(f"        - 描述性统计: 均值={mean_val:.2f}分钟, 标准差={std_val:.2f}分钟, 变异系数(CV)={cv_val:.2f}")
        print(f"        - 中位数={np.median(intervals_arr):.2f}分钟, 最大值={np.max(intervals_arr):.2f}分钟")
        print(f"        - 前10个间隔值: {intervals_arr[:10].round(2)}")
    else:
        print(f"        ❌ 无有效间隔数据")
    return intervals_arr

def fit_and_compare_distributions(intervals, label, color, ax):
    """拟合分布并计算 AIC, K-S（仅保留这两个指标）"""
    print("\n" + "="*50)
    print(f"2. 拟合{label}分布")
    print("="*50)
    if len(intervals) < 10:
        print(f"⚠️ {label}: 样本量不足({len(intervals)})，跳过拟合。")
        return None
    
    fit_results = {}
    
    # 1. 绘图 (直方图)
    sns.histplot(intervals, bins=30, stat='density', alpha=0.3, color=color, label=f'{label} Data', ax=ax)
    x_plot = np.linspace(0, max(intervals), 1000)
    
    # --- 待拟合的分布列表 ---
    candidates = [
        ('Exponential', stats.expon, '--'), 
        ('Gamma', stats.gamma, '-'),
        ('Weibull', stats.weibull_min, ':')
    ]
    
    # 初始化最优AIC
    min_aic = float('inf')
    best_dist_name = ""
    
    # 2. 遍历拟合每个分布
    for name, dist, style in candidates:
        print(f"\n    👉 拟合{name}分布:")
        # A. 拟合参数 (floc=0固定位置参数为0)
        params = dist.fit(intervals, floc=0)
        print(f"      拟合参数: {params} (floc=0固定)")
        
        # B. 计算AIC
        aic = calculate_aic(params, dist, intervals)
        
        # C. 分布参数解析与打印
        if name == 'Exponential':
            scale = params[1]
            lambda_per_min = 1.0 / scale
            lambda_per_hour = lambda_per_min * 60
            param_str = f"λ={lambda_per_min:.3f}/min (λ={lambda_per_hour:.1f}/hr)"
            print(f"      指数分布lambda: {param_str}")
        elif name == 'Gamma':
            shape, scale = params[0], params[2]
            param_str = f"α={shape:.2f}, β={scale:.2f}"
            print(f"      Gamma分布参数: 形状α={shape:.2f}, 尺度β={scale:.2f}")
        elif name == 'Weibull':
            shape, scale = params[0], params[2]
            param_str = f"c={shape:.2f}, β={scale:.2f}"
            print(f"      Weibull分布参数: 形状c={shape:.2f}, 尺度β={scale:.2f}")
        else:
            param_str = ""
        
        # D. K-S Test
        print(f"      计算K-S检验...")
        ks_stat, ks_p = stats.kstest(intervals, dist.name, args=params)
        print(f"      K-S检验结果: 统计量={ks_stat:.4f}, p值={ks_p:.2e}")
        
        # G. 计算理论PDF值
        pdf_vals = dist.pdf(x_plot, *params)
        
        # 保存结果（仅保留AIC和KS）
        fit_results[name] = {
            'params': params,
            'aic': aic,
            'ks_stat': ks_stat, 
            'ks_p': ks_p
        }
        
        # 记录最优AIC
        if aic < min_aic:
            min_aic = aic
            best_dist_name = name
        
        # H. 绘图
        if name == 'Exponential':
            line_color = 'black'
        elif name == 'Gamma':
            line_color = color
        elif name == 'Weibull':
            line_color = 'green'
        ax.plot(x_plot, pdf_vals, linestyle=style, 
                color=line_color, linewidth=2, label=f'{name} ({param_str})')
    
    # 3. 计算ΔAIC并打印
    print(f"\n    📊 {label}拟合结果汇总:")
    for dist_name, metrics in fit_results.items():
        delta_aic = metrics['aic'] - min_aic
        fit_results[dist_name]['delta_aic'] = delta_aic
        print(f"      {dist_name}:")
        print(f"        - AIC={metrics['aic']:.1f}, ΔAIC={delta_aic:.1f}")
        print(f"        - K-S p={metrics['ks_p']:.2e}")
    
    print(f"      ✅ 最优分布(AIC最小): {best_dist_name} (AIC={min_aic:.1f})")
    
    # 4. 生成图上的统计文本 (仅保留AIC和KS)
    stats_text = f"Sample N: {len(intervals)}\nMean: {np.mean(intervals):.2f} min\n"
    # 新增：在图片文本中也加入CV
    if len(intervals) > 0:
        mean_val = np.mean(intervals)
        std_val = np.std(intervals)
        cv_val = std_val / mean_val if mean_val > 1e-8 else np.nan
        stats_text += f"CV: {cv_val:.2f}\n"
    stats_text += "-" * 25 + "\n"
    for dist_name, metrics in fit_results.items():
        delta_aic = metrics['delta_aic']
        stats_text += f"[{dist_name}]\n"
        stats_text += f" AIC: {metrics['aic']:.1f} (Δ={delta_aic:.1f})\n"
        stats_text += f" K-S p: {metrics['ks_p']:.2e}\n"
    
    # 图片文本位置调整
    ax.text(0.55, 0.15, stats_text, transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9), verticalalignment='top')

    # 图片标题/标签
    ax.set_title(f"{label} Arrival Intervals ({PEAK_START_HOUR}:00-{PEAK_END_HOUR}:00)", fontsize=14)
    ax.set_xlabel("Inter-arrival Time (minutes)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 40)
    ax.grid(alpha=0.3)
    
    return fit_results

def step3_ultimate_fitting():
    print("="*60)
    print("开始执行 [Module 3] 分布拟合 (仅KS & AIC + CV统计)")
    print("="*60)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # 1. 加载数据
    df = load_and_preprocess_data()
    if df.empty: 
        print("❌ 无数据，终止程序")
        return

    # 2. 分割工作日/周末并提取间隔
    print("\n" + "="*50)
    print("提取高峰期到达间隔数据 (含CV计算)")
    print("="*50)
    intervals_wd = get_pooled_inter_arrival_times(df[~df['is_weekend']], "工作日")
    intervals_we = get_pooled_inter_arrival_times(df[df['is_weekend']], "周末")
    
    # 3. 绘图拟合
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    print("\n--- 拟合工作日分布 ---")
    res_wd = fit_and_compare_distributions(intervals_wd, "Weekday", "blue", axes[0])
    
    print("\n--- 拟合周末分布 ---")
    res_we = fit_and_compare_distributions(intervals_we, "Weekend", "red", axes[1])
    
    # 保存图片（修改命名为distribution_fitting_clear）
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, 'distribution_fitting_clear.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 图片已保存至: {save_path}")
    
    # 4. 打印最终汇总（仅保留AIC和KS）
    print("\n" + "="*60)
    print("最终拟合结果汇总 (控制台版)")
    print("="*60)
    for label_cn, label_en, res in [("工作日", "Weekday", res_wd), ("周末", "Weekend", res_we)]:
        print(f"\n📈 {label_cn} ({label_en}):")
        if res is None:
            print("  ❌ 无拟合结果")
            continue
        # 补充打印CV到最终汇总
        intervals = intervals_wd if label_cn == "工作日" else intervals_we
        if len(intervals) > 0:
            mean_val = np.mean(intervals)
            std_val = np.std(intervals)
            cv_val = std_val / mean_val if mean_val > 1e-8 else np.nan
            print(f"  📊 基础统计补充: 变异系数(CV)={cv_val:.2f}")
        for dist, metrics in res.items():
            delta_aic = metrics.get('delta_aic', 0)
            print(f"  📊 {dist}:")
            print(f"    - 参数: {metrics['params']}")
            print(f"    - AIC={metrics['aic']:.1f}, ΔAIC={delta_aic:.1f}")
            print(f"    - K-S检验: 统计量={metrics['ks_stat']:.4f}, p值={metrics['ks_p']:.2e}")
        # 标注最优分布
        best_dist = min(res.keys(), key=lambda k: res[k]['aic'])
        print(f"  ✅ 最优分布: {best_dist} (AIC最小={res[best_dist]['aic']:.1f})")

if __name__ == "__main__":
    step3_ultimate_fitting()