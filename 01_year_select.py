import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ================= 配置区域 =================
# 目标：Café Du Monde (New Orleans)
TARGET_BUSINESS_ID = "FEXhWNCMkv22qG04E83Qjg"
DATA_DIR = './data/'
FIGURE_DIR = './figure/'  # 修正命名规范：全大写常量
# ===========================================

def step1_health_check():
    print(f"--- [Module 1] Data Health Check & Selection ---")
    print(f"Target Business ID: {TARGET_BUSINESS_ID}")
    
    # 1. 确保输出目录存在（关键：避免保存图片时目录不存在报错）
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # 2. 商家属性诊断 (Business Metadata Analysis)
    meta = None
    with open(os.path.join(DATA_DIR, 'business.json'), 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data['business_id'] == TARGET_BUSINESS_ID:
                    meta = data
                    break
            except json.JSONDecodeError as e:  # 精准捕获JSON解析错误
                print(f"⚠️  解析business.json行失败: {e}")
                continue
            except Exception as e:  # 其他异常（兜底，不推荐但比裸except好）
                print(f"⚠️  读取business.json时发生未知错误: {e}")
                continue
    
    if not meta:
        print("❌ Error: Business ID not found in business.json!")
        return

    print("\n✅ [1. 商家属性验证 - 用于支撑论文假设]")
    print(f"   Name: {meta['name']}")
    print(f"   City: {meta['city']}")
    print(f"   Review Count: {meta['review_count']}")
    
    # 提取关键属性（增加None判断，避免AttributeError）
    attrs = meta.get('attributes', {}) or {}  # 确保attrs是字典
    is_takeout = attrs.get('RestaurantsTakeOut', 'N/A')
    
    # 修正：精准判断Ambience中的touristy属性
    ambience = attrs.get('Ambience', {})
    if isinstance(ambience, str):  # 部分数据可能是字符串格式的字典
        try:
            ambience = json.loads(ambience.replace("'", '"'))  # 转换为字典
        except (json.JSONDecodeError, TypeError):
            ambience = {}
    is_touristy = ambience.get('touristy', False)  # 精准判断
    
    price_range = attrs.get('RestaurantsPriceRange2', 'N/A')
    
    print(f"   - Take-out Support: {is_takeout} (关键: 支持 M/M/1 单队列假设)")
    print(f"   - Touristy Ambience: {is_touristy} (关键: 支撑'拥堵代表质量'的信号理论)")
    print(f"   - Price Range: {price_range} (1=Cheap, 4=Expensive)")
    
    # 3. 年度数据量分析 (Yearly Volume Analysis)
    print("\n✅ [2. 年份选择分析 - 用于确定 Data Collection]")
    dates = []
    with open(os.path.join(DATA_DIR, 'checkin.json'), 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data['business_id'] == TARGET_BUSINESS_ID:
                    # 解析checkin的date字段（多个时间用逗号分隔）
                    date_strs = [d.strip() for d in data['date'].split(',') if d.strip()]
                    for d_str in date_strs:
                        try:
                            dt = datetime.strptime(d_str, "%Y-%m-%d %H:%M:%S")
                            dates.append(dt)
                        except ValueError:
                            print(f"⚠️  时间格式解析失败: {d_str} (跳过该条)")
                    break
            except json.JSONDecodeError as e:
                print(f"⚠️  解析checkin.json行失败: {e}")
                continue
            except Exception as e:
                print(f"⚠️  读取checkin.json时发生未知错误: {e}")
                continue
    
    # 处理checkin数据为空的情况
    if not dates:
        print("❌ Error: 未找到该商家的Check-in数据！")
        return
    
    df = pd.DataFrame({'dt': dates})
    df['year'] = df['dt'].dt.year
    
    yearly_counts = df['year'].value_counts().sort_index()
    print(yearly_counts.to_string() if not yearly_counts.empty else "   无年度数据")
    
    # 自动推荐（增加空值判断）
    if yearly_counts.empty:
        print("\n⚠️  无有效年度数据，无法推荐年份！")
    else:
        best_year = yearly_counts.idxmax()
        max_count = yearly_counts.max()
        print(f"\n👉 推荐年份: {best_year} (数据量: {max_count})")
        print(f"   理由: 数据密度最高，能够最大程度减少稀疏性带来的拟合误差。")
    
    # 绘图并保存（修正路径拼接 + 确保目录存在）
    plt.figure(figsize=(10, 5))
    yearly_counts.plot(kind='bar', color='#4c72b0', edgecolor='black')
    plt.title(f"Yearly Check-in Volume: {meta['name']}")
    plt.xlabel("Year")
    plt.ylabel("Number of Check-ins")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 修正：拼接完整的保存路径
    fig_path = os.path.join(FIGURE_DIR, 'figure_yearly_trend.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # 增加dpi和bbox_inches优化保存效果
    print(f"   [图表已保存]: {fig_path}")
    plt.show()

if __name__ == "__main__":
    step1_health_check()