import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

try:
    # 使用相对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'bf.csv')
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)  # 加载数据
    
    # 计算交易量增量
    df = df.sort_values(by=['比赛ID', '赛前小时数'], ascending=[False, True])
    df['上一笔交易量'] = df.groupby('比赛ID')['当前交易量'].shift(1)
    df['交易量增量'] = (df['当前交易量'] - df['上一笔交易量']).fillna(0)

    # 数据清洗
    df = df[df['赛前小时数'] >= 0]
    df = df[df['交易量增量'] >= 0]  # 过滤负值

    # 创建分析特征
    df['时间窗口'] = pd.cut(df['赛前小时数'],
                         bins=[0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,48, 72],
                         labels=['0-1h','1-2h', '2-3h', '3-4h', '4-5h', '5-6h','6-7h','7-8','8-9','9-10','10-11','11-12','12-13', '13-14','14-15','15-16','16-17','17-18','18-19','19-20','20-21' ,'21-22','22-23','23-24', '24-48h', '48-72h'])

    # 分组聚合
    window_data = df.groupby('时间窗口', observed=False)['交易量增量'].sum().sort_values(ascending=False)

    # 可视化
    plt.figure(figsize=(16, 8))  # 调整图表尺寸
    sns.set_style('whitegrid')

    # 饼图
    plt.subplot(1,2,2)
    explode = [0.1 if i==window_data.idxmax() else 0 for i in window_data.index]
    window_data.plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f'],
        explode=explode,
        shadow=True
    )
    plt.title('各时间段交易量增量占比', fontsize=12)
    plt.ylabel('')

    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(current_dir, '..', 'output', 'trading_volume_analysis.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    # 显示图表
    plt.show()

except FileNotFoundError as e:
    print(f"错误: {str(e)}")
except pd.errors.EmptyDataError:
    print("错误: 数据文件为空")
except Exception as e:
    print(f"发生未知错误: {str(e)}")