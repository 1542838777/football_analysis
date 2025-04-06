# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:00:00 2024
比赛结果因子分析方案
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据准备 -------------------------------------------------
# 读取数据（请替换为实际路径）
df = pd.read_csv('D:/lqs/life/football/file/football_guessing_match_factor.csv')  # 加载数据
# 删除first_lose_odds_of24h <1.26的
df = df[df['first_lose_odds_of24h'] >= 1.29]
df =  df[df['last_lose_odds_of24h'] >= 1.29]
df =  df[df['first_win_odds_of24h'] >= 1.29]
df =   df[df['last_win_odds_of24h'] >= 1.29]
# 显示数据信息
print(df.info())  # 查看字段类型和缺失值
print(df.describe())  # 查看数值型变量的统计分布

# 2. 基础单变量分析 -------------------------------------------
# 筛选数值型变量（根据实际数据调整）
numeric_cols = ['first_win_odds_of24h',
                'first_draw_odds_of24h',
                'first_lose_odds_of24h',
                'last_win_odds_of24h',
                'last_draw_odds_of24h',
                'last_lose_odds_of24h',
                'win_odds_of24h_drop_num',
                'draw_odds_of24h_drop_num',
                'lose_odds_of24h_drop_num',
                'win_odds_of24h_up_num',
                'draw_odds_of24h_up_num',
                'lose_odds_of24h_up_num',
                'win_odds_of24h_change_num',
                'draw_odds_of24h_change_num',
                'lose_odds_of24h_change_num',
                'win_odds_of24h_stable_num',
                'draw_odds_of24h_stable_num',
                'lose_odds_of24h_stable_num',
                'total_num_of24h',
                'win_odds_of24h_change_num_rank',
                'draw_odds_of24h_change_num_rank',
                'lose_odds_of24h_change_num_rank',
                'win_odds_of24h_stable_num_rank',
                'draw_odds_of24h_stable_num_rank',
                'lose_odds_of24h_stable_num_rank',
                'win_odds_of24h_final_change',
                'draw_odds_of24h_final_change',
                'lose_odds_of24h_final_change',
                'win_odds_of24h_delta',
                'draw_odds_of24h_delta',
                'lose_odds_of24h_delta'

                ]  # 补充其他实际字段名


# 绘制箱线图观察分布
plt.figure(figsize=(12, 6))
sns.boxplot(x='spf_result', y='win_odds_of24h_final_change', data=df)
plt.title('Final Change Distribution by Match Result')
plt.show()

# 3. 数据预处理 -----------------------------------------------
# 处理缺失值（根据实际情况选择）
df.dropna(inplace=True)  # 简单删除缺失值，也可用填充法

# 标准化处理（逻辑回归建议标准化）
scaler = StandardScaler()
X = scaler.fit_transform(df[numeric_cols])
y = df['spf_result']

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. 多变量逻辑回归 ------------------------------------------
# 使用带正则化的逻辑回归防止过拟合（L1正则化有特征选择功能）
model = LogisticRegression(
    multi_class='multinomial',  # 多分类
    solver='saga',  # 支持多分类和L1正则
    penalty='l1',  # L1正则化防止过拟合
    max_iter=1000,
    C=0.5  # 正则化强度，越小正则化越强
)
model.fit(X_train, y_train)

# 5. 结果解读 -------------------------------------------------
# 查看特征重要性
feature_importance = pd.DataFrame({
    'feature': numeric_cols,
    'coefficient': model.coef_[0]  # 取第一个类的系数
})
feature_importance = feature_importance.sort_values('coefficient', ascending=False)
print("特征重要性排序：")
print(feature_importance)

# 评估模型
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# 6. 交互作用分析 ---------------------------------------------
# 示例：创建交互特征（根据实际分析调整）
df['interaction_1'] = (df['last_draw_odds_of24h'] > 1.92) & (df['last_draw_odds_of24h'] <= 3.4) & (
        df['lose_odds_of24h_drop_num'] <= 1) & (df['win_odds_of24h_stable_num'] <= 1)
df['interaction_1'] = df['interaction_1'].astype(int)  # 将布尔值转换为0/1

# 查看交互特征的分布情况
print("\n交互特征分布：")
print(df['interaction_1'].value_counts())

# 查看交互特征与比赛结果的关系
print("\n交互特征与结果交叉表：")
print(pd.crosstab(df['interaction_1'], df['spf_result'], margins=True))

# 将新特征加入分析列表（重要！）
new_numeric_cols = numeric_cols + ['interaction_1']  # 将原始特征与交互特征合并

# 重新预处理数据（包含新特征）
scaler = StandardScaler()
X_new = scaler.fit_transform(df[new_numeric_cols])  # 注意这里使用新特征列表
y = df['spf_result']
# 划分数据集
X_train_new, X_test_new, y_train, y_test = train_test_split(
    X_new, y, test_size=0.2, random_state=42)

# 重新训练模型（包含交互特征）
model_new = LogisticRegression(
    multi_class='multinomial',
    solver='saga',
    penalty='l1',
    max_iter=1000,
    C=0.5
)
model_new.fit(X_train_new, y_train)
# 比较模型性能
print("\n原始模型性能：")
print(classification_report(y_test, model.predict(X_test)))

print("\n包含交互特征后的模型性能：")
y_pred_new = model_new.predict(X_test_new)
print(classification_report(y_test, y_pred_new))

# 查看新特征的重要性
feature_importance_new = pd.DataFrame({
    'feature': new_numeric_cols,
    'coefficient': model_new.coef_[0]
}).sort_values('coefficient', ascending=False)
print("\n包含交互特征后的重要性排序：")
print(feature_importance_new)

# 可视化交互特征效果
plt.figure(figsize=(10,4))
sns.barplot(x='interaction_1', y='spf_result', data=df, estimator=np.mean)
plt.title('Interaction Feature Effect (0=No, 1=Yes)')
plt.ylabel('Win Rate')
plt.show()
# 重新训练包含交互特征的模型
# （重复步骤3-5，观察新特征是否提升模型性能）

# 7. 阈值分析 ------------------------------------------------
# 单变量阈值分析示例
for col in numeric_cols:
    # 按四分位数分析
    thresholds = np.percentile(df[col], [25, 50, 75])
    print(f"\n字段 {col} 阈值分析：")
    print(f"25%分位：{thresholds[0]:.2f}")
    print(f"中位数：{thresholds[1]:.2f}")
    print(f"75%分位：{thresholds[2]:.2f}")

    # 观察不同区间的胜率差异
    df['group'] = pd.qcut(df[col], q=4, duplicates='drop')
    print(df.groupby('group')['spf_result'].mean())
