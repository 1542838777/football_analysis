# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 阶段1：数据预处理
df = pd.read_csv('D:/lqs/life/football/file/football_guessing_match_factor.csv')  # 加载数据
print(df.spf_result.value_counts())  # 检查类别分布
df= df.dropna()
# 处理分类目标变量（将3,1,0映射为分类标签）
y = df['spf_result'].astype('category')
X = df.drop('spf_result', axis=1)

# 标准化处理（逻辑回归对尺度敏感）
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# 阶段2：单因子阈值分析（示例分析2个关键因子）(单个因子不同值的情况)
def plot_threshold(feature_name):
    """可视化因子分布阈值"""
    plt.figure(figsize=(12, 6))
    # 按比赛结果分组绘制分布曲线
    for result in [0, 1, 3]:
        subset = df[df.spf_result == result]
        sns.kdeplot(subset[feature_name], label=f'Result {result}')
    plt.title(f'Threshold  Analysis: {feature_name}')
    plt.legend()
    plt.show()


# 示例分析两个重要因子
plot_threshold('win_odds_of24h_final_change')  # 最终赔率变化
plot_threshold('win_odds_of24h_up_num')  # 赔率上涨次数

# 阶段3：多变量逻辑回归（带正则化防止过拟合）
# 拆分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 构建带L2正则化的多分类模型
model = LogisticRegression(multi_class='multinomial', solver='saga', penalty='l1', C=0.5)
model.fit(X_train, y_train)

# 展示特征重要性
importance = pd.DataFrame({
    'feature': X.columns,
    'coef': model.coef_[0]  # 取第一个类的系数示例
}).sort_values('coef', ascending=False)

print("\n特征重要性排序：")
print(importance.head(10))  # 显示前10重要特征

# 阶段4：交互作用分析
# 创建交互特征示例
df['interaction1'] = (df['win_odds_of24h_up_num'] > 2) & (df['win_odds_of24h_final_change'] < 0)
df['interaction2'] = (df['win_odds_of24h_drop_num'] > 1) & (df['win_odds_of24h_delta'] > 0.5)

# 评估交互特征
X_interact = pd.get_dummies(df[[df.columns[1], df.columns[2]]], drop_first=True)
X_combined = pd.concat([X_scaled, X_interact], axis=1)

#xx
print(X_combined.shape)  # 输出特征矩阵的形状
print(y_train.shape)      # 输出目标变量的形状
# 重新训练模型
model_enhanced = LogisticRegression(max_iter=1000)
model_enhanced.fit(X_combined, y_train)

print("\n包含交互特征的分类报告：")
print(classification_report(y_test, model_enhanced.predict(X_combined)))

# 阶段5：过拟合检查
# 绘制学习曲线（需要sklearn.model_selection  import learning_curve）
train_sizes, train_scores, test_scores = learning_curve(
    model_enhanced, X_combined, y, cv=5)

plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Score')
plt.title('Learning  Curve')
plt.legend()
plt.show()
