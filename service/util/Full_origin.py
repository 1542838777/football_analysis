import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv('D:/lqs/life/football/file/_1亚盘12h11.csv')

X = df.drop(['europe_handicap_result', 'match_time', 'match_id'], axis=1)
df['europe_handicap_result'] = df['europe_handicap_result'].replace(3, 2)
y = df['europe_handicap_result']

# 计算类别权重
class_weights = len(df) / (3 * np.bincount(df['europe_handicap_result']))
weights_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

# 时间序列交叉验证（防止数据泄露）
tscv = TimeSeriesSplit(n_splits=5)

# 特征筛选（基于模型重要性）
xgb = XGBClassifier(objective='multi:softprob',
                    eval_metric='mlogloss',
                    scale_pos_weight=weights_dict,
                    tree_method='hist')
# 使用平衡后的class_weight参数
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
xgb.set_params(scale_pos_weight=class_weights)
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.7, 0.9],
    'gamma': [0, 0.1]
}

# 网格搜索调优
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           cv=tscv,
                           scoring='f1_macro',
                           n_jobs=-1)

X = df.drop(['europe_handicap_result', 'match_time', 'league_id', 'match_id'], axis=1)
df['europe_handicap_result'] = df['europe_handicap_result'].replace(3, 2)
y = df['europe_handicap_result']

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

grid_search.fit(X_scaled, y)
best_model = grid_search.best_estimator_

# 特征重要性可视化
feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.show()

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 分类报告
y_pred = best_model.predict(X_test)

# 打印准确率
print('Accuracy:', best_model.score(X_test, y_test))
print(classification_report(y_test, y_pred, target_names=['负', '平', '胜']))


# 定义预测方法
def predict_results(model, scaler, new_data):
    """
    输入一组不带结果的数据，返回推荐的结果
    :param model: 训练好的模型
    :param scaler: 标准化处理器
    :param new_data: 不带结果的新数据
    :return: 预测结果
    """
    # 确保新数据的列顺序和训练数据一致
    new_data = new_data[X.columns]

    # 标准化处理
    new_data_scaled = scaler.transform(new_data)

    # 进行预测
    predictions = model.predict(new_data_scaled)

    # 将预测结果转换为文字描述
    result_mapping = {0: '负', 1: '平', 2: '胜'}
    predicted_results = [result_mapping[pred] for pred in predictions]

    return predicted_results


# 定义最近N场预测率的方法
def recent_n_games_prediction_rate(model, X, y, N):
    """
    计算最近N场比赛的预测准确率
    :param model: 训练好的模型
    :param X: 特征矩阵
    :param y: 目标变量
    :param N: 最近N场比赛
    :return: 最近N场比赛的预测准确率
    """
    # 选取最后N场比赛的数据
    X_recent = X[-N:]
    y_recent = y[-N:]

    # 标准化处理
    X_recent_scaled = scaler.transform(X_recent)

    # 进行预测
    y_pred_recent = model.predict(X_recent_scaled)

    # 计算准确率
    accuracy = np.mean(y_pred_recent == y_recent)

    return accuracy


# 设置最近N场比赛的数量
N = 50
recent_accuracy = recent_n_games_prediction_rate(best_model, X, y, N)
print(f'最近 {N} 场比赛的预测准确率: {recent_accuracy:.2%}')


# 示例调用

# 定义最近N场胜平负各自的准确率方法

# 定义最近N场胜平负各自的准确率方法
# 定义最近N场胜平负各自的准确率方法
def recent_n_games_win_draw_loss_accuracy(model, scaler, X, y, N):
    """
    计算最近N场比赛中胜、平、负各自的预测准确率

    参数：
    - model: 训练好的模型
    - scaler: 数据标准化器
    - X: 特征矩阵（按时间顺序排列）
    - y: 目标变量（按时间顺序排列，类别为 0:负, 1:平, 2:胜）
    - N: 最近N场比赛

    返回：
    - loss_accuracy: 负类预测准确率
    - draw_accuracy: 平类预测准确率
    - win_accuracy: 胜类预测准确率
    """
    # 选取最后N场比赛的数据
    X_recent = X[-N:]
    y_recent = y[-N:]

    # 标准化处理
    X_recent_scaled = scaler.transform(X_recent)

    # 进行预测
    y_pred_recent = model.predict(X_recent_scaled)

    # 分别计算胜、平、负的准确率
    loss_indices = y_recent == 0  # 负类索引
    draw_indices = y_recent == 1  # 平类索引
    win_indices = y_recent == 2  # 胜类索引

    # 计算各类别准确率（避免除零错误）
    loss_accuracy = (
        np.mean(y_pred_recent[loss_indices] == y_recent[loss_indices])
        if np.sum(loss_indices) > 0 else 0
    )
    draw_accuracy = (
        np.mean(y_pred_recent[draw_indices] == y_recent[draw_indices])
        if np.sum(draw_indices) > 0 else 0
    )
    win_accuracy = (
        np.mean(y_pred_recent[win_indices] == y_recent[win_indices])
        if np.sum(win_indices) > 0 else 0
    )

    return loss_accuracy, draw_accuracy, win_accuracy


# 设置最近N场比赛的数量
N = 10

# 计算最近N场比赛中胜、平、负各自的预测准确率
loss_acc, draw_acc, win_acc = recent_n_games_win_draw_loss_accuracy(
    best_model, scaler, X, y, N
)

# 打印结果
print(f"最近 {N} 场比赛的预测准确率：")
print(f"负类准确率: {loss_acc:.2%}")
print(f"平类准确率: {draw_acc:.2%}")
print(f"胜类准确率: {win_acc:.2%}")
