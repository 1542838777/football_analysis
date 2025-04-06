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
df = pd.read_csv('D:/lqs/life/football/file/_1亚盘12h32.csv')

X = df.drop(['europe_handicap_result', 'match_time','league_id','match_id'], axis=1)
df['europe_handicap_result'] = df['europe_handicap_result'].replace(3, 2)
y = df['europe_handicap_result']

# 修正后的类别权重计算
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
weights_dict = dict(zip(classes, class_weights))

# 划分训练测试集
# 修正数据预处理流程
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 添加sample_weight参数
sample_weights_train = np.array([weights_dict[y_] for y_ in y_train])
# 扩展参数网格
param_grid = {
    'n_estimators': [100, 200],#todo 是否删去
    'max_depth': [ 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.7, 0.9],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0, 0.1]
}
# 时间序列交叉验证（防止数据泄露）
tscv = TimeSeriesSplit(n_splits=5)

# 特征筛选（基于模型重要性）
xgb = XGBClassifier(objective='multi:softprob',
                    eval_metric='mlogloss',
                    scale_pos_weight=weights_dict,
                    tree_method='hist')



# 网格搜索调优

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring='balanced_accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
best_model = grid_search.best_estimator_

# 特征重要性可视化
feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.show()



# 分类报告
y_pred = best_model.predict(X_test)

# 打印准确率
print('Accuracy:', best_model.score(X_test, y_test))
print(classification_report(y_test, y_pred, target_names=[ '负','平', '胜']))


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


# 示例：假设有一组新数据

import pandas as pd
from sklearn.preprocessing import StandardScaler

def predict_results(model, scaler, data, return_probabilities=False):
    """
    预测比赛结果（支持多行数据和概率输出）

    参数：
    - model: 训练好的模型
    - scaler: 数据标准化器
    - data: 输入数据（字符串格式，逗号分隔，多行用 \n 分隔）
    - return_probabilities: 是否返回概率（默认返回类别）

    返回：
    - 预测结果（类别或概率）
    """
    try:
        # 将输入数据按行拆分
        rows = data.strip().split('\n')
        data_list = [list(map(float, row.split(','))) for row in rows]

        # 创建 DataFrame
        new_data = pd.DataFrame(data_list, columns=[
            'handicap', 'first_win_odds_of24h', 'first_draw_odds_of24h', 'first_lose_odds_of24h',
            'last_win_odds_of24h', 'last_draw_odds_of24h', 'last_lose_odds_of24h',
            'win_odds_of24h_drop_num', 'draw_odds_of24h_drop_num', 'lose_odds_of24h_drop_num',
            'win_odds_of24h_up_num', 'draw_odds_of24h_up_num', 'lose_odds_of24h_up_num',
            'win_odds_of24h_change_num', 'draw_odds_of24h_change_num', 'lose_odds_of24h_change_num',
            'win_odds_of24h_stable_num', 'draw_odds_of24h_stable_num', 'lose_odds_of24h_stable_num',
            'total_num_of24h', 'win_odds_of24h_change_num_rank', 'draw_odds_of24h_change_num_rank',
            'lose_odds_of24h_change_num_rank', 'win_odds_of24h_stable_num_rank',
            'draw_odds_of24h_stable_num_rank', 'lose_odds_of24h_stable_num_rank',
            'win_odds_of24h_final_change', 'draw_odds_of24h_final_change', 'lose_odds_of24h_final_change',
            'win_odds_of24h_delta', 'draw_odds_of24h_delta', 'lose_odds_of24h_delta',
            'europe_handicap_result', '亚盘结果', '欧亚盘口差', 'match_time', '欧赔让球数', 'league_id', 'match_id',
            '初组盘口', '二组盘口', '初组上盘赔率', '初组下盘赔率', '初组上盘凯利', '初组下盘凯利',
            '初组盘口中值', '初组凯利中值', '二组上盘赔率', '二组下盘赔率', '二组上盘凯利', '二组下盘凯利',
            '二组盘口中值', '二组凯利中值', '初组上盘赔率变化', '初组下盘赔率变化', '二组上盘赔率变化',
            '二组下盘赔率变化', '盘口变化', '上盘凯利变化', '下盘凯利变化', '盘口中值变化'
        ])

        # 将时间字段转换为 datetime 类型
        new_data['match_time'] = pd.to_datetime(new_data['match_time'])

        # 标准化数据
        X = new_data.drop(['match_time', 'match_id'], axis=1)  # 移除非特征列
        X_scaled = scaler.transform(X)

        # 预测结果
        if return_probabilities:
            # 返回各类别概率
            probabilities = model.predict_proba(X_scaled)
            return probabilities
        else:
            # 返回预测类别
            predictions = model.predict(X_scaled)
            return predictions

    except Exception as e:
        # 异常处理
        print(f"预测失败: {str(e)}")
        return None

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
    win_indices = y_recent == 2   # 胜类索引

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



# 加载新数据
new_data = pd.read_csv('D:/lqs/life/football/file/_1亚盘12h32.csv')

# 确保特征列正确
required_columns = X.columns.tolist()  # 使用训练时的特征列
new_data = new_data[required_columns]

# 执行预测
# results = predict_results(best_model, scaler, new_data)

# 显示预测结果
print("\n预测结果明细：")
print(pd.DataFrame({
    "场次": range(1, len(results)+1),
    "推荐结果": results
}).to_string(index=False))