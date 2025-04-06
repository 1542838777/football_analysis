import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 加载数据并按时间排序
df = pd.read_csv('D:/lqs/life/football/file/_1亚盘12h11.csv').sort_values('match_time')
df['europe_handicap_result'] = df['europe_handicap_result'].replace(3, 2)


# ----------------------
# 特征工程模块
# ----------------------
def create_features(df):
    """创建增强型特征"""
    df = df.copy()

    # 基础特征
    base_cols = [c for c in df.columns if c not in
                 ['europe_handicap_result', 'match_time', 'match_id', 'league_id']]

    # 赔率差异特征

    df['first_shang_xia_odds_diff'] = np.abs(df['first_shang_pan_odds'] - df['first_xia_pan_odds'])
    df['second_shang_xia_odds_diff'] = np.abs(df['second_shang_pan_odds'] - df['second_xia_pan_odds'])
    df['first_both_odds_balance_val'] = 1 / (1 + np.exp(-(df['first_shang_pan_odds'] + df['first_xia_pan_odds']) / 2))
    df['second_both_odds_balance_val'] = 1 / (
            1 + np.exp(-(df['second_shang_pan_odds'] + df['second_xia_pan_odds']) / 2))

    # 历史平局率（需要league_id存在）
    if 'league_id' in df:
        league_draw_rates = df.groupby('league_id')['europe_handicap_result'].apply(
            lambda x: (x == 1).mean()).to_dict()
        df['league_draw_ratio'] = df['league_id'].map(league_draw_rates)

    # 时间相关特征
    if 'match_time' in df:
        df['match_time'] = pd.to_datetime(df['match_time'])
        df['month'] = df['match_time'].dt.month
        df['weekday'] = df['match_time'].dt.weekday

    return df[base_cols + ['first_shang_xia_odds_diff', 'second_shang_xia_odds_diff', 'first_both_odds_balance_val',
                           'second_both_odds_balance_val', 'league_draw_ratio', 'month', 'weekday']]


# ----------------------
# 数据预处理
# ----------------------
# 时序分割（80%训练，20%测试）
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# 特征创建
X_train = create_features(train_df)
X_test = create_features(test_df)
y_train = train_df['europe_handicap_result']
y_test = test_df['europe_handicap_result']

# 标准化处理（仅在训练集上fit）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------
# 类别权重计算
# ----------------------
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
weights_dict = {i: w for i, w in zip(classes, class_weights)}
# ----------------------
# 模型构建与调优
# ----------------------
# 扩展参数网格
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.02, 0.04 ],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0.1],
    'reg_alpha': [0],
    'reg_lambda': [0.1],  # [0,0.1]
    'n_estimators': [100]
}

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 使用class_weight参数的正确方式
xgb = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    tree_method='hist',
    use_label_encoder=False,
    class_weight=weights_dict  # 修正多分类权重传递方式
)

# 网格搜索
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
print("\n最佳参数组合：")
print(grid_search.best_params_)

print("\n各参数重要性排序：")
results = pd.DataFrame(grid_search.cv_results_)
param_importance = results.loc[:, [c for c in results.columns if c.startswith('param_')]]
param_importance['mean_test_score'] = results['mean_test_score']
print(param_importance.corr()['mean_test_score'].sort_values(ascending=False))

# ----------------------
# 模型评估
# ----------------------
# 特征重要性可视化
plt.figure(figsize=(10, 6))
pd.Series(best_model.feature_importances_, index=X_train.columns
          ).nlargest(15).plot(kind='barh')
plt.title('Top 15 Feature Importances')
plt.show()

# 测试集评估
y_pred = best_model.predict(X_test_scaled)
print("\n优化后测试集表现:")
print(f"平衡准确率: {balanced_accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=['负', '平', '胜']))


# ----------------------
# 近期表现分析（正确方法）
# ----------------------
def evaluate_recent_performance(model, X_full, y_full, last_n=200):
    """正确评估最近N场表现"""
    # 获取最后N场（按时间排序）
    recent_X = X_full[-last_n:]
    recent_y = y_full[-last_n:]

    # 标准化处理
    recent_X_scaled = scaler.transform(recent_X)

    # 预测
    y_pred = model.predict(recent_X_scaled)

    # 分类报告
    print(f"\n最近 {last_n} 场真实分布:\n{recent_y.value_counts()}")
    print(f"\n最近 {last_n} 场预测报告:")
    print(classification_report(recent_y, y_pred, target_names=['负', '平', '胜']))


evaluate_recent_performance(best_model, create_features(df), df['europe_handicap_result'])


# ----------------------
# 预测函数
# ----------------------
def safe_predict(model, scaler, new_data, feature_columns):
    """
    安全预测函数
    :param feature_columns: 训练时的特征列顺序
    """
    try:
        # 确保列顺序正确
        new_data = new_data.reindex(columns=feature_columns, fill_value=0)
        scaled_data = scaler.transform(new_data)
        return model.predict(scaled_data)
    except Exception as e:
        print(f"预测错误: {str(e)}")
        return None


# ----------------------
# 新增方法：获取最近N场准确率
# ----------------------
def get_recent_n_accuracy(model, X_full, y_full, n_games, scaler):
    """
    计算并返回最近N场比赛的预测准确率
    参数：
    - model: 训练好的模型
    - X_full: 完整特征数据集（需按时间排序）
    - y_full: 完整目标变量（需按时间排序）
    - n_games: 需要分析的最近比赛场次数
    - scaler: 标准化处理器

    返回：
    - accuracy: 最近n_games场的预测准确率（浮点数）
    """
    try:
        # 获取最近N场比赛数据
        recent_X = X_full[-n_games:]
        recent_y = y_full[-n_games:]

        # 标准化处理
        recent_X_scaled = scaler.transform(recent_X)

        # 预测
        y_pred = model.predict(recent_X_scaled)

        # 计算准确率
        return np.mean(y_pred == recent_y)
    except Exception as e:
        logging.error(f"计算最近{n_games}场准确率时出错: {str(e)}")
        return None


# ----------------------
# 在模型评估部分添加调用示例
# ----------------------
# 在现有 evaluate_recent_performance 函数之后添加：

# 计算并打印最近50场和10场的准确率
for n in [100, 35]:
    acc = get_recent_n_accuracy(best_model,
                                create_features(df),
                                df['europe_handicap_result'],
                                n_games=n,
                                scaler=scaler)
    print(f"\n最近 {n} 场整体准确率: {acc:.2%}")

# 同时可以获取准确率数值用于后续分析
recent_50_acc = get_recent_n_accuracy(best_model, create_features(df),
                                      df['europe_handicap_result'], 50, scaler)
recent_10_acc = get_recent_n_accuracy(best_model, create_features(df),
                                      df['europe_handicap_result'], 10, scaler)