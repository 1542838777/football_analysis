import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 加载数据并按时间排序
def getOriginData():
    pass  # 你需要根据实际数据源进行加载


def get_match_level_df():
    origin_df = getOriginData()
    # 这里可以根据需要做一些数据预处理
    return origin_df


# 获取训练所需的 match_level_df 和相关配置
def getSelf(target_column, useless_cols=None):
    if useless_cols is None:
        useless_cols = ['europe_handicap_result', 'match_time', 'match_id', 'league_id']  # 可自定义

    match_level_df = get_match_level_df()
    return match_level_df, useless_cols, target_column


# 动态创建增强特征
def create_features(df, useless_cols=None):
    """创建增强型特征"""
    if useless_cols is None:
        useless_cols = ['europe_handicap_result', 'match_time', 'match_id', 'league_id']

    df = df.copy()
    base_cols = [col for col in df.columns if col not in useless_cols]

    # 新增特征（根据实际情况调整）
    new_cols = []  # 根据需求创建新特征
    return df[base_cols + new_cols]


# 数据预处理：时序分割，特征处理，标准化
def preprocess_data(df, target_column, useless_cols=None, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = create_features(train_df, useless_cols)
    X_test = create_features(test_df, useless_cols)

    y_train = train_df[target_column]
    y_test = test_df[target_column]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# 类别权重计算
def compute_class_weights(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weights_dict = {i: w for i, w in zip(classes, class_weights)}
    return weights_dict


# XGBoost模型训练与调参
def train_xgb_model(X_train, y_train, class_weights, param_grid):
    xgb = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        tree_method='hist',
        use_label_encoder=False,
        class_weight=class_weights  # 修正多分类权重传递方式
    )

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=tscv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# 模型评估
def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    print("\n测试集表现:")
    print(f"平衡准确率: {balanced_accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=target_names))


# 特征重要性可视化
def plot_feature_importance(model, X_train):
    plt.figure(figsize=(10, 6))
    pd.Series(model.feature_importances_, index=X_train.columns).nlargest(15).plot(kind='barh')
    plt.title('Top 15 Feature Importances')
    plt.show()


# 近期表现分析
def evaluate_recent_performance(model, X_full, y_full, scaler, target_names, last_n=200):
    """评估最近N场表现"""
    recent_X = X_full[-last_n:]
    recent_y = y_full[-last_n:]

    # 标准化处理
    recent_X_scaled = scaler.transform(recent_X)

    # 预测
    y_pred = model.predict(recent_X_scaled)

    # 分类报告
    print(f"\n最近 {last_n} 场真实分布:\n{recent_y.value_counts()}")
    print(f"\n最近 {last_n} 场预测报告:")
    print(classification_report(recent_y, y_pred, target_names=target_names))


# 安全预测函数
def safe_predict(model, scaler, new_data, feature_columns):
    """确保列顺序正确并进行预测"""
    try:
        new_data = new_data.reindex(columns=feature_columns, fill_value=0)
        scaled_data = scaler.transform(new_data)
        return model.predict(scaled_data)
    except Exception as e:
        print(f"预测错误: {str(e)}")
        return None


# 获取最近N场准确率
def get_recent_n_accuracy(model, X_full, y_full, n_games, scaler):
    """计算最近N场比赛的预测准确率"""
    try:
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


# 主程序
if __name__ == '__main__':
    # 动态选择目标变量和对应的标签
    target_column = 'match_result'  # 根据实际情况选择
    target_type = 'win_draw_loss'  # 目标类型可以是 'win_draw_loss'（胜平负）、'asian_handicap'（亚盘）、'goals'（进球数）

    # 根据目标类型生成相应的标签
    if target_type == 'win_draw_loss':
        target_names = ['负', '平', '胜']
    elif target_type == 'asian_handicap':
        target_names = ['上盘', '下盘']
    elif target_type == 'goals':
        target_names = [str(i) for i in range(8)] + ['7+']  # 进球数0-7+
    else:
        raise ValueError(f"未识别的目标类型: {target_type}")

    useless_cols = ['europe_handicap_result', 'match_time', 'match_id', 'league_id']

    # 获取数据
    match_level_df, useless_cols, y_train_colum = getSelf(target_column, useless_cols)

    # 数据预处理
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(
        match_level_df, y_train_colum, useless_cols)

    # 类别权重计算
    class_weights = compute_class_weights(y_train)

    # 扩展参数网格
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.02, 0.04],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0.1],
        'reg_alpha': [0],
        'reg_lambda': [0.1],
        'n_estimators': [100]
    }

    # 训练XGBoost模型
    best_model, best_params = train_xgb_model(X_train_scaled, y_train, class_weights, param_grid)
    print("\n最佳参数组合：", best_params)

    # 模型评估
    evaluate_model(best_model, X_test_scaled, y_test, target_names)
    plot_feature_importance(best_model, X_train_scaled)

    # 近期表现分析
    evaluate_recent_performance(best_model, X_train_scaled, y_train, scaler, target_names, last_n=200)

    # 获取最近N场的准确率
    for n in [100, 35]:
        acc = get_recent_n_accuracy(best_model, X_train_scaled, y_train, n_games=n, scaler=scaler)
        print(f"\n最近 {n} 场准确率: {acc:.2%}")
