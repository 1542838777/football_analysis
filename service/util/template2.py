import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 加载数据并按时间排序
def getOriginData():
    pass  # 你需要根据实际数据源进行加载


def get_match_level_df(guess_type):
    origin_df = getOriginData(guess_type)
    # 这里可以根据需要做一些数据预处理
    return origin_df


# 获取训练所需的 match_level_df 和相关配置
def getSelf():
    # 动态选择目标变量和对应的标签
    target_column = 'todo'  # 根据实际情况选择
    guess_type = 'todo'  # 目标类型可以是 'win_draw_loss'（胜平负）、'asian_handicap'（亚盘）、'goals'（进球数）

    useless_cols = ['todo']  # 可自定义 # match_id

    match_level_df = get_match_level_df('guess_type')
    return target_column, guess_type, useless_cols, match_level_df


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


# 定义多个模型
def get_models():
    models = {
        'XGBoost': XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False),
        'LightGBM': LGBMClassifier(objective='multiclass', metric='multi_logloss'),
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(probability=True)  # 这里需要设置probability=True来支持概率输出
    }
    return models


# 为不同模型设置不同的参数网格
def get_param_grids():
    param_grids = {
        'XGBoost': {
            'max_depth': [3, 5],
            'learning_rate': [0.02, 0.04],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.8, 1.0],
            'n_estimators': [100]
        },
        'LightGBM': {
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.05],
            'n_estimators': [100]
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    }
    return param_grids


# 训练并调优模型
def train_and_evaluate_models(X_train, y_train, X_test, y_test, param_grids, models):
    best_models = {}
    for model_name, model in models.items():
        print(f"\n正在调参 {model_name} ...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=TimeSeriesSplit(n_splits=5),
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)
        best_models[model_name] = {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

        # 模型评估
        y_pred = grid_search.best_estimator_.predict(X_test)
        print(f"\n{model_name} 模型的最佳参数组合：")
        print(grid_search.best_params_)
        print(f"\n{model_name} 模型的测试集表现：")
        print(f"平衡准确率: {balanced_accuracy_score(y_test, y_pred):.2%}")
        target_names = np.unique(y_train)
        print(classification_report(y_test, y_pred, target_names=target_names))
        # 写一个 返回最近N场的预测准确率 的函数
        for n in [20, 150]:
            acc = get_recent_n_accuracy(
                grid_search.best_estimator_,
                X_test,
                y_test,
                n
            )
            print(f"\n{model_name}模型最近{n}场平衡准确率: {acc:.2%}")

    return best_models


# 新增函数：获取最近N场准确率
def get_recent_n_accuracy(model, X_test, y_test, n_games):
    """
    计算模型在最近N场比赛的预测准确率
    :param model: 训练好的模型
    :param X_test: 测试集特征（已标准化）
    :param y_test: 测试集标签
    :param n_games: 需要评估的最近比赛场次
    :return: 平衡准确率
    """
    if n_games > len(X_test):
        n_games = len(X_test)  # 防止超出测试集长度
        print(f"注意：请求的{n_games}场超过测试集最大长度，已自动调整为{len(X_test)}场")

    recent_X = X_test[-n_games:]  # 取最后N场特征
    recent_y = y_test[-n_games:]  # 取最后N场标签

    y_pred = model.predict(recent_X)
    return balanced_accuracy_score(recent_y, y_pred)


# 特征重要性可视化
def plot_feature_importance(models, X_train):
    for model_name, model_info in models.items():
        model = model_info['best_estimator']
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            pd.Series(model.feature_importances_, index=X_train.columns).nlargest(15).plot(kind='barh')
            plt.title(f'{model_name} Top 15 Feature Importances')
            plt.show()


# 主程序
def guess_type(param):
    # 根据目标类型生成相应的标签
    if guess_type == 'win_draw_loss':
        target_names = ['负', '平', '胜']
    elif guess_type == 'asian_handicap':
        target_names = ['上盘', '下盘']
    elif guess_type == 'goals':
        target_names = [str(i) for i in range(8)] + ['7+']  # 进球数0-7+
    else:
        raise ValueError(f"未识别的目标类型: {guess_type}")
    return target_names


if __name__ == '__main__':
    # 获取数据
    target_column, guess_type, useless_cols, match_level_df = getSelf()

    # 数据预处理
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(
        match_level_df, target_column, useless_cols)

    # 类别权重计算
    class_weights = compute_class_weights(y_train)

    # 获取模型和参数网格
    models = get_models()
    param_grids = get_param_grids()

    # 训练并评估模型
    best_models = train_and_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test, param_grids, models)

    # 特征重要性可视化
    plot_feature_importance(best_models, X_train_scaled)
