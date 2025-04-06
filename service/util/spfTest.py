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
from itertools import combinations
#呢哇tes

# 加载数据并按时间排序
def getOriginData(guess_type):
    """加载原始数据并按时间排序"""
    if guess_type == 'win_draw_loss':
        from service.spf.initData.data.mysql_data import load_europe_odds_not_handicap_data
        df = load_europe_odds_not_handicap_data()

        return df
    raise ValueError('Invalid guess_type')  # 你需要根据实际数据源进行加载


# 市场分歧
def add_institution_discrepancy_colum(match_level_df):
    """添加机构分歧相关特征"""
    # 市场分歧指数
    match_level_df['market_disagreement'] = (
            match_level_df['first_win_sp_std'] +
            match_level_df['first_draw_sp_std'] +
            match_level_df['first_lose_sp_std']
    ) / 3

    # 分歧方向强度
    match_level_df['ddi_win'] = match_level_df['first_win_sp_std'] / (
            match_level_df['first_draw_sp_std'] + match_level_df['first_lose_sp_std'] + 1e-6)
    match_level_df['ddi_draw'] = match_level_df['first_draw_sp_std'] / (
            match_level_df['first_win_sp_std'] + match_level_df['first_lose_sp_std'] + 1e-6)
    match_level_df['ddi_lose'] = match_level_df['first_lose_sp_std'] / (
            match_level_df['first_win_sp_std'] + match_level_df['first_draw_sp_std'] + 1e-6)

    # 动态分位数排名（按联赛分组）
    for col in ['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']:
        match_level_df[f'{col}_rank'] = match_level_df.groupby('league_id')[col].transform(
            lambda x: x.rank(pct=True, method='first')
        )

    # 构建两两差异矩阵
    match_level_df['win_draw_gap'] = match_level_df['first_win_sp_std'] - match_level_df['first_draw_sp_std']
    match_level_df['win_lose_gap'] = match_level_df['first_win_sp_std'] - match_level_df['first_lose_sp_std']
    match_level_df['draw_lose_gap'] = match_level_df['first_draw_sp_std'] - match_level_df['first_lose_sp_std']

    # 符号编码
    match_level_df['gap_direction'] = (
            (match_level_df['win_draw_gap'] > 0).astype(int) * 100 +
            (match_level_df['win_lose_gap'] > 0).astype(int) * 10 +
            (match_level_df['draw_lose_gap'] > 0).astype(int)
    )

    # 熵值分歧指数
    def calculate_entropy(row):
        total = row.sum()
        probs = row / total
        return -np.sum(probs * np.log(probs + 1e-6))

    match_level_df['disagreement_entropy'] = match_level_df[
        ['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']].apply(calculate_entropy, axis=1)

    # 主导分歧指标
    match_level_df['dominant_outcome'] = match_level_df[
        ['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']].idxmax(axis=1, skipna=True)

    # 独热编码
    dominant_dummies = pd.get_dummies(match_level_df['dominant_outcome'], prefix='dominant')
    match_level_df = pd.concat([match_level_df, dominant_dummies], axis=1)

    # 相对分歧动量
    window_size = 5
    for col in ['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']:
        try:
            # 确保列存在且不为空
            if col in match_level_df.columns and not match_level_df[col].isna().all():
                match_level_df[f'{col}_momentum'] = match_level_df.groupby('league_id')[col].transform(
                    lambda x: x.pct_change(window_size, fill_method=None).rolling(window_size, min_periods=1).mean()
                )
                # 填充可能的NaN值
                match_level_df[f'{col}_momentum'] = match_level_df[f'{col}_momentum'].fillna(0)
        except Exception as e:
            print(f"计算 {col}_momentum 时出错: {str(e)}")

    # 分歧平衡指数
    try:
        match_level_df['balance_index'] = np.arctan2(
            match_level_df['first_draw_sp_std'] - match_level_df['first_win_sp_std'],
            match_level_df['first_lose_sp_std'] - match_level_df['first_win_sp_std']
        )
        # 填充可能的NaN值
        match_level_df['balance_index'] = match_level_df['balance_index'].fillna(0)
    except Exception as e:
        print(f"计算 balance_index 时出错: {str(e)}")

    # 分歧离群检测
    try:
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=0.1)
        # 确保所有需要的列都存在
        required_cols = ['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']
        if all(col in match_level_df.columns for col in required_cols):
            match_level_df['discrepancy_outlier'] = clf.fit_predict(
                match_level_df[required_cols]
            )
    except Exception as e:
        print(f"计算 discrepancy_outlier 时出错: {str(e)}")

    # 博弈论特征
    try:
        match_level_df['nash_ratio'] = (
                (match_level_df['first_win_sp_std'] * match_level_df['first_draw_sp_std']) /
                (match_level_df['first_lose_sp_std'] ** 2 + 1e-6)
        )
        # 填充可能的NaN值
        match_level_df['nash_ratio'] = match_level_df['nash_ratio'].fillna(0)
    except Exception as e:
        print(f"计算 nash_ratio 时出错: {str(e)}")

    try:
        match_level_df['minimax'] = match_level_df[['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']].max(axis=1) - \
                                    match_level_df[['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']].min(axis=1)
        # 填充可能的NaN值
        match_level_df['minimax'] = match_level_df['minimax'].fillna(0)
    except Exception as e:
        print(f"计算 minimax 时出错: {str(e)}")

    # 验证所有特征列是否已生成
    expected_columns = [
        'first_win_sp_std_momentum', 'first_draw_sp_std_momentum', 'first_lose_sp_std_momentum',
        'balance_index', 'discrepancy_outlier', 'nash_ratio', 'minimax'
    ]
    
    missing_columns = [col for col in expected_columns if col not in match_level_df.columns]
    if missing_columns:
        print(f"警告：以下特征列未生成: {missing_columns}")
    return match_level_df

def _process_single_match(group):
    """处理单个比赛的所有赔率数据，返回一行特征"""
    match_id = group.name
    features = {'match_id': match_id}

    # 基础统计特征
    for outcome in ['win', 'draw', 'lose']:
        # 赔率统计
        sp_series = group[f'first_{outcome}_sp']
        features.update({
            f'first_{outcome}_sp_mean': sp_series.mean(),
            f'first_{outcome}_sp_std': sp_series.std(),
            f'first_{outcome}_sp_max': sp_series.max(),
            f'first_{outcome}_sp_min': sp_series.min(),
            f'first_{outcome}_sp_range': sp_series.max() - sp_series.min(),
            f'first_{outcome}_sp_skew': sp_series.skew(),  # 偏度（分歧方向）
            f'first_{outcome}_sp_kurt': sp_series.kurt()  # 峰度（尾部风险）
        })

        # 凯利指数统计
        kelly_series = group[f'first_{outcome}_kelly_index']
        features.update({
            f'first_{outcome}_kelly_index_mean': kelly_series.mean(),
            f'first_{outcome}_kelly_index_std': kelly_series.std(),
            f'first_{outcome}_kelly_index_max': kelly_series.max(),
            f'first_{outcome}_kelly_index_min': kelly_series.min(),
            f'first_{outcome}_kelly_index_range': kelly_series.max() - kelly_series.min(),
            f'first_{outcome}_kelly_index_skew': kelly_series.skew(),  # 偏度（分歧方向）
            f'first_{outcome}_kelly_index_kurt': kelly_series.kurt()  # 峰度（尾部风险）
        })
        # 凯利值分布情况统计
        kelly_distribution_num_series = group[f'first_{outcome}_kelly_index']
        # 大于1.05的
        features[f'{outcome}_kelly_high_val_distribution_num'] = kelly_distribution_num_series.apply(
            lambda x: 1 if x > 1.05 else 0).sum()
        # 小于0.92的
        features[f'{outcome}_kelly_low_val_distribution_num'] = kelly_distribution_num_series.apply(
            lambda x: 1 if x < 0.92 else 0).sum()

        # 极值 机构数
        for target in ['max', 'min']:
            agency_extreme_num_series = group[f'{target}_first_{outcome}_sp']
            features[f'{outcome}_{target}_agency_num'] = agency_extreme_num_series.apply(
                lambda x: 1 if x == target else 0).sum()

    # 重点机构特征
    key_bookmakers = [1000, 11, 57]  # 定义重点机构ID
    for bid in key_bookmakers:
        agency_data = group[group['bookmaker_id'] == bid]
        for outcome in ['win', 'draw', 'lose']:
            key = f'bid_{bid}_{outcome}'
            features[key] = agency_data[f'first_{outcome}_sp'].iloc[0] if not agency_data.empty else np.nan

    features['league_id'] = group['league_id'].max()
    features['nwdl_result'] = group['nwdl_result'].max()

    # 以上结果 进行处理
    for outcome in ['win', 'draw', 'lose']:
        sp_ratio_target_key = f'{outcome}_kelly_sp_ratio'
        kelly_key = f'first_{outcome}_kelly_index_mean'
        outcome_sp_key = f'first_{outcome}_sp_mean'
        features[sp_ratio_target_key] = features[kelly_key] / features[outcome_sp_key]

        # 两者赔率比率
        both_outcome_aver_sp_devision_target_key = f'{outcome}_both_outcome_aver_sp_devision'
        win_outcome_aver_sp_target_key = 'first_win_sp_mean'
        if (outcome == 'win'):
            continue
        cur_outcome_aver_sp_target_key = f'first_{outcome}_sp_mean'

        features[both_outcome_aver_sp_devision_target_key] = (
                features[win_outcome_aver_sp_target_key] / features[cur_outcome_aver_sp_target_key]
        )
        # 两者赔率相减
        both_outcome_aver_sp_sub_target_key = f'{outcome}_both_outcome_aver_sp_sub'
        features[both_outcome_aver_sp_sub_target_key] = (
                features[win_outcome_aver_sp_target_key] - features[cur_outcome_aver_sp_target_key]
        )
    # 将 calculate_odds_difference(group) 合并 到 features
    features.update(calculate_odds_difference(group))

    return pd.Series(features)


def calculate_odds_difference(group):
    features = {}
    # 获取所有唯一的机构ID
    unique_agencies = group['bookmaker_id'].unique()
    # 生成两两组合
    agency_pairs = list(combinations(unique_agencies, 2))
    
    for agency1, agency2 in agency_pairs:
        suffix = f'{agency1}_{agency2}'
        odds1 = group[group['bookmaker_id'] == agency1][
            ['first_win_sp', 'first_draw_sp', 'first_lose_sp']].values
        odds2 = group[group['bookmaker_id'] == agency2][
            ['first_win_sp', 'first_draw_sp', 'first_lose_sp']].values
        if len(odds1) > 0 and len(odds2) > 0:
            diff = odds1[0] - odds2[0]
            features.update({
                f'odds_win_diff_{suffix}': diff[0],
                f'odds_draw_diff_{suffix}': diff[1],
                f'odds_lose_diff_{suffix}': diff[2]
            })
        else:
            features.update({
                f'odds_win_diff_{suffix}': 0,
                f'odds_draw_diff_{suffix}': 0,
                f'odds_lose_diff_{suffix}': 0
            })
    return pd.Series(features)


def create_match_level_future_by_match_group(df):
    """保留所有原有特征，增加关键新特征，保持数据顺序"""


    # 调用 _process_single_match，排除分组列
    match_level_df = df.groupby('match_id', sort=False, group_keys=False).apply(_process_single_match)

    # 保持原始顺序
    match_level_df = match_level_df.reindex(df['match_id'].unique())
    
    # 分歧排名，基于 first_win_sp_std first_draw_sp_std first_lose_sp_std
    match_level_df = add_institution_discrepancy_colum(match_level_df)
    
    return match_level_df


def get_match_level_df(guess_type, unless_colum):
    origin_df = getOriginData(guess_type)
    # 这里可以根据需要做一些数据预处理
    # 移除unless_colum的字段
    origin_df = origin_df.drop(unless_colum, axis=1)
    # 挑选出 类型为 Timestamp的列

    match_level_df = create_match_level_future_by_match_group(origin_df)
    return match_level_df


# 获取训练所需的 match_level_df 和相关配置
def getSelf():
    # 动态选择目标变量和对应的标签
    y_column = 'nwdl_result'  # 根据实际情况选择
    guess_type = 'win_draw_loss'  # 目标类型可以是 'win_draw_loss'（胜平负）、'asian_handicap'（亚盘）、'goals'（进球数）

    useless_cols = ['bet_time']  # 可自定义 # match_id

    match_level_df = get_match_level_df(guess_type, useless_cols)
    return y_column, guess_type, useless_cols, match_level_df


# 动态创建增强特征
def create_features(df, useless_cols=None):
    """创建增强型特征，只选择数值类型的列"""
    if useless_cols is None:
        useless_cols = ['europe_handicap_result', 'match_time', 'match_id', 'league_id', 'nwdl_result']

    df = df.copy()
    
    # 只选择数值类型的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    #挑出非数值类型的列
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    base_cols = [col for col in numeric_cols if col not in useless_cols]
    
    # 新增特征（根据实际情况调整）
    new_cols = []  # 根据需求创建新特征
    return df[base_cols + new_cols]


# 数据预处理：时序分割，特征处理，标准化
def preprocess_data(df, target_column, guess_type, useless_cols=None, test_size=0.2):
    """数据预处理：时序分割，特征处理，标准化"""

    
    # 时序分割
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # 特征处理
    X_train = create_features(train_df, useless_cols)
    X_test = create_features(test_df, useless_cols)
    
    # 标签处理
    y_train = train_df[target_column]
    y_train, label_map = map_labels(train_df[target_column], guess_type)
    y_test = np.array([label_map[str(label)] for label in test_df[target_column]])
    
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
        target_names = [str(c) for c in np.unique(target_names)]
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
        n_games = len(X_test)  # 防止超出测试集最大长度
        print(f"注意：请求的{n_games}场超过测试集最大长度，已自动调整为{len(X_test)}场")

    recent_X = X_test[-n_games:]  # 取最后N场特征
    recent_y = y_test[-n_games:]  # 取最后N场标签

    y_pred = model.predict(recent_X)
    return balanced_accuracy_score(recent_y, y_pred)


def map_labels(y, guess_type):
    """
    对不同预测类型进行标签映射，确保标签从0开始连续
    :param y: 原始标签（可能是字符串或非连续整数）
    :param guess_type: 预测类型（'win_draw_loss', 'asian_handicap', 'goals'等）
    :return: 映射后的标签和映射关系字典
    """
    # 创建标签映射关系
    if guess_type == 'win_draw_loss':
        label_map = {'0': 0, '1': 1, '3': 2}
    elif guess_type == 'asian_handicap':
        label_map = {'下盘': 0, '上盘': 1}
    elif guess_type == 'goals':
        label_map = {str(i): i for i in range(8)}
        label_map['7+'] = 8
    else:
        # 自动处理未知类型：将唯一值映射为0~n-1
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # 转换标签
    y_mapped = np.array([label_map[str(label)] for label in y])

    # 验证映射结果
    unique_mapped = np.unique(y_mapped)
    expected = np.arange(len(unique_mapped))
    if not np.array_equal(unique_mapped, expected):
        raise ValueError(f"映射后标签不连续，得到：{unique_mapped}，期望：{expected}")

    return y_mapped, label_map


# 特征重要性可视化
def plot_feature_importance(models, feature_names):
    """特征重要性可视化"""
    for model_name, model_info in models.items():
        model = model_info['best_estimator']
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            pd.Series(model.feature_importances_, index=feature_names).nlargest(15).plot(kind='barh')
            plt.title(f'{model_name} Top 15 Feature Importances')
            plt.show()


# 主程序
def get_target_names(prediction_type):
    """根据预测类型生成相应的标签"""
    if prediction_type == 'win_draw_loss':
        target_names = ['负', '平', '胜']
    elif prediction_type == 'asian_handicap':
        target_names = ['上盘', '下盘']
    elif prediction_type == 'goals':
        target_names = [str(i) for i in range(8)] + ['7+']  # 进球数0-7+
    else:
        raise ValueError(f"未识别的目标类型: {prediction_type}")
    return target_names


if __name__ == '__main__':
    # 获取数据
    y_column, guess_type, useless_cols, match_level_df = getSelf()

    # 数据预处理
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(
        match_level_df, y_column, guess_type, useless_cols)

    # 获取特征名称
    feature_names = create_features(match_level_df, useless_cols).columns.tolist()

    # 类别权重计算
    class_weights = compute_class_weights(y_train)

    # 获取模型和参数网格
    models = get_models()
    param_grids = get_param_grids()

    # 训练并评估模型
    best_models = train_and_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test, param_grids, models)

    # 特征重要性可视化
    plot_feature_importance(best_models, feature_names)
