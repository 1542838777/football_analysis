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
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
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

    # # 添加赔率排名
    # rank_cols  = ['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']
    # match_level_df = add_rank_columns(match_level_df, rank_cols)


    # 分歧平衡指数
    try:
        match_level_df['balance_index'] = np.arctan2(
            match_level_df['first_draw_sp_std'] - match_level_df['first_win_sp_std'],
            match_level_df['first_lose_sp_std'] - match_level_df['first_win_sp_std']
        )
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

def _process_single_match(group,agency_pairs):
    """处理单个比赛的所有赔率数据，返回一行特征"""
    match_id = group.name
    features = {'match_id': match_id}

    # 基础统计特征
    for outcome in ['win', 'draw', 'lose']:
        # 赔率统计
        sp_series = group[f'first_{outcome}_sp']
        if len(sp_series.dropna()) >= 3:  # 确保有足够的数据计算统计量
            features.update({
                f'first_{outcome}_sp_mean': sp_series.mean(),
                f'first_{outcome}_sp_std': sp_series.dropna().size >= 2 and sp_series.std() or 0,  # 判断长度是否大于等于2，如果没有，默认填写0
                f'first_{outcome}_sp_max': sp_series.max(),
                f'first_{outcome}_sp_min': sp_series.min(),
                f'first_{outcome}_sp_range': sp_series.max() - sp_series.min(),
                f'first_{outcome}_sp_skew': sp_series.skew(),
                f'first_{outcome}_sp_kurt': sp_series.kurt()
            })
        else:
            features.update({
                f'first_{outcome}_sp_mean': sp_series.mean(),
                f'first_{outcome}_sp_std': sp_series.dropna().size >= 2 and sp_series.std() or 0,  # 判断长度是否大于等于2，如果没有，默认填写0
                f'first_{outcome}_sp_max': sp_series.max(),
                f'first_{outcome}_sp_min': sp_series.min(),
                f'first_{outcome}_sp_range': sp_series.max() - sp_series.min(),
                f'first_{outcome}_sp_skew': 0,
                f'first_{outcome}_sp_kurt': 0
            })

        # 凯利指数统计
        kelly_series = group[f'first_{outcome}_kelly_index']
        if len(kelly_series.dropna()) >= 3:
            features.update({
                f'first_{outcome}_kelly_index_mean': kelly_series.mean(),
                f'first_{outcome}_kelly_index_std': kelly_series.dropna().size >= 2 and kelly_series.std() or 0,  # 判断长度是否大于等于2，如果没有，默认填写0
                f'first_{outcome}_kelly_index_max': kelly_series.max(),
                f'first_{outcome}_kelly_index_min': kelly_series.min(),
                f'first_{outcome}_kelly_index_range': kelly_series.max() - kelly_series.min(),
                f'first_{outcome}_kelly_index_skew': kelly_series.skew(),
                f'first_{outcome}_kelly_index_kurt': kelly_series.kurt()
            })
        else:
            features.update({
                f'first_{outcome}_kelly_index_mean': kelly_series.mean(),
                f'first_{outcome}_kelly_index_std': kelly_series.dropna().size >= 2 and kelly_series.std() or 0,  # 判断长度是否大于等于2，如果没有，默认填写0
                f'first_{outcome}_kelly_index_max': kelly_series.max(),
                f'first_{outcome}_kelly_index_min': kelly_series.min(),
                f'first_{outcome}_kelly_index_range': kelly_series.max() - kelly_series.min(),
                f'first_{outcome}_kelly_index_skew': 0,
                f'first_{outcome}_kelly_index_kurt': 0
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
        # 赔率统计
    sp_series = group['first_back_rate']
    features.update({
        f'first_back_rate_sp_mean': sp_series.mean(),
        f'first_back_rate_sp_std': sp_series.dropna().size >= 2 and sp_series.std() or 0,
        # 判断长度是否大于等于2，如果没有，默认填写0
        f'first_back_rate_sp_max': sp_series.max(),
        f'first_back_rate_sp_min': sp_series.min(),
        f'first_back_rate_sp_range': sp_series.max() - sp_series.min(),
        f'first_back_rate_sp_skew': sp_series.skew(),
        f'first_back_rate_sp_kurt': sp_series.kurt()
    })
    # 重点机构特征
    key_bookmakers = [82,39,6,9,64,1000,39,11,57]  # 定义重点机构ID
    for bid in key_bookmakers:
        agency_data = group[group['bookmaker_id'] == bid]
        for outcome in ['win', 'draw', 'lose']:
            key = f'bid_{bid}_{outcome}'
            if not agency_data.empty:
                features[key] = agency_data[f'first_{outcome}_sp'].iloc[0]
            else:
                # 如果机构没有数据，使用该场比赛的平均值
                features[key] = group[f'first_{outcome}_sp'].mean()

    features['league_id'] = group['league_id'].max()
    #group是否含nwdl_result
    if 'nwdl_result' in group.columns:
        features['nwdl_result'] = group['nwdl_result'].max()

    # 以上结果 进行处理
    for outcome in ['win', 'draw', 'lose']:
        sp_ratio_target_key = f'{outcome}_kelly_sp_ratio'
        kelly_key = f'first_{outcome}_kelly_index_mean'
        outcome_sp_key = f'first_{outcome}_sp_mean'
        if features[outcome_sp_key] != 0:  # 避免除以0
            features[sp_ratio_target_key] = features[kelly_key] / features[outcome_sp_key]
        else:
            features[sp_ratio_target_key] = 0

            # 两者赔率比率
        both_outcome_aver_sp_devision_target_key = f'win_{outcome}_both_outcome_aver_sp_devision'
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

    features.update(calculate_odds_difference(group, agency_pairs))



    # # 添加排名
    # odds_mean_rank_cols = ['first_win_sp_mean', 'first_draw_sp_mean', 'first_lose_sp_mean']
    # odds_std_rank_cols = ['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']
    # kelly_mean_rank_cols = ['first_win_kelly_index_mean', 'first_draw_kelly_index_mean', 'first_lose_kelly_index_mean']
    # kelly_std_rank_cols = ['first_win_kelly_index_std', 'first_draw_kelly_index_std', 'first_lose_kelly_index_std']
    # features = add_rank_columns(features, odds_mean_rank_cols)
    # features = add_rank_columns(features, odds_std_rank_cols)
    # features = add_rank_columns(features, kelly_mean_rank_cols)
    # features = add_rank_columns(features, kelly_std_rank_cols)

    return pd.Series(features)


def add_rank_columns(features, rank_cols):
    """
    为特征字典添加横向排名

    Args:
        features: 特征字典
        rank_cols: 需要排名的列名列表
    Returns:
        添加了排名的特征字典
    """
    # 从字典中提取需要排名的值
    values = [features[col] for col in rank_cols]

    # 计算排名
    try:
        ranks = pd.Series(values).rank(method='dense')
    except Exception as e:
        print(f'match_id: {features["match_id"]}')
        raise


    # 添加排名到特征字典
    for col, rank in zip(rank_cols, ranks):
        features[f'{col}_rank'] = int(rank)

    return features
def calculate_odds_difference(group,agency_pairs):
    features = {}
    # 生成两两组合
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

    unique_agencies = [110,3,82,6,64,9,57,106,39,84,1000]
    unique_agencies = [82,39,110,3,84,6,64,9,57,106,39,84,1000]
    unique_agencies = [6,9,39,84,110,64,1000]
    unique_agencies = [ 64,39, 84]
    uiniqyue_agencies = [82,39,6,9,64]
    # 生成两两组合
    agency_pairs = list(combinations(unique_agencies, 2))
    # 调用 _process_single_match，排除分组列
    match_level_df = df.groupby('match_id', sort=False, group_keys=False).apply(_process_single_match,agency_pairs)

    # 保持原始顺序
    match_level_df = match_level_df.reindex(df['match_id'].unique())

    # 分歧排名，基于 first_win_sp_std first_draw_sp_std first_lose_sp_std
    # match_level_df的league_id强转为int类型
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
    """创建增强型特征"""
    if useless_cols is None:
        useless_cols = ['europe_handicap_result', 'match_time', 'match_id', 'league_id', 'nwdl_result']

    df = df.copy()

    # 只选择数值类型的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    #
    base_cols = [col for col in numeric_cols if col not in useless_cols]

    # 检查并处理缺失值
    missing_cols = df[base_cols].columns[df[base_cols].isna().all()].tolist()
    if missing_cols:
        print(f"以下列完全缺失，将被移除: {missing_cols}")
        base_cols = [col for col in base_cols if col not in missing_cols]

    # 处理NaN值
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(df[base_cols])

    # 创建新的DataFrame
    features_df = pd.DataFrame(imputed_data, columns=base_cols, index=df.index)

    # 添加基础特征
    for col in base_cols:
        # 为std相关的特征添加统计特征
        if 'std' in col or 'mean' in col:
            features_df[f'{col}_rank'] = features_df[col].rank(pct=True)
            features_df[f'{col}_zscore'] = (features_df[col] - features_df[col].mean()) / features_df[col].std()

    # 添加比率特征
    sp_mean_cols = [col for col in base_cols if 'sp_mean' in col]
    if len(sp_mean_cols) >= 2:
        for i in range(len(sp_mean_cols)):
            for j in range(i+1, len(sp_mean_cols)):
                col1, col2 = sp_mean_cols[i], sp_mean_cols[j]
                features_df[f'{col1}_{col2}_ratio'] = features_df[col1] / features_df[col2]
                features_df[f'{col1}_{col2}_diff'] = features_df[col1] - features_df[col2]

    # 添加凯利指数相关特征
    kelly_cols = [col for col in base_cols if 'kelly' in col.lower()]
    for col in kelly_cols:
        if 'mean' in col:
            features_df[f'{col}_rank'] = features_df[col].rank(pct=True)
            features_df[f'{col}_zscore'] = (features_df[col] - features_df[col].mean()) / features_df[col].std()

    return features_df


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

    # 确保训练集和测试集的特征一致
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # 保存特征名称
    feature_names = X_train.columns.tolist()

    # 标签处理
    y_train = train_df[target_column]
    y_train, label_map = map_labels(train_df[target_column], guess_type)
    y_test = np.array([label_map[str(label)] for label in test_df[target_column]])

    # 处理NaN值
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=feature_names,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed),
        columns=feature_names,
        index=X_test.index
    )

    # 使用SMOTE处理类别不平衡
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # 获取特征名称
    selected_feature_names = feature_names

    return (X_train_balanced,
            X_test_scaled,
            y_train_balanced, y_test, scaler, selected_feature_names)


# 类别权重计算
def compute_class_weights(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weights_dict = {i: w for i, w in zip(classes, class_weights)}
    return weights_dict


# 定义多个模型
def get_models():
    models = {
        'XGBoost': XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            scale_pos_weight=1.5,
            tree_method='hist',  # 使用直方图算法加速训练
            grow_policy='lossguide'  # 使用损失导向的生长策略
        ),
        'LightGBM': LGBMClassifier(
            objective='multiclass',
            metric='multi_logloss',
            class_weight='balanced',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9
        ),
        'RandomForest': RandomForestClassifier(
            class_weight='balanced',
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt'
        ),
        'SVM': SVC(
            probability=True,
            kernel='rbf',  # 使用RBF核
            class_weight='balanced',
            gamma='scale',
            C=1.0
        )
    }
    return models


# 为不同模型设置不同的参数网格
def get_param_grids():
    param_grids = {
        'XGBoost': {
            'max_depth': [ 3],#ok
            'learning_rate': [ 0.04],#todo[0.01,0.03] ok
            'subsample': [0.8 ],#ok
            'colsample_bytree':[  1.0 ],#todo[0.8,0.9,1.1]ok
            'n_estimators': [25]#todo[50,125]#ok
        },
        'LightGBM': {
            'num_leaves': [18],#todo[22,40]#ok
            'learning_rate': [0.03],# [0.02, 0.04]#ok
            'n_estimators': [100]#todo [50,125]#ok
        },
        'RandomForest': {
            'n_estimators': [150],#todo[75,125] 小于200 ok
            'max_depth': [7],#todo [3,5,10] 小于10 大于5
            'min_samples_split': [4] #todo[1,3]
        },
        'SVM': {
            'C': [0.2],#大于0.1 小于0.5
            'kernel': ['linear'],#ok
            'gamma': ['scale']#ok
        }
    }
    return param_grids


def analyze_feature_importance(model, X_train, model_name, feature_names=None):
    """分析并打印模型的特征重要性

    Args:
        model: 训练好的模型
        X_train: 训练数据
        model_name: 模型名称
        feature_names: 特征名称列表
    """
    print(f"\n{model_name} 模型的特征重要性（按重要性降序排列）：")

    # 获取特征名称
    if feature_names is None:
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]

    # 根据不同模型类型获取特征重要性
    if hasattr(model, 'feature_importances_'):
        # 适用于XGBoost、LightGBM、RandomForest等
        importances = model.feature_importances_
        if len(importances) != len(feature_names):
            print(f"警告：特征重要性数量({len(importances)})与特征名称数量({len(feature_names)})不匹配")
            # 取较小的长度
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        print(importance_df.to_string())
    elif hasattr(model, 'coef_'):
        # 适用于SVM等线性模型
        coef = model.coef_
        if len(coef[0]) != len(feature_names):
            print(f"警告：系数数量({len(coef[0])})与特征名称数量({len(feature_names)})不匹配")
            # 取较小的长度
            min_len = min(len(coef[0]), len(feature_names))
            coef = coef[:, :min_len]
            feature_names = feature_names[:min_len]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef[0]  # 对于多分类，可能需要处理多个系数
        }).sort_values('coefficient', ascending=False)
        print(importance_df.to_string())
    else:
        print("该模型不支持特征重要性分析")

    return importance_df if 'importance_df' in locals() else None


def train_and_evaluate_models(X_train, y_train, X_test, y_test, param_grids, models, feature_names=None):
    best_models = {}
    estimators = []  # 用于存储所有训练好的模型

    for model_name, model in models.items():
        print(f"\n正在调参 {model_name} ...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=TimeSeriesSplit(n_splits=3),
            scoring='balanced_accuracy',
            n_jobs=2,
            verbose=2
        )

        # 转换数据类型为float32以减少内存使用
        X_train_32 = X_train.astype(np.float32)
        X_test_32 = X_test.astype(np.float32)

        grid_search.fit(X_train_32, y_train)
        best_models[model_name] = {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

        # 将训练好的模型添加到estimators列表
        estimators.append((model_name, grid_search.best_estimator_))

        # 模型评估
        y_pred = grid_search.best_estimator_.predict(X_test_32)
        print(f"\n{model_name} 模型的最佳参数组合：")
        print(grid_search.best_params_)
        print(f"\n{model_name} 模型的测试集表现：")
        print(f"平衡准确率: {balanced_accuracy_score(y_test, y_pred):.2%}")
        target_names = np.unique(y_train)
        target_names = [str(c) for c in np.unique(target_names)]
        print(classification_report(y_test, y_pred, target_names=target_names))

        # 分析特征重要性
        try:
            analyze_feature_importance(grid_search.best_estimator_, X_train_32, model_name, feature_names)
        except Exception as e:
            print(f"分析特征重要性时出错: {str(e)}")

        # 计算最近N场的准确率
        for n in [20, 150]:
            acc = get_recent_n_accuracy(
                grid_search.best_estimator_,
                X_test_32,
                y_test,
                n
            )
            print(f"\n{model_name}模型最近{n}场平衡准确率: {acc:.2%}")

    # 创建投票集成模型
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',  # 使用软投票，考虑预测概率
        weights=[1, 1, 1, 1]  # 可以调整权重
    )

    # 训练投票集成模型
    print("\n训练投票集成模型...")
    voting_clf.fit(X_train_32, y_train)

    # 评估投票集成模型
    y_pred_voting = voting_clf.predict(X_test_32)
    print("\n投票集成模型的测试集表现：")
    print(f"平衡准确率: {balanced_accuracy_score(y_test, y_pred_voting):.2%}")
    print(classification_report(y_test, y_pred_voting, target_names=target_names))

    # 计算投票集成模型的最近N场准确率
    for n in [20, 150]:
        acc = get_recent_n_accuracy(
            voting_clf,
            X_test_32,
            y_test,
            n
        )
        print(f"\n投票集成模型最近{n}场平衡准确率: {acc:.2%}")

    # 添加投票集成模型到best_models
    best_models['Voting'] = {
        'best_estimator': voting_clf,
        'best_params': None,
        'best_score': balanced_accuracy_score(y_test, y_pred_voting)
    }

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
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(
        match_level_df, y_column, guess_type, useless_cols)

    # 类别权重计算
    class_weights = compute_class_weights(y_train)

    # 获取模型和参数网格
    models = get_models()
    param_grids = get_param_grids()

    # 训练并评估模型
    best_models = train_and_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test, param_grids, models, feature_names)

    # 特征重要性可视化
    plot_feature_importance(best_models, feature_names)
