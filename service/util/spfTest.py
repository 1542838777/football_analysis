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
def _add_basic_discrepancy_features(df):
    """
    添加基础的机构分歧特征

    Args:
        df: 数据框

    Returns:
        添加了基础分歧特征的数据框
    """
    # 市场分歧指数 - 三种赔率标准差的平均值
    df['market_disagreement'] = (
        df['first_win_sp_std'] +
        df['first_draw_sp_std'] +
        df['first_lose_sp_std']
    ) / 3

    # 分歧方向强度 - 衡量某一结果的分歧相对于其他结果的强度
    df['ddi_win'] = df['first_win_sp_std'] / (
        df['first_draw_sp_std'] + df['first_lose_sp_std'] + 1e-6)  # 避免除零
    df['ddi_draw'] = df['first_draw_sp_std'] / (
        df['first_win_sp_std'] + df['first_lose_sp_std'] + 1e-6)
    df['ddi_lose'] = df['first_lose_sp_std'] / (
        df['first_win_sp_std'] + df['first_draw_sp_std'] + 1e-6)

    # 构建两两差异矩阵 - 计算各赔率标准差之间的差值
    df['win_draw_gap'] = df['first_win_sp_std'] - df['first_draw_sp_std']
    df['win_lose_gap'] = df['first_win_sp_std'] - df['first_lose_sp_std']
    df['draw_lose_gap'] = df['first_draw_sp_std'] - df['first_lose_sp_std']

    # 符号编码 - 将差异的方向编码为一个整数
    df['gap_direction'] = (
        (df['win_draw_gap'] > 0).astype(int) * 100 +
        (df['win_lose_gap'] > 0).astype(int) * 10 +
        (df['draw_lose_gap'] > 0).astype(int)
    )

    return df


def _add_entropy_features(df):
    """
    添加基于熵的分歧特征

    Args:
        df: 数据框

    Returns:
        添加了熵相关特征的数据框
    """
    # 熵值分歧指数 - 使用信息熵衡量分歧程度
    def calculate_entropy(row):
        total = row.sum()
        probs = row / total
        return -np.sum(probs * np.log(probs + 1e-6))  # 避免log(0)

    df['disagreement_entropy'] = df[
        ['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']].apply(calculate_entropy, axis=1)

    # 主导分歧指标 - 哪个结果的分歧最大
    df['dominant_outcome'] = df[
        ['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']].idxmax(axis=1, skipna=True)

    return df


def _add_advanced_discrepancy_features(df):
    """
    添加高级分歧特征，包括平衡指数、离群检测和博弈论特征

    Args:
        df: 数据框

    Returns:
        添加了高级分歧特征的数据框
    """
    # 分歧平衡指数 - 使用极坐标角度表示分歧的平衡性
    try:
        df['balance_index'] = np.arctan2(
            df['first_draw_sp_std'] - df['first_win_sp_std'],
            df['first_lose_sp_std'] - df['first_win_sp_std']
        )
        df['balance_index'] = df['balance_index'].fillna(0)
    except Exception as e:
        print(f"计算 balance_index 时出错: {str(e)}")

    # 分歧离群检测 - 使用隔离森林检测异常分歧模式
    try:
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=0.1)  # 假设10%的数据是异常值
        required_cols = ['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']
        if all(col in df.columns for col in required_cols):
            df['discrepancy_outlier'] = clf.fit_predict(df[required_cols])
    except Exception as e:
        print(f"计算 discrepancy_outlier 时出错: {str(e)}")

    # 博弈论特征 - Nash比率和极小极大值
    try:
        # Nash比率 - 胜平赔率标准差的乘积与负赔率标准差平方的比值
        df['nash_ratio'] = (
            (df['first_win_sp_std'] * df['first_draw_sp_std']) /
            (df['first_lose_sp_std'] ** 2 + 1e-6)  # 避免除零
        )
        df['nash_ratio'] = df['nash_ratio'].fillna(0)

        # 极小极大值 - 最大分歧与最小分歧的差值
        df['minimax'] = df[['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']].max(axis=1) - \
                         df[['first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']].min(axis=1)
        df['minimax'] = df['minimax'].fillna(0)
    except Exception as e:
        print(f"计算博弈论特征时出错: {str(e)}")

    return df


def add_institution_discrepancy_colum(match_level_df):
    """
    添加机构分歧相关特征

    Args:
        match_level_df: 比赛级别的数据框，包含各机构赔率标准差

    Returns:
        添加了分歧特征的数据框
    """
    # 1. 添加基础分歧特征
    match_level_df = _add_basic_discrepancy_features(match_level_df)

    # 2. 添加熵相关特征
    match_level_df = _add_entropy_features(match_level_df)

    # 3. 添加高级分歧特征
    match_level_df = _add_advanced_discrepancy_features(match_level_df)

    # 4. 验证所有特征列是否已生成
    expected_columns = [
        'first_win_sp_std_momentum', 'first_draw_sp_std_momentum', 'first_lose_sp_std_momentum',
        'balance_index', 'discrepancy_outlier', 'nash_ratio', 'minimax'
    ]

    missing_columns = [col for col in expected_columns if col not in match_level_df.columns]
    if missing_columns:
        print(f"警告：以下特征列未生成: {missing_columns}")

    return match_level_df

def _calculate_series_stats(series, prefix, min_samples=3):
    """
    计算数据序列的统计特征

    Args:
        series: 数据序列
        prefix: 特征名前缀
        min_samples: 计算高阶统计量所需的最小样本数

    Returns:
        包含统计特征的字典
    """
    stats = {}
    # 基本统计量总是计算
    stats.update({
        f'{prefix}_mean': series.mean(),
        f'{prefix}_std': series.dropna().size >= 2 and series.std() or 0,
        f'{prefix}_max': series.max(),
        f'{prefix}_min': series.min(),
        f'{prefix}_range': series.max() - series.min(),
    })

    # 高阶统计量需要足够的样本
    if len(series.dropna()) >= min_samples:
        stats.update({
            f'{prefix}_skew': series.skew(),
            f'{prefix}_kurt': series.kurt()
        })
    else:
        stats.update({
            f'{prefix}_skew': 0,
            f'{prefix}_kurt': 0
        })

    return stats


def _extract_odds_features(group, features):
    """
    提取各种赔率相关的统计特征

    Args:
        group: 比赛分组数据
        features: 特征字典，将被更新

    Returns:
        更新后的特征字典
    """
    # 处理胜平负赔率
    for outcome in ['win', 'draw', 'lose']:
        # 赔率统计特征
        sp_series = group[f'first_{outcome}_sp']
        features.update(_calculate_series_stats(sp_series, f'first_{outcome}_sp'))

        # 凯利指数统计特征
        kelly_series = group[f'first_{outcome}_kelly_index']
        features.update(_calculate_series_stats(kelly_series, f'first_{outcome}_kelly_index'))

        # 凯利值分布情况统计
        kelly_distribution_num_series = group[f'first_{outcome}_kelly_index']
        # 统计高值(>1.05)和低值(<0.92)的数量
        features[f'{outcome}_kelly_high_val_distribution_num'] = kelly_distribution_num_series.apply(
            lambda x: 1 if x > 1.05 else 0).sum()
        features[f'{outcome}_kelly_low_val_distribution_num'] = kelly_distribution_num_series.apply(
            lambda x: 1 if x < 0.92 else 0).sum()

        # 统计极值机构数量
        for target in ['max', 'min']:
            agency_extreme_num_series = group[f'{target}_first_{outcome}_sp']
            features[f'{outcome}_{target}_agency_num'] = agency_extreme_num_series.apply(
                lambda x: 1 if x == target else 0).sum()

    # 返还率统计特征
    sp_series = group['first_back_rate']
    features.update(_calculate_series_stats(sp_series, 'first_back_rate_sp'))

    return features


def _extract_key_bookmaker_features(group, features):
    """
    提取重点博彩公司的赔率特征

    Args:
        group: 比赛分组数据
        features: 特征字典，将被更新

    Returns:
        更新后的特征字典
    """
    # 定义重点机构ID
    key_bookmakers = [3,
            11,99,63,75,64,39,84,91,68,79,22,32,6,24,126,82,161,18,74,57,192,93,72,47,25,80,17,127,9,106,48,115,42,121,130,70,60,1000,
    110]

    for bid in key_bookmakers:
        agency_data = group[group['bookmaker_id'] == bid]
        for outcome in ['win', 'draw', 'lose']:
            key = f'bid_{bid}_{outcome}'
            if not agency_data.empty:
                features[key] = agency_data[f'first_{outcome}_sp'].iloc[0]
            else:
                # 如果机构没有数据，使用该场比赛的平均值
                features[key] = group[f'first_{outcome}_sp'].mean()

    return features


def _calculate_derived_features(features):
    """
    计算衍生特征，如比率、差值等

    Args:
        features: 包含基础特征的字典

    Returns:
        添加了衍生特征的字典
    """
    # 计算凯利指数与赔率的比率
    for outcome in ['win', 'draw', 'lose']:
        sp_ratio_target_key = f'{outcome}_kelly_sp_ratio'
        kelly_key = f'first_{outcome}_kelly_index_mean'
        outcome_sp_key = f'first_{outcome}_sp_mean'

        if features[outcome_sp_key] != 0:  # 避免除以0
            features[sp_ratio_target_key] = features[kelly_key] / features[outcome_sp_key]
        else:
            features[sp_ratio_target_key] = 0

        # 计算胜赔与其他赔率的比率和差值
        if outcome != 'win':
            # 赔率比率
            both_outcome_aver_sp_devision_target_key = f'win_{outcome}_both_outcome_aver_sp_devision'
            win_outcome_aver_sp_target_key = 'first_win_sp_mean'
            cur_outcome_aver_sp_target_key = f'first_{outcome}_sp_mean'

            features[both_outcome_aver_sp_devision_target_key] = (
                features[win_outcome_aver_sp_target_key] / features[cur_outcome_aver_sp_target_key]
            )

            # 赔率差值
            both_outcome_aver_sp_sub_target_key = f'{outcome}_both_outcome_aver_sp_sub'
            features[both_outcome_aver_sp_sub_target_key] = (
                features[win_outcome_aver_sp_target_key] - features[cur_outcome_aver_sp_target_key]
            )

    return features


def _process_single_match(group, agency_pairs):
    """
    处理单个比赛的所有赔率数据，提取特征

    Args:
        group: 单场比赛的所有赔率数据分组
        agency_pairs: 需要计算赔率差异的机构对列表

    Returns:
        包含该场比赛所有特征的Series
    """
    # 初始化特征字典
    match_id = group.name
    features = {'match_id': match_id}

    # 1. 提取赔率相关统计特征
    features = _extract_odds_features(group, features)

    # 2. 提取重点博彩公司特征
    features = _extract_key_bookmaker_features(group, features)

    # 3. 记录联赛ID和比赛结果
    features['league_id'] = group['league_id'].max()
    if 'nwdl_result' in group.columns:
        features['nwdl_result'] = group['nwdl_result'].max()

    # 4. 计算衍生特征
    features = _calculate_derived_features(features)

    # 5. 添加机构间赔率差异特征
    features.update(calculate_odds_difference(group, agency_pairs))

    # 注释掉的排名特征代码
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
        ranks = pd.Series(values).rank(method='dense', axis=0)
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

    # unique_agencies = [110,3,82,6,64,9,57,106,39,84,1000]
    # unique_agencies = [82,39,110,3,84,6,64,9,57,106,39,84,1000]
    # unique_agencies = [6,9,39,84,110,64,1000]
    unique_agencies = [ 64,39]
    # uiniqyue_agencies = [82,39,6,9,64]
    # uiniqyue_agencies = [110,84,9,82] # XGBoost
    # uiniqyue_agencies = [82,57,84,1000,110,82,39] # LightGBM
    # uiniqyue_agencies = [ ] # RandomForest
    # uiniqyue_agencies = [3,1000,82,110,64,106,9,84,39] # svm
    # 生成两两组合
    agency_pairs = list(combinations(unique_agencies, 2))
    # 调用 _process_single_match，排除分组列
    match_level_df = df.groupby('match_id', sort=False, group_keys=False).apply(_process_single_match,agency_pairs)

    # 保持原始顺序
    match_level_df = match_level_df.reindex(df['match_id'].unique())#193

    # 分歧排名，基于 first_win_sp_std first_draw_sp_std first_lose_sp_std
    # match_level_df的league_id强转为int类型
    match_level_df = add_institution_discrepancy_colum(match_level_df)

    return match_level_df #207


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


def _prepare_feature_data(df, useless_cols):
    """
    准备特征数据，处理缺失值和数据类型

    Args:
        df: 原始数据框
        useless_cols: 需要排除的列

    Returns:
        处理后的特征数据框和基础列名列表
    """
    # 只选择数值类型的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_cols = [col for col in numeric_cols if col not in useless_cols]

    # 检查并处理完全缺失的列
    missing_cols = df[base_cols].columns[df[base_cols].isna().all()].tolist()
    if missing_cols:
        print(f"以下列完全缺失，将被移除: {missing_cols}")
        base_cols = [col for col in base_cols if col not in missing_cols]

    # 使用均值填充缺失值
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(df[base_cols])

    # 创建新的DataFrame
    features_df = pd.DataFrame(imputed_data, columns=base_cols, index=df.index)

    return features_df, base_cols


def _add_zscore_features(features_df, feature_cols):
    """
    为指定特征添加Z-score标准化特征

    Args:
        features_df: 特征数据框
        feature_cols: 需要添加Z-score的列名列表

    Returns:
        添加了Z-score特征的数据框
    """
    for col in feature_cols:
        # 避免除以0
        std_val = features_df[col].std()
        if std_val > 0:
            features_df[f'{col}_zscore'] = (features_df[col] - features_df[col].mean()) / std_val
        else:
            features_df[f'{col}_zscore'] = 0

    return features_df


def _add_rank_features(features_df, feature_groups):
    """
    为不同类型的特征组添加横向排名特征

    Args:
        features_df: 特征数据框
        feature_groups: 字典，包含不同类型的特征列名列表

    Returns:
        添加了排名特征的数据框
    """
    # 对每组特征进行横向排名
    for group_name, cols in feature_groups.items():
        if len(cols) > 0:
            # 使用百分比排名，使排名在[0,1]范围内
            ranks = features_df[cols].rank(axis=1, pct=True)
            for col in cols:
                features_df[f'{col}_rank'] = ranks[col]

    return features_df


def _add_ratio_features(features_df, cols):
    """
    为指定列添加比率和差值特征

    Args:
        features_df: 特征数据框
        cols: 需要计算比率和差值的列名列表

    Returns:
        添加了比率和差值特征的数据框
    """
    if len(cols) >= 2:
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                col1, col2 = cols[i], cols[j]
                # 比率特征，避免除以0
                features_df[f'{col1}_{col2}_ratio'] = features_df[col1] / (features_df[col2] + 1e-6)
                # 差值特征
                features_df[f'{col1}_{col2}_diff'] = features_df[col1] - features_df[col2]

    return features_df


# 动态创建增强特征
def create_features(df, useless_cols=None):
    """
    创建增强型特征

    Args:
        df: 原始数据框
        useless_cols: 需要排除的列，默认为空

    Returns:
        包含原始和增强特征的数据框
    """
    # 设置默认排除列
    if useless_cols is None:
        useless_cols = ['europe_handicap_result', 'match_time', 'match_id', 'league_id', 'nwdl_result']

    # 复制数据框避免修改原始数据
    df = df.copy()

    # 1. 准备特征数据
    features_df, base_cols = _prepare_feature_data(df, useless_cols)

    # 2. 按特征类型分组
    # 收集不同类型的特征列
    feature_groups = {
        'kelly_mean': [col for col in base_cols if 'kelly_index_mean' in col],
        'kelly_std': [col for col in base_cols if 'kelly_index_std' in col],
        'sp_mean': [col for col in base_cols if 'sp_mean' in col],
        'sp_std': [col for col in base_cols if 'sp_std' in col]
    }

    # 3. 添加Z-score标准化特征
    all_stat_cols = []
    for cols in feature_groups.values():
        all_stat_cols.extend(cols)
    features_df = _add_zscore_features(features_df, all_stat_cols)

    # 4. 添加横向排名特征
    features_df = _add_rank_features(features_df, feature_groups)

    # 5. 添加比率和差值特征
    features_df = _add_ratio_features(features_df, feature_groups['sp_mean'])

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


def _select_features_for_model(model, model_name, X_train, y_train, feature_names):
    """
    为模型选择特征子集

    Args:
        model: 模型实例
        model_name: 模型名称
        X_train: 训练数据
        y_train: 训练标签
        feature_names: 特征名称列表

    Returns:
        选择后的训练数据、测试数据和选择的特征列表
    """
    # 对于树模型，使用特征选择
    if model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
        # 先训练一个简单模型用于特征选择
        from sklearn.feature_selection import SelectFromModel
        temp_model = model
        temp_model.fit(X_train, y_train)

        # 基于特征重要性选择特征
        selector = SelectFromModel(temp_model, threshold='mean', prefit=True)
        feature_mask = selector.get_support()
        selected_features = [feature for feature, selected in zip(feature_names, feature_mask) if selected]

        print(f"为 {model_name} 选择了 {len(selected_features)} 个特征")

        # 使用选定的特征子集
        X_train_selected = selector.transform(X_train)

        return X_train_selected, selected_features, selector
    else:
        # 对于非树模型，使用全部特征
        return X_train, feature_names, None


def _train_and_evaluate_base_model(model, model_name, param_grid, X_train, y_train, X_test, y_test, feature_names):
    """
    训练和评估单个基础模型

    Args:
        model: 模型实例
        model_name: 模型名称
        param_grid: 参数网格
        X_train: 训练数据
        y_train: 训练标签
        X_test: 测试数据
        y_test: 测试标签
        feature_names: 特征名称列表

    Returns:
        模型评估结果字典和训练好的模型
    """
    print(f"\n正在调参 {model_name} ...")

    # 2. 网格搜索调参
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='balanced_accuracy',
        n_jobs=2,
        verbose=2
    )

    # 3. 训练模型
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 4. 模型评估
    y_pred = best_model.predict(X_test)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # 5. 计算最近N场的准确率
    recent_30_accuracy = get_recent_n_accuracy(best_model, X_test, y_test, 30)
    recent_150_accuracy = get_recent_n_accuracy(best_model, X_test, y_test, 150)

    # 6. 计算综合性能指标
    composite_score = calculate_composite_score({
        'best_score': grid_search.best_score_,
        'test_balanced_accuracy': test_balanced_accuracy,
        'recent_30_accuracy': recent_30_accuracy,
        'recent_150_accuracy': recent_150_accuracy
    })

    # 7. 计算模型权重
    weight = max(0.5, composite_score * 2)  # 确保权重至少为0.5

    # 8. 打印模型评估结果
    print(f"\n{model_name} 模型的最佳参数组合：")
    print(grid_search.best_params_)
    print(f"\n{model_name} 模型的测试集表现：")
    print(f"平衡准确率: {test_balanced_accuracy:.2%}")
    print(f"综合评分: {composite_score:.2%}")
    print(f"分配权重: {weight:.2f}")

    target_names = [str(c) for c in np.unique(y_train)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 9. 分析特征重要性
    try:
        analyze_feature_importance(best_model, X_train, model_name, feature_names)
    except Exception as e:
        print(f"分析特征重要性时出错: {str(e)}")

    print(f"\n{model_name}模型最近30场平衡准确率: {recent_30_accuracy:.2%}")
    print(f"\n{model_name}模型最近150场平衡准确率: {recent_150_accuracy:.2%}")

    # 10. 返回评估结果
    model_result = {
        'best_estimator': best_model,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'test_balanced_accuracy': test_balanced_accuracy,
        'recent_30_accuracy': recent_30_accuracy,
        'recent_150_accuracy': recent_150_accuracy,
        'composite_score': composite_score,
        'selected_features': feature_names,
        'weight': weight
    }

    return model_result, (model_name, best_model), X_test


def _create_ensemble_model(ensemble_type, estimators, weights, X_train, y_train, X_test, y_test):
    """
    创建和评估集成模型

    Args:
        ensemble_type: 集成类型，'voting'或'stacking'
        estimators: 基础模型列表
        weights: 模型权重列表
        X_train: 训练数据
        y_train: 训练标签
        X_test: 测试数据
        y_test: 测试标签

    Returns:
        集成模型评估结果字典
    """
    if ensemble_type == 'voting':
        print("\n创建优化的投票集成模型...")
        print(f"使用的模型权重: {weights}")

        # 创建投票集成模型
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # 使用软投票，考虑预测概率
            weights=weights  # 使用基于性能的权重
        )
    else:  # stacking
        print("\n创建堆叠集成模型...")
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression

        # 使用逻辑回归作为元分类器
        meta_classifier = LogisticRegression(max_iter=1000, class_weight='balanced')

        # 创建堆叠集成模型
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=3,  # 使用3折交叉验证
            stack_method='predict_proba',  # 使用概率预测
            passthrough=False  # 不传递原始特征
        )

    # 训练集成模型
    print(f"\n训练{ensemble_type}集成模型...")
    ensemble.fit(X_train, y_train)

    # 评估集成模型
    y_pred = ensemble.predict(X_test)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # 计算最近N场的准确率
    recent_30_accuracy = get_recent_n_accuracy(ensemble, X_test, y_test, 30)
    recent_150_accuracy = get_recent_n_accuracy(ensemble, X_test, y_test, 150)

    # 计算综合评分
    composite_score = calculate_composite_score({
        'best_score': test_balanced_accuracy,  # 使用测试集准确率作为交叉验证得分
        'test_balanced_accuracy': test_balanced_accuracy,
        'recent_30_accuracy': recent_30_accuracy,
        'recent_150_accuracy': recent_150_accuracy
    })

    # 打印评估结果
    print(f"\n{ensemble_type}集成模型的测试集表现：")
    print(f"平衡准确率: {test_balanced_accuracy:.2%}")
    print(f"综合评分: {composite_score:.2%}")

    target_names = [str(c) for c in np.unique(y_test)]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"\n{ensemble_type}集成模型最近30场平衡准确率: {recent_30_accuracy:.2%}")
    print(f"\n{ensemble_type}集成模型最近150场平衡准确率: {recent_150_accuracy:.2%}")

    # 返回评估结果
    ensemble_result = {
        'best_estimator': ensemble,
        'best_params': {'weights': weights} if ensemble_type == 'voting' else None,
        'best_score': test_balanced_accuracy,
        'test_balanced_accuracy': test_balanced_accuracy,
        'recent_30_accuracy': recent_30_accuracy,
        'recent_150_accuracy': recent_150_accuracy,
        'composite_score': composite_score
    }

    return ensemble_result


def train_and_evaluate_models(X_train, y_train, X_test, y_test, param_grids, models, feature_names=None):
    """
    训练和评估多个模型，包括基础模型和集成模型

    Args:
        X_train: 训练数据
        y_train: 训练标签
        X_test: 测试数据
        y_test: 测试标签
        param_grids: 参数网格字典
        models: 模型字典
        feature_names: 特征名称列表，默认为None

    Returns:
        所有模型的评估结果字典
    """
    # 初始化结果存储
    best_models = {}
    estimators = []  # 用于存储所有训练好的模型
    model_weights = []  # 用于存储模型权重

    # 转换数据类型为float32以减少内存使用
    X_train_32 = X_train.astype(np.float32)
    X_test_32 = X_test.astype(np.float32)

    # 第一阶段：训练和评估基础模型
    for model_name, model in models.items():
        # 训练和评估单个模型
        model_result, estimator, _ = _train_and_evaluate_base_model(
            model, model_name, param_grids[model_name],
            X_train_32, y_train, X_test_32, y_test, feature_names
        )

        # 存储模型结果
        best_models[model_name] = model_result
        estimators.append(estimator)
        model_weights.append(model_result['weight'])

    # 第二阶段：创建和评估投票集成模型
    voting_result = _create_ensemble_model(
        'voting', estimators, model_weights,
        X_train_32, y_train, X_test_32, y_test
    )
    best_models['Voting'] = voting_result

    # 第三阶段：创建和评估堆叠集成模型
    stacking_result = _create_ensemble_model(
        'stacking', estimators, model_weights,
        X_train_32, y_train, X_test_32, y_test
    )
    best_models['Stacking'] = stacking_result

    # 将每个模型选择的特征写入CSV文件
    save_selected_features_to_csv(best_models)

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


# 新增函数：计算综合评分
def calculate_composite_score(model_metrics, weights=None):
    """
    根据多个评估指标计算综合评分
    :param model_metrics: 包含各项评估指标的字典
    :param weights: 各指标的权重字典，如果为None则使用默认权重
    :return: 综合评分
    """
    # 默认权重配置 - 优化后的权重分配
    default_weights = {
        'best_score': 0.15,           # 交叉验证得分权重（降低权重）
        'test_balanced_accuracy': 0.20, # 测试集平衡准确率权重
        'recent_30_accuracy': 0.40,     # 最近30场准确率权重（大幅提高权重，更看重近期表现）
        'recent_150_accuracy': 0.25      # 最近150场准确率权重（略微提高权重）
    }

    # 使用提供的权重或默认权重
    weights = weights or default_weights

    # 计算加权得分
    composite_score = 0.0
    total_weight = 0.0

    # 只考虑存在的指标
    for metric, weight in weights.items():
        if metric in model_metrics and model_metrics[metric] is not None:
            composite_score += model_metrics[metric] * weight
            total_weight += weight

    # 归一化得分（确保权重总和为1）
    if total_weight > 0:
        composite_score = composite_score / total_weight

    return composite_score


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
            # 检查是否有选定的特征列表
            selected_features = model_info.get('selected_features', feature_names)

            # 处理VotingClassifier和StackingClassifier
            if hasattr(model, 'estimators_') and not hasattr(model, 'feature_importances_'):
                # 对于集成模型，使用第一个基础模型的特征重要性
                if len(model.estimators_) > 0 and hasattr(model.estimators_[0], 'feature_importances_'):
                    base_model = model.estimators_[0]
                    importances = base_model.feature_importances_
                else:
                    print(f"{model_name} 模型没有可用的特征重要性")
                    continue
            else:
                importances = model.feature_importances_

            # 检查特征重要性和特征名称的长度是否匹配
            if len(importances) != len(selected_features):
                print(f"\n警告：{model_name} 模型的特征重要性长度({len(importances)})与特征名称长度({len(selected_features)})不匹配")
                # 如果是集成模型，使用原始特征名称
                if model_name in ['Voting', 'Stacking']:
                    print(f"使用原始特征名称列表")
                    # 使用前 N 个特征，其中 N 是特征重要性的长度
                    selected_features = feature_names[:len(importances)]
                else:
                    # 对于其他模型，使用自动生成的特征名称
                    print(f"使用自动生成的特征名称")
                    selected_features = [f'feature_{i}' for i in range(len(importances))]

            # 创建特征重要性的Series并可视化
            plt.figure(figsize=(10, 6))
            importance_series = pd.Series(importances, index=selected_features)
            importance_series.nlargest(15).plot(kind='barh')
            plt.title(f'{model_name} Top 15 Feature Importances')
            plt.tight_layout()
            plt.show()


# 主程序
def save_selected_features_to_csv(best_models):
    """
    将每个模型选择的特征写入CSV文件

    Args:
        best_models: 包含所有模型结果的字典
    """
    import pandas as pd

    # 只处理树模型（XGBoost、LightGBM、RandomForest）
    tree_models = ['XGBoost', 'LightGBM', 'RandomForest']

    # 创建一个字典，用于存储每个模型的特征选择情况
    model_features = {}
    all_data = []
    # 声明一个map
    map = {

    }
    # 收集每个模型的选定特征
    for model_name in tree_models:
        if model_name in best_models and 'selected_features' in best_models[model_name]:
            selected_features = best_models[model_name]['selected_features']
            model_features[model_name] = selected_features
            print(f"\n{model_name} 选择了 {len(selected_features)} 个特征")
            # selected_features 转成一大个字符串
            # 确保所有元素都是字符串类型
            features_str = ','.join([str(feature) for feature in selected_features])
            map[model_name] = features_str

    #map 转成csv 存到本地
    # 将字典转换为 DataFrame
    if map:
        # 创建一个列表来存储数据
        data = []
        for model_name, features_str in map.items():
            data.append({'model_name': model_name, 'selected_features': features_str})

        # 使用列表创建DataFrame
        df = pd.DataFrame(data)

        # 将 DataFrame 保存为 CSV 文件
        df.to_csv('model_selected_features.csv', index=False)
        print(f"\n已将模型特征映射(map)保存到 model_selected_features.csv")
    else:
        print("\n没有模型特征映射数据要保存")

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
