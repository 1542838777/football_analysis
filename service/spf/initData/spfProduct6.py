import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report
from collections import Counter

from service.spf.initData.data.mysql_data import load_europe_odds_not_handicap_data


class MatchFeatureGenerator(BaseEstimator, TransformerMixin):
    """统一比赛维度特征生成器"""

    def __init__(self):
        self.key_bookmakers = [1000, 57, 25, 11]
        self.match_ids_ = None

    def fit(self, X, y=None):
        self.match_ids_ = X['match_id'].unique()
        return self

    def _process_single_match(self, group):
        """处理单个比赛的所有赔率数据，返回一行特征"""
        match_id = group.name
        features = {'match_id': match_id}

        # 基础统计特征
        for outcome in ['win', 'draw', 'lose']:
            # 赔率统计
            sp_series = group[f'first_{outcome}_sp']
            features.update({
                f'{outcome}_sp_mean': sp_series.mean(),
                f'{outcome}_sp_std': sp_series.std(),
                f'{outcome}_sp_max': sp_series.max(),
                f'{outcome}_sp_min': sp_series.min(),
                f'{outcome}_sp_range': sp_series.max() - sp_series.min()
            })

            # 凯利指数统计
            kelly_series = group[f'first_{outcome}_kelly_index']
            features.update({
                f'{outcome}_kelly_mean': kelly_series.mean(),
                f'{outcome}_kelly_std': kelly_series.std(),
                f'{outcome}_kelly_max': kelly_series.max(),
                f'{outcome}_kelly_min': kelly_series.min()
            })

        # 重点机构特征
        for bid in self.key_bookmakers:
            agency_data = group[group['bookmaker_id'] == bid]
            for outcome in ['win', 'draw', 'lose']:
                key = f'bid_{bid}_{outcome}'
                features[key] = agency_data[f'first_{outcome}_sp'].iloc[0] if not agency_data.empty else np.nan

        # 时间特征
        features['update_count'] = len(group)
        features['last_update_gap'] = group['last_update_time_distance'].max()
        features['league_id'] = group['league_id'].max()
        features['nwdl_result'] = group['nwdl_result'].max()


        # 市场共识特征
        features.update({
            'max_win_sp': group['max_first_win_sp'].max(),
            'min_win_sp': group['min_first_win_sp'].min(),
            'max_draw_sp': group['max_first_draw_sp'].max(),
            'min_draw_sp': group['min_first_draw_sp'].min()
        })

        # 机构差异特征
        for a1, a2 in [(1000, 57), (11, 57)]:
            odds1 = group[group['bookmaker_id'] == a1]
            odds2 = group[group['bookmaker_id'] == a2]
            for outcome in ['win', 'draw', 'lose']:
                key = f'diff_{a1}_{a2}_{outcome}'
                val = (odds1[f'first_{outcome}_sp'].mean() -
                       odds2[f'first_{outcome}_sp'].mean()) if not odds1.empty and not odds2.empty else np.nan
                features[key] = val

        return pd.Series(features)

    def transform(self, X):
        # 统一分组处理
        features_df = X.groupby('match_id', group_keys=False).apply(self._process_single_match)

        # 确保保留所有原始match_id
        full_df = pd.DataFrame({'match_id': self.match_ids_})
        return full_df.merge(features_df, on='match_id', how='left')


class MatchModelPipeline:
    """统一比赛维度的模型管道"""

    def __init__(self):
        self.pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LGBMClassifier(class_weight='balanced'))
        ])

    def train(self, X, y):
        # 获取标签 按 match_id 分组 取第一个， 并且 映射 nwdl_result {'0': 0, '1': 1, '3': 2}

        nwdl_mapping = {'0': 0, '1': 1, '3': 2}
        y['nwdl_result'] = y['nwdl_result'].map(nwdl_mapping)

        # 对齐数据
        # common_ids = np.intersect1d(self.pipeline['features'].match_ids_, y.index)
        # X_filtered = X[X['match_id'].isin(common_ids)]
        # y_filtered = y.loc[common_ids]

        self.pipeline.fit(X, y)

    def evaluate(self, X, y):
        nwdl_mapping = {'0': 0, '1': 1, '3': 2}
        y['nwdl_result']  = y['nwdl_result'].map(nwdl_mapping)


        print(classification_report(y, self.pipeline['model'].predict(X)))


# 使用示例
if __name__ == "__main__":
    # 数据加载
    raw_data = load_europe_odds_not_handicap_data()
    raw_data = raw_data.sort_values(['bet_time', 'match_id'])
    pipeline = MatchFeatureGenerator()

    match_level_df =raw_data.groupby('match_id', group_keys=False).apply(pipeline._process_single_match)

    # match_level_df划分训练测试集 0.8 训练 ，0.2测试
    match_level_df = match_level_df.dropna()


    split_idx = int(len(match_level_df) * 0.8)
    train_matches = match_level_df[:split_idx]
    test_matches = match_level_df[split_idx:]


    # 初始化管道
    pipeline = MatchModelPipeline()

    # 训练模型
    pipeline.train(train_matches, train_matches[['nwdl_result']])

    # 评估模型
    pipeline.evaluate(train_matches, train_matches[['nwdl_result']])

    # 保存模型
    joblib.dump(pipeline, 'match_model_pipeline.pkl')