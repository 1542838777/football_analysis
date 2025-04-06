from collections import Counter

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class StrictTimeAggregator(BaseEstimator, TransformerMixin):
    """时间感知特征聚合器（最终修复版）"""

    def __init__(self):
        self.key_bookmakers = [1000, 57, 25, 11]
        self.static_features = ['league_id']
        self.time_cutoff_ = None
        self.match_ids_ = None

    def fit(self, X, y=None):
        if 'bet_time' in X.columns:
            self.time_cutoff_ = X['bet_time'].max()
        self.match_ids_ = X['match_id'].unique()
        return self

    def transform(self, df):
        # 确保包含所有原始match_id（关键修复）
        all_match_ids = pd.DataFrame({'match_id': self.match_ids_})

        # 过滤未来数据并复制
        df = df.copy()
        if self.time_cutoff_ is not None and 'bet_time' in df.columns:
            df = df[df['bet_time'] <= self.time_cutoff_]

        # --- 特征工程 ---
        # 时间衰减权重
        df['time_weight'] = np.exp(-0.1 * (df['last_update_time_distance'] / 3600))

        # --- 基础聚合 ---
        base_agg = {
            'first_win_sp': ['mean', 'std', 'skew','max', 'min', lambda x: x.max() - x.min()],
            'first_draw_sp': ['mean', 'std', 'skew','max', 'min', lambda x: x.quantile(0.8)],
            'first_lose_sp': ['mean', 'std', 'skew','max', 'min'],
            'first_win_kelly_index': ['mean', 'std', 'max', 'min'],
            'first_draw_kelly_index': ['mean', 'std', 'max', 'min'],
            'first_lose_kelly_index': ['mean', 'std', 'max', 'min'],
            'bookmaker_id': ['nunique'],
            'last_update_time_distance': ['max']
        }

        # 添加机构特征
        for bid in self.key_bookmakers:
            for outcome in ['win', 'draw', 'lose']:
                col_name = f'bid_{bid}_{outcome}'
                df[col_name] = np.where(df['bookmaker_id'] == bid, df[f'first_{outcome}_sp'], np.nan)
                base_agg[col_name] = lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan

        # 执行聚合并保留所有match_id
        match_df = df.groupby('match_id').agg(base_agg)
        match_df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in match_df.columns]
        match_df = match_df.reset_index()  # 关键修复：确保match_id转为列

        # --- 市场共识特征 ---
        consensus_features = df.groupby('match_id').agg({
            'max_first_win_sp': 'max',
            'min_first_win_sp': 'min',
            'max_first_draw_sp': 'max',
            'min_first_draw_sp': 'min',
            'max_first_lose_sp': 'max',
            'min_first_lose_sp': 'min'
        }).reset_index()  # 重置索引确保有match_id列

        # --- 合并所有特征 ---
        # 使用outer合并确保不丢失任何match_id
        merged = pd.merge(all_match_ids, match_df, on='match_id', how='left')
        merged = pd.merge(merged, consensus_features, on='match_id', how='left')

        # --- 凯利警报特征 ---
        kelly_alerts = df.groupby('match_id').apply(
            lambda g: pd.Series({
                'win_kelly_alert': ((g['first_win_kelly_index'] >= 1.05) | (g['first_win_kelly_index'] <= 0.9)).sum(),
                'draw_kelly_alert': (
                            (g['first_draw_kelly_index'] >= 1.05) | (g['first_draw_kelly_index'] <= 0.9)).sum(),
                'lose_kelly_alert': ((g['first_lose_kelly_index'] >= 1.05) | (g['first_lose_kelly_index'] <= 0.9)).sum()
            })
        ).reset_index()  # 关键修复：确保有match_id列

        merged = pd.merge(merged, kelly_alerts, on='match_id', how='left')

        # --- 机构差异特征 ---
        def get_agency_diff(g):
            res = {}
            for a1, a2 in [(1000, 57), (11, 57)]:
                odds1 = g[g['bookmaker_id'] == a1][['first_win_sp', 'first_draw_sp', 'first_lose_sp']].mean()
                odds2 = g[g['bookmaker_id'] == a2][['first_win_sp', 'first_draw_sp', 'first_lose_sp']].mean()
                res.update({
                    f'diff_{a1}_{a2}_win': (odds1[0] - odds2[0]) if not np.isnan(odds1[0]) else np.nan,
                    f'diff_{a1}_{a2}_draw': (odds1[1] - odds2[1]) if not np.isnan(odds1[1]) else np.nan,
                    f'diff_{a1}_{a2}_lose': (odds1[2] - odds2[2]) if not np.isnan(odds1[2]) else np.nan
                })
            return pd.Series(res)

        agency_diffs = df.groupby('match_id').apply(get_agency_diff).reset_index()
        merged = pd.merge(merged, agency_diffs, on='match_id', how='left')

        # --- 静态特征 ---
        static_data = df.groupby('match_id')[self.static_features].first().reset_index()
        merged = pd.merge(merged, static_data, on='match_id', how='left')

        # --- 缺失值处理 ---
        num_cols = merged.select_dtypes(include=np.number).columns
        merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median())

        return merged


class MatchLevelOddsAnalyzer:
    def __init__(self):
        self.feature_processor = None
        self.model = None
        self.class_distribution_ = None

    def _create_features(self, df):
        # 特征增强
        df['win_sp_range'] = df['first_win_sp_max'] - df['first_win_sp_min']
        df['draw_sp_range'] = df['first_draw_sp_max'] - df['first_draw_sp_min']
        df['lose_sp_range'] = df['first_lose_sp_max'] - df['first_lose_sp_min']

        df['total_kelly_alert'] = df[['win_kelly_alert', 'draw_kelly_alert', 'lose_kelly_alert']].sum(axis=1)

        # 凯利分歧指数
        for outcome in ['win', 'draw', 'lose']:
            mean_col = f'first_{outcome}_kelly_index_mean'
            max_col = f'first_{outcome}_kelly_index_max'
            min_col = f'first_{outcome}_kelly_index_min'
            df[f'{outcome}_kelly_divergence'] = (df[max_col] - df[min_col]) / (df[mean_col] + 1e-6)

        return df

    def _build_pipeline(self):
        return Pipeline([
            ('aggregator', StrictTimeAggregator()),
            ('feature_engineer', FunctionTransformer(self._create_features)),
            ('processor', ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), make_column_selector(
                    pattern=r'^(.*)(kdiv|kratio|first|win|draw|lose|bookmaker_id|last_update_time_distance|league_id|rate|diff|alert|range)(.*)$',
                    dtype_include=np.number
                ))
            ], remainder='drop'))
        ])

    def _get_labels(self, df):
        label_map = df.groupby('match_id')['nwdl_result'].first().map({'0': 0, '1': 1, '3': 2})
        return label_map.dropna()

    def _dynamic_sampling_strategy(self, y):
        """动态生成采样策略"""
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)

        # 自动设置采样策略：将少数类提升到与多数类相同数量
        strategy = {
            cls: max(count, class_counts[majority_class])
            for cls, count in class_counts.items()
        }
        print(f"动态采样策略: {strategy}")
        return strategy

    def train_model(self, df):
        # 时间序列分割
        df = df.sort_values(['bet_time','match_id'])
        unique_match_ids = df['match_id'].unique()
        split_idx = int(len(unique_match_ids) * 0.8)

        train_ids = unique_match_ids[:split_idx]
        test_ids = unique_match_ids[split_idx:]

        train_data = df[df['match_id'].isin(train_ids)]
        test_data = df[df['match_id'].isin(test_ids)]

        # 构建处理管道
        self.feature_processor = self._build_pipeline()
        X_train = self.feature_processor.fit_transform(train_data)
        y_train = self._get_labels(train_data)

        # 记录类别分布
        self.class_distribution_ = Counter(y_train)
        print("原始类别分布:", self.class_distribution_)

        # 动态调整采样策略
        sampling_strategy = self._dynamic_sampling_strategy(y_train)

        # 模型配置
        self.model = ImbPipeline([
            ('smote', SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=5,
                random_state=42
            )),
            ('calibrated', CalibratedClassifierCV(
                LGBMClassifier(
                    n_estimators=1200,
                    learning_rate=0.02,
                    max_depth=4,
                    class_weight='balanced'
                ),
                cv=TimeSeriesSplit(4),
                method='isotonic'
            ))
        ])

        # 训练模型
        self.model.fit(X_train, y_train)
        print("过采样后类别分布:", Counter(self.model.named_steps['smote'].fit_resample(X_train, y_train)[1]))

        # 测试验证
        X_test = self.feature_processor.transform(test_data)
        y_test = self._get_labels(test_data)

        print("\n测试集表现:")
        print(classification_report(y_test, self.model.predict(X_test)))

        joblib.dump(self.feature_processor, 'feature_processor_v2.pkl')

    def evaluate_performance(self, df, n=300):
        processed = self.feature_processor.transform(df)
        recent_data = processed[-n:]

        labels = df.groupby('match_id')['nwdl_result'].first().map({'0': 0, '1': 1, '3': 2})
        y_true = labels.iloc[-n:]

        y_pred = self.model.predict(recent_data)
        probs = self.model.predict_proba(recent_data)

        print("\n近期表现评估:")
        print(classification_report(y_true, y_pred, target_names=['输', '平', '赢']))

        plt.figure(figsize=(10, 6))
        pd.DataFrame(probs, columns=['输', '平', '赢']).plot.kde()
        plt.title('预测置信度分布')
        plt.savefig('confidence_distribution.png')
        plt.close()


# 使用示例
if __name__ == "__main__":
    from service.spf.initData.data.mysql_data import load_europe_odds_not_handicap_data

    analyzer = MatchLevelOddsAnalyzer()
    raw_data = load_europe_odds_not_handicap_data()

    # 数据预处理
    raw_data = raw_data.sort_values(['bet_time','match_id']).dropna()
    raw_data['nwdl_result'] = raw_data['nwdl_result'].astype(str)

    # 训练和评估
    analyzer.train_model(raw_data)
    analyzer.evaluate_performance(raw_data)

