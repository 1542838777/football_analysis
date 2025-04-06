import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE  # 新增
from imblearn.pipeline import make_pipeline as make_imb_pipeline

from service.spf.initData.data.mysql_data import load_europe_odds_not_handicap_data


class MatchAggregator(BaseEstimator, TransformerMixin):
    """比赛维度特征聚合器（修复版）"""

    def __init__(self):
        self.key_bookmakers = [1000, 57, 25, 11]  # 重点监控机构
        self.static_features = ['league_id']  # 静态特征
        self.agency_features = []  # 记录生成的机构特征

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # 初始化聚合规则和动态列名记录器
        agg_rules = {}
        dynamic_columns = []

        # 基础聚合规则
        base_features = {
            'first_win_sp': ['mean', 'std', 'skew', lambda x: x.max() - x.min()],# todo min max
            'first_draw_sp': ['mean', 'std', 'skew',lambda x: x.quantile(0.8)],
            'first_lose_sp': ['mean', 'std','skew'],
            'first_win_kelly_index': ['mean', 'std', 'max', 'min'],
            'first_draw_kelly_index': ['mean', 'std', 'max', 'min'],
            'first_lose_kelly_index': ['mean', 'std', 'max', 'min'],
            'bookmaker_id': ['nunique'],
            'last_update_time_distance': ['max']
        }

        # 构建基础特征列名
        for col, methods in base_features.items():
            agg_rules[col] = methods
            for m in methods:
                suffix = m.__name__ if callable(m) else m
                dynamic_columns.append(f"{col}_{suffix}")

        # 动态生成机构特征列名
        agency_columns = []
        for bid in self.key_bookmakers:
            for outcome in ['win', 'draw','lose']:
                col_name = f'bid_{bid}_{outcome}'
                agg_rules[col_name] = lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan
                agency_columns.append(col_name)

        # 添加重点机构特征（含缺失处理）
        for bid in self.key_bookmakers:
            df[f'bid_{bid}_win'] = np.where(df['bookmaker_id'] == bid, df['first_win_sp'], np.nan)
            df[f'bid_{bid}_draw'] = np.where(df['bookmaker_id'] == bid, df['first_draw_sp'], np.nan)
            df[f'bid_{bid}_lose'] = np.where(df['bookmaker_id'] == bid, df['first_lose_sp'], np.nan)

            agg_rules.update({
                f'bid_{bid}_win': lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan,
                f'bid_{bid}_draw': lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan,
                f'bid_{bid}_lose': lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan
            })
            self.agency_features.extend([f'bid_{bid}_win', f'bid_{bid}_draw', f'bid_{bid}_lose'])


        # 原有市场共识特征
        grp = df.groupby('match_id')
        df['max_win_agency'] = grp['max_first_win_sp'].transform('sum')
        df['min_win_agency'] = grp['min_first_win_sp'].transform('sum')
        df['max_draw_agency'] = grp['max_first_draw_sp'].transform('sum')
        df['min_draw_agency'] = grp['min_first_draw_sp'].transform('sum')
        df['max_lose_agency'] = grp['max_first_lose_sp'].transform('sum')
        df['min_lose_agency'] = grp['min_first_lose_sp'].transform('sum')

        # 按matchId分组 统计first_win_kelly_index大于1.05的机构数
        df['win_more_one_kelly_alert_count'] = grp['first_win_kelly_index'].transform(
            lambda x: (x >= 1.05).sum())
        df['draw_more_one_kelly_alert_count'] = grp['first_draw_kelly_index'].transform(
            lambda x: (x >= 1.05).sum())
        df['lose_more_one_kelly_alert_count'] = grp['first_lose_kelly_index'].transform(
            lambda x: (x >= 1.05).sum())

        # 按matchId分组 统计first_win_kelly_index小于0.9的机构数
        df['win_less_one_kelly_alert_count'] = grp['first_win_kelly_index'].transform(
            lambda x: (x <= 0.9).sum())
        df['draw_less_one_kelly_alert_count'] = grp['first_draw_kelly_index'].transform(
            lambda x: (x <= 0.9).sum())

        df['lose_less_one_kelly_alert_count'] = grp['first_lose_kelly_index'].transform(
            lambda x: (x <= 0.9).sum())

        # 原有特殊机构对比特征
        def calculate_odds_difference(group):
            features = {}
            comparisons = [(1000, 57, '1000_57'), (11, 57, '11_57')]
            for agency1, agency2, suffix in comparisons:
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
                        f'odds_win_diff_{suffix}': np.nan,
                        f'odds_draw_diff_{suffix}': np.nan,
                        f'odds_lose_diff_{suffix}': np.nan
                    })
            return pd.Series(features)

        odds_diff_features = df.groupby('match_id').apply(calculate_odds_difference)
        # 执行聚合
        match_df = df.groupby('match_id').agg(agg_rules)
        # 统一列名处理（修复关键点）
        final_columns = []
        for col in match_df.columns:
            if isinstance(col, tuple):
                # 处理 (feature, method) 格式列名
                base_name = col[0]
                method_name = col[1].__name__ if callable(col[1]) else col[1]
                # method_name =='<lambda>'
                if method_name == '<lambda>':
                    final_columns.append(f"{base_name}")
                else:
                    final_columns.append(f"{base_name}_{method_name}")

            else:
                final_columns.append(str(col))

        match_df.columns = final_columns
        match_df = match_df.merge(odds_diff_features, how='left', on='match_id')
        self.agency_features = agency_columns  # 更新正确的机构特征列名

        # 添加静态特征
        static_data = df.groupby('match_id')[self.static_features].first()
        processed = pd.concat([match_df, static_data], axis=1).reset_index()

        # 填充机构特征缺失值（使用正确的列名）
        processed[self.agency_features] = processed[self.agency_features].fillna(
            processed[self.agency_features].median()
        )

        return processed

class MatchLevelOddsAnalyzer:
    def __init__(self):
        self.feature_processor = None
        self.model = None
        self.feature_importance = None

    def _create_features(self, df):
        """安全特征工程（含NaN处理）"""
        # 凯利分歧指数（安全计算）
        win_kelly_mean = df['first_win_kelly_index_mean'] + 1e-6
        draw_kelly_mean = df['first_draw_kelly_index_mean'] + 1e-6
        lose_kelly_mean = df['first_lose_kelly_index_mean'] + 1e-6
        df['win_kelly_divergence'] = (df['first_win_kelly_index_max'] - df['first_win_kelly_index_min']) / win_kelly_mean
        df['draw_kelly_divergence'] = (df['first_draw_kelly_index_max'] - df['first_draw_kelly_index_min']) / draw_kelly_mean
        df['lose_kelly_divergence'] = (df['first_lose_kelly_index_max'] - df['first_lose_kelly_index_min']) / lose_kelly_mean

         # 平局特征增强
        win_sp_mean = df['first_win_sp_mean'] + 1e-6
        draw_sp_mean = df['first_draw_sp_mean'] + 1e-6
        lose_sp_mean = df['first_lose_sp_mean'] + 1e-6
        df['win_volatility'] = df['first_win_sp_std'] / win_sp_mean
        df['draw_volatility'] = df['first_draw_sp_std'] / draw_sp_mean
        df['lose_volatility'] = df['first_lose_sp_std'] / lose_sp_mean
        df['win_kelly_ratio'] = win_kelly_mean / win_sp_mean
        df['draw_kelly_ratio'] = draw_kelly_mean / draw_sp_mean
        df['lose_kelly_ratio'] = lose_kelly_mean / lose_sp_mean

        # 填充剩余缺失值
        return df.fillna(0)
    def _build_pipeline(self):
        """动态特征选择管道"""
        return Pipeline([
            ('aggregator', MatchAggregator()),
            ('feature_engineer', FunctionTransformer(self._create_features)),
            ('processor', ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), make_column_selector(
                    pattern=r'^(.*)(kdiv|kratio|first|win|draw|lose|bookmaker_id|last_update_time_distance|league_id|rate)(.*)$',
                    dtype_include=np.number
                )),
                ('passthrough', 'passthrough', ['bookmaker_id_nunique'])
            ]))
        ])


    def train_model(self, df):
        """改进的训练流程"""
        # 构建处理管道
        self.feature_processor = self._build_pipeline()
        processed = self.feature_processor.fit_transform(df)

# df 里面 哪些列 不符合pattern=r'^(.*)(kdiv|kratio|first_|win_|draw_|lose_|bookmaker_id|last_update_time_distance|league_id|rate)(.*)$'
        # 假设你的 DataFrame 是 df
        pattern = r'^(.*)(kdiv|kratio|first|win|draw|lose|bookmaker_id|last_update_time_distance|league_id|rate)(.*)$'

        # 筛选不符合条件的列名
        non_matching_cols = df.columns[~df.columns.str.contains(pattern, regex=True)].tolist()

        print("不符合正则的列：", non_matching_cols)

        # 获取标签
        y = df.groupby('match_id')['nwdl_result'].first().map({'0': 0, '1': 1, '3': 2}).values

        # 处理类别不平衡
        sm = SMOTE(sampling_strategy={1: int(len(y) * 0.3)}, random_state=42, k_neighbors=5)
        base_model = LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.02,
            max_depth=4,
            class_weight={0: 1, 1: 3, 2: 1},
            reg_alpha=0.2,
            min_child_samples=50,
            importance_type='gain'
        )

        # 构建完整模型管道
        self.model = make_imb_pipeline(
            sm,
            CalibratedClassifierCV(base_model, cv=TimeSeriesSplit(4), method='isotonic')
        )

        # 训练模型
        self.model.fit(processed, y)

        # 评估训练集
        y_pred = self.model.predict(processed)
        print("\n训练集表现：")
        print(classification_report(y, y_pred, target_names=['Lose', 'Draw', 'Win']))

        # 保存模型
        joblib.dump(self.model, 'improved_model.pkl')
        joblib.dump(self.feature_processor, 'feature_pipeline.pkl')


    def evaluate_performance(self, df, n=100):
        """近期表现评估"""
        processed = self.feature_processor.transform(df)
        recent_data = processed[-n:]
        # df 按matchId分组，取nwdl_result ,并且对其数据 进行映射 map({'0': 0, '1': 1, '3': 2})
        sorted_df = df.sort_values(['bet_time', 'match_id'])
        grouped = sorted_df.groupby('match_id', sort=False)
        first_values = grouped['nwdl_result'].first()
        y_true = first_values.map({'0': 0, '1': 1, '3': 2})
        y_true = y_true[-n:]
        y_pred = self.model.predict(recent_data)
        probs = self.model.predict_proba(recent_data)

        # 生成评估报告
        report = classification_report(y_true, y_pred, target_names=['Lose', 'Draw', 'Win'])
        print("近期表现评估：")
        print(report)

        # 可视化置信度分布
        plt.figure(figsize=(10, 6))
        pd.DataFrame(probs, columns=['Lose', 'Draw', 'Win']).plot.kde()
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Probability')
        plt.show()

        return report


# 使用示例
if __name__ == "__main__":
    analyzer = MatchLevelOddsAnalyzer()

    # 加载数据
    raw_data = load_europe_odds_not_handicap_data()
    raw_data = raw_data.sort_values(['bet_time', 'match_id'])
    raw_data = raw_data.dropna()
    # 训练模型
    analyzer.train_model(raw_data)

    # 评估表现
    analyzer.evaluate_performance(raw_data, n=300)
