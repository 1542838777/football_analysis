import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

from service.spf.initData.data.mysql_data import load_europe_odds_not_handicap_data

class MatchAggregator(BaseEstimator, TransformerMixin):
    """比赛维度特征聚合器"""

    def __init__(self):
        self.key_bookmakers = [1000, 57, 11]  # 重点监控机构
        self.static_features = ['league_id']  # 静态特征
        self.agency_features = []  # 记录生成的机构特征

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # 按比赛时间和ID排序
        df = df.sort_values(['bet_time', 'match_id'])

        # 基础聚合规则
        agg_rules = {
            'first_win_sp': ['mean', 'std', 'skew', lambda x: x.max() - x.min()],
            'first_draw_sp': ['mean', 'std', lambda x: x.quantile(0.8)],
            'first_lose_sp': ['mean', 'std'],
            'first_win_kelly_index': ['mean', 'std', 'max', 'min'],
            'bookmaker_id': ['nunique'],
            'last_update_time_distance': ['max']
        }

        # 添加重点机构特征（关键修复点）
        for bid in self.key_bookmakers:
            # 生成机构特征列
            df[f'bid_{bid}_win'] = df.where(df['bookmaker_id'] == bid)['first_win_sp']
            df[f'bid_{bid}_draw'] = df.where(df['bookmaker_id'] == bid)['first_draw_sp']

            # 添加聚合规则
            agg_rules[f'bid_{bid}_win'] = 'first'
            agg_rules[f'bid_{bid}_draw'] = 'first'
            self.agency_features.extend([f'bid_{bid}_win', f'bid_{bid}_draw'])

        # 执行聚合
        match_df = df.groupby('match_id').agg(agg_rules)

        # 扁平化列名
        new_columns = []
        for col in agg_rules:
            if isinstance(agg_rules[col], list):
                for stat in agg_rules[col]:
                    if callable(stat):
                        new_columns.append(f"{col}_range")
                    else:
                        new_columns.append(f"{col}_{stat}")
            else:
                new_columns.append(col)
        match_df.columns = new_columns

        # 添加静态特征
        static_data = df.groupby('match_id')[self.static_features].first()
        return pd.concat([match_df, static_data], axis=1).reset_index()


class MatchLevelOddsAnalyzer:
    def __init__(self):
        self.feature_processor = None
        self.model = None
        self.feature_importance = None

    def _create_features(self, df):
        """比赛维度特征工程（增强版）"""
        # 时间衰减因子
        df['time_decay'] = 1 / (1 + np.exp(-df['last_update_time_distance_max'] / 1440))

        # 凯利分歧指数（带安全计算）
        df['kelly_divergence'] = np.where(
            df['first_win_kelly_index_mean'] > 0,
            (df['first_win_kelly_index_max'] - df['first_win_kelly_index_min']) /
            (df['first_win_kelly_index_mean'] + 1e-6),
            0
        )

        # 重点机构差异（关键修复点）
        df['key_agency_spread'] = df['bid_1000_win'] - df['bid_57_win']

        # 平局特征增强
        df['draw_odds_volatility'] = df['first_draw_sp_std'] / df['first_draw_sp_mean']
        df['draw_kelly_ratio'] = (df['first_win_kelly_index_mean'] + 1e-6) / \
                                 (df['first_draw_sp_mean'] + 1e-6)

        # 市场平衡指标
        df['market_balance'] = df['first_win_sp_mean'] * df['first_lose_sp_mean']
        # 风险平衡指标
        df['risk_balance'] = df['first_win_kelly_index_std'] / (df['first_draw_sp_std'] + 1e-6)

        return df

    def _build_pipeline(self):
        """特征处理管道（优化版）"""
        numeric_features = [
            'first_win_sp_mean', 'first_win_sp_std', 'first_win_sp_skew',
            'first_draw_sp_mean', 'first_draw_sp_std', 'draw_odds_volatility',
            'first_win_kelly_index_mean', 'first_win_kelly_index_std',
            'kelly_divergence', 'key_agency_spread', 'draw_kelly_ratio',
            'market_balance', 'time_decay'
        ]

        return make_pipeline(
            MatchAggregator(),
            FunctionTransformer(self._create_features),
            ColumnTransformer([
                ('num', StandardScaler(), numeric_features),
                ('passthrough', 'passthrough', ['bookmaker_id_nunique'])
            ])
        )

    def train_model(self, df):
        """模型训练流程（改进版）"""
        # 数据预处理
        self.feature_processor = self._build_pipeline()
        processed = self.feature_processor.fit_transform(df)

        # 获取标签并验证
        y = df.groupby('match_id')['nwdl_result'].first()
        y = y.map({'0': 0, '1': 1, '3': 2}).values  # 确保标签映射正确

        # 检查标签分布
        print("\n训练数据分布：")
        print(pd.Series(y).value_counts())

        # 处理类别不平衡
        sm = SMOTE(sampling_strategy={1: int(len(y) * 0.3)}, random_state=42)
        X_res, y_res = sm.fit_resample(processed, y)

        # 动态类别权重
        class_weights = {0: 1.0, 1: 3.0, 2: 1.0}  # 提高平局权重

        # 初始化模型（参数优化）
        base_model = LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.02,
            max_depth=4,
            class_weight=class_weights,
            reg_alpha=0.2,
            reg_lambda=0.2,
            min_child_samples=50,
            importance_type='gain'
        )

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=4)
        self.model = CalibratedClassifierCV(base_model, cv=tscv, method='isotonic')

        # 训练
        self.model.fit(X_res, y_res)

        # 评估
        y_pred = self.model.predict(processed)
        print("\n训练集表现：")
        print(classification_report(y, y_pred, target_names=['Lose', 'Draw', 'Win']))

        # 保存模型
        joblib.dump(self.model, 'match_level_model_v2.pkl')
        joblib.dump(self.feature_processor, 'feature_pipeline_v2.pkl')

    def evaluate_performance(self, df, n=100):
        """近期表现评估"""
        processed = self.feature_processor.transform(df)
        recent_data = processed[-n:]
        # df 按matchId分组，取nwdl_result ,并且对其数据 进行映射 map({'0': 0, '1': 1, '3': 2})
        sorted_df = df.sort_values(['bet_time', 'match_id'])
        grouped = sorted_df.groupby('match_id',sort=False)
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
    # 初始化
    analyzer = MatchLevelOddsAnalyzer()

    # 加载数据（假设数据包含match_id, bookmaker_id, 赔率特征，比赛结果等）
    raw_data = load_europe_odds_not_handicap_data();
    raw_data = raw_data.sort_values(['bet_time', 'match_id'])

    # 训练模型
    analyzer.train_model(raw_data)

    # # 预测新比赛
    # new_matches = pd.read_csv('new_odds.csv')
    # predictions = analyzer.predict_new_matches(new_matches)
    # print(predictions.head())

    # 评估近期表现
    analyzer.evaluate_performance(raw_data, n=800)
