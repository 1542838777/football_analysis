import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from service.spf.initData.data.mysql_data import load_europe_odds_not_handicap_data


class EnhancedFootballOddsAnalyzer:
    def __init__(self):
        self.valid_bookmakers = None
        self.feature_processor = None
        self.model = None
        self.agency_hist = {}
        self.feature_importance = None

    def _filter_bookmakers(self, df, coverage_threshold=0.85):
        """优化后的机构筛选（保留原有逻辑）"""
        total_matches = df['match_id'].nunique()
        bookmaker_coverage = df.groupby('bookmaker_id')['match_id'].nunique() / total_matches
        self.valid_bookmakers = bookmaker_coverage[bookmaker_coverage >= coverage_threshold].index.tolist()
        return df[df['bookmaker_id'].isin(self.valid_bookmakers)]

    def xx(self, df):
        """保留所有原有特征，增加关键新特征"""
        # 原有机构历史表现特征
        if not self.agency_hist:
            self.agency_hist = df.groupby('bookmaker_id')['nwdl_result'].value_counts(normalize=True) \
                .unstack().fillna(0).to_dict('index')

        # 原有市场共识特征
        grp = df.groupby('match_id')
        df['market_win_std'] = grp['first_win_sp'].transform('std')
        df['market_draw_std'] = grp['first_draw_sp'].transform('std')
        df['market_lose_std'] = grp['first_lose_sp'].transform('std')

        df['market_win_kelly_std'] = grp['first_win_kelly_index'].transform('std')
        df['market_draw_kelly_std'] = grp['first_draw_kelly_index'].transform('std')
        df['market_lose_kelly_std'] = grp['first_lose_kelly_index'].transform('std')

        df['market_win_skew'] = grp['first_win_sp'].transform(lambda x: x.skew())
        df['market_draw_skew'] = grp['first_draw_sp'].transform(lambda x: x.skew())
        df['market_lose_skew'] = grp['first_lose_sp'].transform(lambda x: x.skew())

        # 原有极值机构特征
        df['max_win_agency']  = grp['max_first_win_sp'].transform('sum')
        df['min_win_agency']  = grp['min_first_win_sp'].transform('sum')
        df['max_draw_agency'] = grp['max_first_draw_sp'].transform('sum')
        df['min_draw_agency'] = grp['min_first_draw_sp'].transform('sum')
        df['max_lose_agency'] = grp['max_first_lose_sp'].transform('sum')
        df['min_lose_agency'] = grp['min_first_lose_sp'].transform('sum')

        # 原有时间衰减因子
        df['time_factor'] = 1 / (1 + np.exp(-df['last_update_time_distance'] / 1440))

        # 原有赔率离散度特征
        df['win_draw_division'] = grp['first_win_sp'].transform('mean') / grp['first_draw_sp'].transform('mean')
        df['draw_lose_division'] = grp['first_draw_sp'].transform('mean') / grp['first_lose_sp'].transform('mean')
        df['win_lose_division'] = grp['first_win_sp'].transform('mean') / grp['first_lose_sp'].transform('mean')

        df['win_draw_spread'] = grp['first_win_sp'].transform('mean') - grp['first_draw_sp'].transform('mean')
        df['draw_lose_spread'] = grp['first_draw_sp'].transform('mean') - grp['first_lose_sp'].transform('mean')
        df['win_lose_spread'] = grp['first_win_sp'].transform('mean') - grp['first_lose_sp'].transform('mean')
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
            comparisons = [(1000, 57, '1000_57'), (11,57, '11_57')]
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
        df = df.merge(odds_diff_features, how='left', on='match_id')

        # ========== 新增关键特征 ==========
        # 凯利指数动态变化
        df['kelly_volatility'] = df.groupby('match_id')['first_win_kelly_index'].transform('std')

        # 赔率变化动量
        df['sp_momentum'] = df.groupby('bookmaker_id')['first_win_sp'].transform(
            lambda x: x.diff(3) / x.shift(3)
        )

        # 市场分歧指数
        df['market_disagreement'] = (
                                            df['market_win_std'] +
                                            df['market_draw_std'] +
                                            df['market_lose_std']
                                    ) / 3
        return df

    def _build_pipeline(self):
        """优化后的特征处理流程（保留原有特征结构）"""
        numeric_features = [
            'first_win_sp', 'first_draw_sp', 'first_lose_sp',
            'market_win_std', 'market_draw_std', 'market_lose_std',
            'market_win_kelly_std', 'market_draw_kelly_std', 'market_lose_kelly_std',
            'market_win_skew', 'market_draw_skew', 'market_lose_skew',

            'win_draw_division', 'draw_lose_division', 'win_lose_division',
            'win_draw_spread', 'draw_lose_spread', 'win_lose_spread',
            'time_factor',
            # 新增特征

            'win_more_one_kelly_alert_count', 'draw_more_one_kelly_alert_count', 'lose_more_one_kelly_alert_count',
            'win_less_one_kelly_alert_count', 'draw_less_one_kelly_alert_count', 'lose_less_one_kelly_alert_count',

            'kelly_volatility', 'sp_momentum',
            'market_disagreement'

        ]

        agency_features = [
            'max_win_agency', 'max_draw_agency', 'min_lose_agency',
            'min_win_agency', 'min_draw_agency', 'min_lose_agency'

        ]

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('agency', 'passthrough', agency_features)
        ])

        return make_pipeline(
            FunctionTransformer(self._create_features),
            preprocessor
        )

    def train_model(self, df):
        """改进的模型训练方法"""
        # 数据准备
        df = df.sort_values('bet_time')
        df = self._filter_bookmakers(df)
        self.feature_processor = self._build_pipeline()
        X = self.feature_processor.fit_transform(df)
        y = df['nwdl_result'].astype(int)  # 确保标签为整数

        # 动态类别权重
        class_dist = y.value_counts(normalize=True)
        class_weight = {i: 1 / (p + 0.1) for i, p in class_dist.items()}

        # 优化模型参数
        base_model = LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=7,
            class_weight=class_weight,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            importance_type='gain'
        )

        # 改进校准方法
        self.model = CalibratedClassifierCV(
            base_model,
            cv=TimeSeriesSplit(n_splits=3),
            method='isotonic'
        )

        # 训练流程
        self.model.fit(X, y)

        # 特征重要性分析
        # self._analyze_feature_importance(base_model)

        # 模型评估
        y_pred = self.model.predict(X)
        print(classification_report(y, y_pred))

        # 保存模型
        self._persist_models()

    def _analyze_feature_importance(self, model):
        """特征重要性分析"""
        feature_names = self.feature_processor.named_steps['columntransformer'].get_feature_names_out()
        self.feature_importance = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        self.feature_importance[:25].plot(kind='barh')
        plt.title('Top 25 Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

    def simulate_betting(self, df, initial_capital=200):
        """改进的投注模拟"""
        # 数据准备
        X = self.feature_processor.transform(df)
        probs = self.model.predict_proba(X)
        odds = df[['first_win_sp', 'first_draw_sp', 'first_lose_sp']].values
        results = df['nwdl_result'].astype(int).values

        capital = initial_capital
        history = [capital]
        bet_count = 0
        win_count = 0

        # 动态风险控制参数
        max_drawdown = 0.3  # 最大回撤30%
        min_confidence = 0.7  # 初始最小置信度
        decay_factor = 0.95  # 连续亏损时降低风险

        for idx in range(len(results)):
            current_confidence = np.max(probs[idx])
            current_threshold = min_confidence * (decay_factor ** bet_count)

            # 动态调整投注阈值
            if current_confidence < current_threshold:
                continue

            # 凯利公式优化
            prob = probs[idx][np.argmax(probs[idx])]
            outcome = results[idx]
            odd = odds[idx][outcome] if outcome < 3 else odds[idx][1]  # 处理异常结果

            # 保守凯利计算
            stake_ratio = self._conservative_kelly(prob, odd)
            stake = capital * stake_ratio

            if stake < 10:
                continue

            # 执行投注
            if outcome == np.argmax(probs[idx]):
                capital += stake * (odd - 1)
                win_count += 1
                decay_factor = min(decay_factor * 1.05, 0.95)  # 获胜后恢复风险承受
            else:
                capital -= stake
                decay_factor *= 0.95  # 失败后降低风险

            bet_count += 1
            history.append(capital)

            # 风险控制
            current_drawdown = (initial_capital - capital) / initial_capital
            if current_drawdown >= max_drawdown:
                print(f"触发最大回撤限制于第{bet_count}次投注")
                break

        # 输出报告
        print(f"最终资金: {capital:.2f}")
        print(f"投注次数: {bet_count}")
        print(f"胜率: {win_count / bet_count:.1%}") if bet_count > 0 else None

        plt.figure(figsize=(10, 6))
        plt.plot(history)
        plt.title("Capital Curve")
        plt.xlabel("Bets")
        plt.ylabel("Amount")
        plt.show()

        return capital

    def _conservative_kelly(self, prob, odds, risk_factor=0.5):
        """保守版凯利公式"""
        b = odds - 1
        q = 1 - prob
        fraction = (b * prob - q) / (b + 1e-6)  # 防止除以零
        return max(min(fraction * risk_factor, 0.3), 0)  # 最高投入30%

    def _persist_models(self):
        """持久化所有组件"""
        joblib.dump(self.model, '../enhanced_odds_model.pkl')
        joblib.dump(self.feature_processor, '../feature_processor.pkl')
        joblib.dump({
            'valid_bookmakers': self.valid_bookmakers,
            'agency_hist': self.agency_hist,
            'feature_importance': self.feature_importance
        }, '../metadata.pkl')
    def load_production_model(self):
        """加载已训练好的模型和特征处理器"""
        self.model = joblib.load('../enhanced_odds_model.pkl')
        # 需要同时保存特征处理器状态
        self.feature_processor = joblib.load('../feature_processor.pkl')
        self.valid_bookmakers = joblib.load('../valid_bookmakers.pkl')
        self.agency_hist = joblib.load('../agency_hist.pkl')


    def predict_new_matches(self, new_df):
        """
        预测新比赛结果
        :param new_df: 包含新比赛数据的新DataFrame
        :return: 预测结果DataFrame
        """
        # 数据预处理
        new_df = self._preprocess_new_data(new_df)

        # 特征转换
        X_new = self.feature_processor.transform(new_df)

        # 预测概率
        probs = self.model.predict_proba(X_new)

        # 构建结果
        result_df = new_df[['match_id', 'bookmaker_id', 'first_win_sp',
                        'first_draw_sp', 'first_lose_sp']].copy()
        result_df['pred_prob_win'] = probs[:, 2]  # 假设类别顺序是[0,1,3]
        result_df['pred_prob_draw'] = probs[:, 1]
        result_df['pred_prob_lose'] = probs[:, 0]

        # 添加凯利建议
        result_df['suggested_stake'] = self._generate_bet_suggestions(result_df)

        return result_df


    def _preprocess_new_data(self, new_df):
        """新数据预处理"""
        # 应用相同过滤条件
        new_df = new_df[
            (new_df['first_handicap'] == 0) &
            (new_df['first_win_sp'] >= 1.12) &
            (new_df['first_lose_sp'] >= 1.12)
            ]

        # 筛选有效机构
        new_df = new_df[new_df['bookmaker_id'].isin(self.valid_bookmakers)]
        # 应用特征工程
        return self._create_features(new_df)
    def _kelly_strategy(self, prob, odds):
        """凯利公式投注策略"""
        b = odds - 1
        q = 1 - prob
        fraction = (b * prob - q) / b
        return max(min(fraction, 0.5), 0)  # 限制单次投注不超过本金的50%
    def _generate_bet_suggestions(self, result_df):
        """生成投注建议"""
        suggestions = []
        for _, row in result_df.iterrows():
            max_prob = max(row['pred_prob_win'], row['pred_prob_draw'], row['pred_prob_lose'])

            # 获取最大概率的索引
            max_index = np.argmax([row['pred_prob_win'], row['pred_prob_draw'], row['pred_prob_lose']])

            # 映射索引到对应的赔率
            odds_keys = ['first_win_sp', 'first_draw_sp', 'first_lose_sp']
            corresponding_odd = row[odds_keys[max_index]]  # 使用映射的赔率

            stake = self._kelly_strategy(max_prob, corresponding_odd)
            suggestions.append(stake)
        return suggestions

    def evaluate_recent_performance(self, df, n=100):
        """
        评估最近N场比赛的预测表现
        :param df: 包含实际比赛结果的历史数据
        :param n: 要分析的最近比赛场次数量
        :return: 包含评估指标的字典
        """
        # 数据预处理
        df = self._preprocess_new_data(df)

        # 获取最近的N场比赛
        recent_matches = df.sort_values('bet_time').tail(n)
        if len(recent_matches) < n:
            print(f"警告：只有{len(recent_matches)}场可用数据")

        # 特征转换
        X = self.feature_processor.transform(recent_matches)

        # 获取实际结果和预测概率
        y_true = recent_matches['nwdl_result'].astype(int).values
        probs = self.model.predict_proba(X)

        # 转换为类别预测（假设类别顺序为[0, 1, 3]对应负、平、胜）
        y_pred = np.argmax(probs, axis=1)
        #y_pred 把2 替换成3
        y_true = np.where(y_true == 3,2, y_true)

        label_map = {0: 'lose', 1: 'draw', 2: 'win'}  # 根据实际类别顺序调整

        # 初始化结果字典
        results = {
            'total_accuracy': np.mean(y_pred == y_true),
            'details': {
                'win': {'accuracy': 0, 'confidence': 0, 'count': 0},
                'draw': {'accuracy': 0, 'confidence': 0, 'count': 0},
                'lose': {'accuracy': 0, 'confidence': 0, 'count': 0}
            }
        }

        # 分类别统计
        for idx, true_label in enumerate(y_true):
            pred_label = label_map[y_pred[idx]]
            actual_label = label_map.get(true_label, 'unknown')

            # 置信度（预测概率）
            confidence = probs[idx][y_pred[idx]]

            # 更新总体统计
            if pred_label in results['details']:
                results['details'][pred_label]['count'] += 1
                results['details'][pred_label]['confidence'] += confidence

                # 更新准确率
                if pred_label == actual_label:
                    results['details'][pred_label]['accuracy'] += 1

        # 计算最终指标
        for outcome in ['win', 'draw', 'lose']:
            detail = results['details'][outcome]
            if detail['count'] > 0:
                detail['accuracy'] = detail['accuracy'] / detail['count']
                detail['confidence'] = detail['confidence'] / detail['count']
            else:
                detail['accuracy'] = 0
                detail['confidence'] = 0

        # 打印易读结果
        print(f"\n最近{len(recent_matches)}场比赛分析报告：")
        print(f"整体准确率：{results['total_accuracy']:.1%}")
        print("\n分项表现：")
        for outcome in ['win', 'draw', 'lose']:
            data = results['details'][outcome]
            print(
                f"{outcome.upper():5} | 预测次数：{data['count']:3} | 准确率：{data['accuracy']:.1%} | 平均置信度：{data['confidence']:.1%}")

        return results


# 使用示例
if __name__ == "__main__":
    analyzer = EnhancedFootballOddsAnalyzer()

    # 训练阶段
    train_df = load_europe_odds_not_handicap_data()
    analyzer.train_model(train_df)

    # 模拟投注
    # final_capital = analyzer.simulate_betting(test_df, initial_capital=1000)
