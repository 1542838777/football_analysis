import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib

from service.spf.initData.data.mysql_data import load_europe_odds_not_handicap_data


class FootballOddsAnalyzer:
    def __init__(self):
        self.valid_bookmakers = None
        self.feature_processor = None
        self.model = None
        self.agency_hist = {}

    def _filter_bookmakers(self, df, coverage_threshold=0.85):
        """筛选覆盖率达到要求的机构"""
        total_matches = df['match_id'].nunique()
        bookmaker_coverage = df.groupby('bookmaker_id')['match_id'].nunique() / total_matches
        self.valid_bookmakers = bookmaker_coverage[bookmaker_coverage >= coverage_threshold].index.tolist()
        return df[df['bookmaker_id'].isin(self.valid_bookmakers)]

    def _create_features(self, df):
        """高级特征工程"""
        # 机构历史表现
        if not self.agency_hist:
            self.agency_hist = df.groupby('bookmaker_id')['nwdl_result'].value_counts(normalize=True) \
                .unstack().fillna(0).to_dict('index')

        # 个体机构特征
        df['agency_win_ratio'] = df['bookmaker_id'].map(lambda x: self.agency_hist.get(x, {}).get('3', 0.33))
        df['agency_draw_ratio'] = df['bookmaker_id'].map(lambda x: self.agency_hist.get(x, {}).get('1', 0.33))
        df['agency_lose_ratio'] = df['bookmaker_id'].map(lambda x: self.agency_hist.get(x, {}).get('0', 0.33))

        # 市场共识特征
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

        # 极值机构特征
        df['max_win_agency'] = grp['max_first_win_sp'].transform('sum')
        df['min_win_agency'] = grp['min_first_win_sp'].transform('sum')
        df['max_draw_agency'] = grp['max_first_draw_sp'].transform('sum')
        df['min_draw_agency'] = grp['min_first_draw_sp'].transform('sum')
        df['max_lose_agency'] = grp['max_first_lose_sp'].transform('sum')
        df['min_lose_agency'] = grp['min_first_lose_sp'].transform('sum')

        # 凯利异常特征
        df['win_less_kelly_alert'] = (df['first_win_kelly_index'] < 0.9)
        df['draw_less_kelly_alert'] = (df['first_draw_kelly_index'] < 0.9)
        df['lose_less_kelly_alert'] = (df['first_lose_kelly_index'] < 0.9)

        df['win_more_one_kelly_alert'] = (df['first_win_kelly_index'] >= 1)
        df['draw_more_one_kelly_alert'] = (df['first_draw_kelly_index'] >=1  )
        df['lose_more_one_less_kelly_alert'] = (df['first_lose_kelly_index'] >=1)


        # 时间衰减因子
        df['time_factor'] = 1 / (1 + np.exp(-df['last_update_time_distance'] / 1440))

        # 赔率离散度
        df['win_draw_division'] = grp['first_win_sp'].transform('mean') / grp['first_draw_sp'].transform('mean')
        df['draw_lose_division'] = grp['first_draw_sp'].transform('mean') / grp['first_lose_sp'].transform('mean')
        df['win_lose_division'] = grp['first_win_sp'].transform('mean') / grp['first_lose_sp'].transform('mean')

        df['win_draw_spread'] = grp['first_win_sp'].transform('mean') - grp['first_draw_sp'].transform('mean')
        df['draw_lose_spread'] = grp['first_draw_sp'].transform('mean') - grp['first_lose_sp'].transform('mean')
        df['win_lose_spread'] = grp['first_win_sp'].transform('mean') - grp['first_lose_sp'].transform('mean')

        #特殊机构对比
        #bookmaker_id 为1000的机构 与 bookmaker_id 为57的机构 赔率值相差大小

        # 应用函数并生成结果 DataFrame
        # 特殊机构对比特征（新增部分）
        def calculate_odds_difference(group):
            """计算特定机构间的赔率差异"""
            features = {}

            # 定义需要对比的机构组合
            comparisons = [
                (1000, 57, '1000_57')
            ]

            for agency1, agency2, suffix in comparisons:
                # 获取机构1的赔率
                odds1 = group[group['bookmaker_id'] == agency1][
                    ['first_win_sp', 'first_draw_sp', 'first_lose_sp']].values
                # 获取机构2的赔率
                odds2 = group[group['bookmaker_id'] == agency2][
                    ['first_win_sp', 'first_draw_sp', 'first_lose_sp']].values

                # 计算差值（当两个机构都存在时）
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

        # 应用特征计算
        odds_diff_features = df.groupby('match_id').apply(calculate_odds_difference)
        df = df.merge(odds_diff_features, how='left', on='match_id')

        return df
    def _build_pipeline(self):
        numeric_features = [
            'first_win_sp', 'first_draw_sp', 'first_lose_sp',
            'market_win_std', 'market_draw_std', 'market_lose_std',
            'win_draw_division', 'win_lose_division', 'draw_lose_division',
            'win_draw_spread', 'win_lose_spread', 'draw_lose_spread',
            'time_factor'
        ]
        agency_features = [
            'agency_win_ratio', 'agency_draw_ratio','agency_lose_ratio',
            'max_win_agency','max_draw_agency', 'min_lose_agency',

            'min_win_agency','min_draw_agency', 'min_lose_agency'
        ]

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('agency', 'passthrough', agency_features)
        ])
        return make_pipeline(
            FunctionTransformer(self._create_features),
            preprocessor
        )

    def _kelly_strategy(self, prob, odds):
        """凯利公式投注策略"""
        b = odds - 1
        q = 1 - prob
        fraction = (b * prob - q) / b
        return max(min(fraction, 0.5), 0)  # 限制单次投注不超过本金的50%

    def train_model(self, df):
        """模型训练"""
        # 数据预处理
        df = df.sort_values('bet_time')
        df = self._filter_bookmakers(df)
        self.feature_processor = self._build_pipeline()
        X = self.feature_processor.fit_transform(df)
        y = df['nwdl_result']
        class_weight = {'0': 1 / (3 * 0.30), '1': 1 / (3 * 0.26), '3': 1 / (3 * 0.44)}

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=7,
            class_weight=class_weight,
            subsample=0.8,
            colsample_bytree=0.7
        )

        # 训练并校准概率
        calibrated_model = CalibratedClassifierCV(model, cv=tscv, method='isotonic')
        calibrated_model.fit(X, y)

        # 评估模型
        print(classification_report(y, calibrated_model.predict(X)))

        self.model = calibrated_model
        # 保存模型和特征处理器
        joblib.dump(self.model, '../football_odds_model.pkl')
        joblib.dump(self.feature_processor, '../feature_processor.pkl')  # 保存特征处理器
        joblib.dump(self.valid_bookmakers, '../valid_bookmakers.pkl')  # 如果需要保存的其他对象
        joblib.dump(self.agency_hist, '../agency_hist.pkl')  # 如果需要保存的其他对象

    def simulate_betting(self, df, initial_capital=200):
        """模拟投注"""
        if not self.model:
            raise ValueError("需要先训练模型")

        X = self.feature_processor.transform(df)
        probs = self.model.predict_proba(X)
        odds = df[['first_win_sp', 'first_draw_sp', 'first_lose_sp']].values
        results = df['nwdl_result'].values

        # 确保 results 中的值是整数
        results = results.astype(int)

        capital = initial_capital
        history = [capital]

        for idx in range(len(results)):
            prob = probs[idx][np.argmax(probs[idx])]
            outcome = results[idx]
            if prob < 0.8:  # 最小投注额限制
                continue
            # 确保 outcome 是整数
            if isinstance(outcome, str):
                try:
                    outcome = int(outcome)
                except ValueError:
                    print(f"无法将 outcome 转换为整数: {outcome}")
                    continue

            if 0 <= outcome < len(odds[idx]):
                odd = odds[idx][outcome]
            else:
                odd = odds[idx][1]

            # 确定投注比例
            stake_ratio = self._kelly_strategy(prob, odd)
            stake = capital * stake_ratio

            if stake < 10:  # 最小投注额限制
                continue

            if outcome == np.argmax(probs[idx]):
                capital += stake * (odd - 1)
            else:
                capital -= stake

            # 风险控制
            if capital < initial_capital * 0.7:
                print("触发风险控制，停止投注")
                break

            history.append(capital)

        # 可视化资金曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history)
        plt.title("资金变动曲线")
        plt.xlabel("投注次数")
        plt.ylabel("资金量")
        plt.show()
        return capital


    def load_production_model(self):
        """加载已训练好的模型和特征处理器"""
        self.model = joblib.load('../football_odds_model.pkl')
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




# 使用示例
if __name__ == "__main__":
    raw_df = load_europe_odds_not_handicap_data()
    analyzer = FootballOddsAnalyzer()
    analyzer.train_model(raw_df)
    print("模型类别顺序：", analyzer.model.classes_)

    # 模拟投注
    final_capital = analyzer.simulate_betting(raw_df)
    print(f"最终资金: {final_capital:.2f} 元")
