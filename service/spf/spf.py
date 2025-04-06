import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

from service.spf.initData.data.mysql_data import load_europe_odds_not_handicap_data

warnings.filterwarnings('ignore')


# 特征工程函数
def feature_engineering(df):
    """利用max/min标记和其他字段生成高级特征"""

    # 机构极端值特征
    df['is_extreme_agency'] = df[['max_first_win_sp', 'max_first_lose_sp',
                                  'max_first_draw_sp', 'min_first_draw_sp',
                                  'min_first_win_sp', 'min_first_lose_sp']].max(axis=1)

    # 机构共识特征
    match_stats = df.groupby('match_id').agg({
        'max_first_win_sp': 'sum',
        'min_first_win_sp': 'sum',
        'first_win_sp': ['std', 'max'],

        'max_first_draw_sp': 'sum',
        'min_first_draw_sp': 'sum',
        'first_draw_sp': ['std', 'max'],

        'max_first_lose_sp': 'sum',
        'min_first_lose_sp': 'sum',
        'first_lose_sp': ['std', 'max'],

        'first_back_rate': 'median'
    })
    match_stats.columns = ['max_win_count', 'min_win_count', 'win_std', 'win_max',
                           'max_draw_count', 'min_draw_count', 'draw_std', 'draw_max',
                           'max_lose_count', 'min_lose_count', 'lose_std', 'lose_max',
                           'median_back_rate']

    # 凯利异常特征
    df['kelly_alert'] = ((df['first_win_kelly_index'] < 0.9) &
                         (df['first_win_sp'] > df.groupby('match_id')['first_win_sp'].transform('mean')))

    # 合并比赛级特征
    df = df.merge(match_stats, on='match_id')

    # 时间敏感权重
    df['time_weight'] = 1 / (1 + np.exp(-df['last_update_time_distance'] / 1440))

    return df



# 预处理管道
def preprocess_pipeline(df):
    # 生成特征
    df = feature_engineering(df)

    # 选择特征列
    features = [
        'max_first_win_sp', 'min_first_win_sp',
        'first_win_sp', 'first_win_kelly_index',
        'win_std','max_win_count',

        'max_first_draw_sp', 'min_first_draw_sp',
        'first_draw_sp', 'first_draw_kelly_index',
        'draw_std', 'max_draw_count',

        'max_first_lose_sp', 'min_first_lose_sp',
        'first_lose_sp', 'first_lose_kelly_index',
        'lose_std', 'max_lose_count',

        'median_back_rate',
        'time_weight', 'is_extreme_agency',
        'kelly_alert'
    ]

    # 处理缺失值
    imputer = SimpleImputer(strategy='median')
    df[features] = imputer.fit_transform(df[features])

    # 标准化
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df[features + ['nwdl_result', 'first_win_sp', 'match_id']]


# 盈利回测系统
class BettingSimulator:
    def __init__(self, model):
        self.model = model
        self.profit = 0
        self.history = []

    def simulate(self, X, y, odds):
        proba = self.model.predict_proba(X)[:, 1]
        for p, true_result, odd in zip(proba, y, odds):
            # 投注策略：预测概率>60%且赔率有优势
            if p > 0.6 and odd > 1 / (p):
                stake = 100  # 每注金额
                if true_result == 3:
                    self.profit += (odd - 1) * stake
                else:
                    self.profit -= stake
                self.history.append(self.profit)
        return self.profit


# 主流程
def main():
    # 加载数据
    df = load_europe_odds_not_handicap_data()
    print(f"总数据量：{len(df)}条")

    # 预处理
    processed_df = preprocess_pipeline(df)

    # 拆分数据集
    X = processed_df.drop(['nwdl_result', 'match_id'], axis=1)
    y = processed_df['nwdl_result']
    odds = processed_df['first_win_sp']

    X_train, X_test, y_train, y_test, odds_train, odds_test = train_test_split(
        X, y, odds, test_size=0.2, stratify=y
    )

    # 模型训练
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 模型评估
    print("\n模型评估报告：")
    print(classification_report(y_test, model.predict(X_test)))

    # 特征重要性
    plt.figure(figsize=(10, 6))
    pd.Series(model.feature_importances_, index=X.columns).sort_values().plot.barh()
    plt.title("Feature Importance")
    plt.show()

    # 盈利回测
    simulator = BettingSimulator(model)
    final_profit = simulator.simulate(X_test, y_test, odds_test)
    print(f"\n最终盈利：{final_profit:.2f} 元")

    # 盈利曲线可视化
    plt.plot(simulator.history)
    plt.title("Profit Curve")
    plt.xlabel("Bets")
    plt.ylabel("Cumulative Profit")
    plt.show()


if __name__ == '__main__':
    main()