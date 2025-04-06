# 模拟10家公司的初始赔率数据（主胜/平局/客胜）
odds_data = [
    {'company': 'William Hill', 'home': 1.85, 'draw': 3.40, 'away': 4.20},
    {'company': 'Bet365', 'home': 1.90, 'draw': 3.30, 'away': 4.50},
    {'company': 'Pinnacle', 'home': 1.88, 'draw': 3.50, 'away': 4.00},
    # 添加其他7家公司数据...
]


def validate_odds(odds):
    return all(1.0 < x < 500.0 for x in [odds['home'], odds['draw'], odds['away']])


cleaned_data = [x for x in odds_data if validate_odds(x)]

import numpy as np

# 定义公司权重（示例：按公司信誉分级）
company_weights = {
    'William Hill': 0.95,
    'Bet365': 0.93,
    'Pinnacle': 0.90,
    # 其他公司权重...
}


def calculate_discrete_degree(odds_list, outcome_type='home'):
    """
    计算指定结果类型的加权离散度
    :param odds_list: 赔率数据列表
    :param outcome_type: 结果类型（home/draw/away）
    :return: 离散度指标（0-100，越高分歧越大）
    """
    # 提取指定结果的赔率及权重
    values = []
    weights = []
    for entry in odds_list:
        if entry['company'] in company_weights:
            values.append(entry[outcome_type])
            weights.append(company_weights[entry['company']])

    if not values:
        return 0.0

    # 加权平均值
    weighted_mean = np.average(values, weights=weights)

    # 加权方差
    weighted_variance = np.average(
        [(x - weighted_mean) ** 2 for x in values],
        weights=weights
    )

    # 标准化离散度（0-100范围）
    max_variance = 10.0  # 根据历史数据调整
    discrete_degree = min((weighted_variance ** 0.5) / max_variance * 100, 100)

    return round(discrete_degree, 2)


from collections import deque


class OddsMonitor:
    def __init__(self, window_size=5):
        self.history = {
            'home': deque(maxlen=window_size),
            'draw': deque(maxlen=window_size),
            'away': deque(maxlen=window_size)
        }

    def update_and_analyze(self, new_odds):
        # 计算当前离散度
        current = {
            'home': calculate_discrete_degree(new_odds, 'home'),
            'draw': calculate_discrete_degree(new_odds, 'draw'),
            'away': calculate_discrete_degree(new_odds, 'away')
        }

        # 存储历史数据
        for key in self.history:
            self.history[key].append(current[key])

        # 计算变化趋势
        trend = {}
        for key in self.history:
            if len(self.history[key]) >= 2:
                trend[f'{key}_trend'] = current[key] - np.mean(list(self.history[key])[:-1])
            else:
                trend[f'{key}_trend'] = 0.0

        return {**current, **trend}

import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(odds_data):
        # 创建矩阵数据
        companies = [x['company'] for x in odds_data]
        outcomes = ['home', 'draw', 'away']
        data = np.array([[x[outcome] for outcome in outcomes] for x in odds_data])

        # 绘制热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.T, annot=True, xticklabels=companies, yticklabels=outcomes)
        plt.title('Oddis Distribution Heatmap')
        plt.xlabel('Bookmakers')
        plt.ylabel('Outcomes')
        plt.show()


def plot_trend_analysis(history_data):
    plt.figure(figsize=(12, 6))
    for outcome in ['home', 'draw', 'away']:
        plt.plot(history_data[outcome], label=outcome.upper())

    plt.title('Odds Discrete Trend Over Time')
    plt.xlabel('Time Window')
    plt.ylabel('Discrete Degree')
    plt.legend()
    plt.grid(True)
    plt.show()


monitor = OddsMonitor()

def realtime_processing(new_odds):
    analysis = monitor.update_and_analyze(new_odds)

    # 触发告警条件
    if analysis['home'] > 85.0 and analysis['home_trend'] > 15.0:
        print(f"[ALERT] 主胜分歧激增！当前离散度: {analysis['home']}%")

    return analysis


def pre_match_risk_assessment(odds_data):
    risk_scores = {}
    for outcome in ['home', 'draw', 'away']:
        score = calculate_discrete_degree(odds_data, outcome)
        risk_scores[outcome] = '高风险' if score > 75 else '中风险' if score > 50 else '低风险'

    return risk_scores

if __name__ == '__main__':
    pre_match_risk_assessment(odds_data)
