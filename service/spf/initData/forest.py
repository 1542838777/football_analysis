import pandas as pd

from config.mysql import engine
from service.spf.initData.spfProduct import FootballOddsAnalyzer


def fetch_new_matches():
    """从数据库获取待预测的新比赛"""
    query = """
  SELECT o.*,r.host_name,r.guest_name,order_queue,league_id,bet_time
FROM europe_odds o
         JOIN match_info r ON o.match_id = r.id
WHERE o.first_handicap = 0
  and first_win_sp >= 1.12
  and first_lose_sp >= 1.12
  and bet_time >= '2025-03-06'

    """
    return pd.read_sql(query, engine)


def save_predictions(predictions):
    """保存预测结果到数据库"""
    predictions.to_sql('odds_predictions', engine, if_exists='append', index=False)


def main_production():
    # 初始化分析器
    analyzer = FootballOddsAnalyzer()
    analyzer.load_production_model()

    while True:  # 可改为定时任务
        # 获取新比赛数据
        new_data = fetch_new_matches()

        if not new_data.empty:
            # 进行预测
            predictions = analyzer.predict_new_matches(new_data)

            # 保存结果
            # save_predictions(predictions)

            # 打印实时建议
            print(f"新预测结果（{pd.Timestamp.now()}）：")
            #按match_id分组。只展示第一个
            #new_data 和 predictions 字段靠match_id 连接，predictions基础上 追加host_name,字段
            predictions = pd.merge(new_data, predictions, on='match_id')
            predictions = predictions.groupby('match_id').first().reset_index()
            # 按照 betTime order_queue 排序
            predictions = predictions.sort_values(['bet_time', 'order_queue'], ascending=[True, True])
            # 设置全局参数
            pd.set_option('display.max_columns', None)
            pd.set_option('display.expand_frame_repr', False)
            pd.set_option('display.max_colwidth', 20)
            pd.set_option('display.width', 200)  # 根据你的终端宽度调整
            # 设置显示的最大行数和列数
            pd.set_option('display.max_rows', None)  # 显示所有行
            pd.set_option('display.max_columns', None)  # 显示所有列
            #只要 pred_prob_win 大于0.7的

            print(predictions[['match_id', 'pred_prob_win',
                               'pred_prob_draw','pred_prob_lose','host_name', 'suggested_stake','order_queue','bet_time']].head(10000))
            break;

        # 间隔1小时轮询
        # time.sleep(3600)


if __name__ == '__main__':
    main_production();
