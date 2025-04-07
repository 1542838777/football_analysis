import pandas as pd

from config.mysql import engine


def load_europe_odds_not_handicap_data():
    # 创建数据库连接
    #and bookmaker_id in(1000,57,25,112,11)
    # 加载原始数据
    query = """
    
   

SELECT bookmaker_id,
       o.match_id,
       first_win_sp,
       first_draw_sp,
       first_lose_sp,
       first_win_kelly_index,
       first_draw_kelly_index,
       first_lose_kelly_index,
       first_handicap,
       first_back_rate,
       max_first_win_sp,
       max_first_draw_sp,
       max_first_lose_sp,
       min_first_win_sp,
       min_first_draw_sp,
       min_first_lose_sp,
       last_update_time_distance,
       r.nwdl_result,
       league_id,
       bet_time
FROM europe_odds o
         JOIN match_result r ON o.match_id = r.match_id
WHERE o.first_handicap = 0
  and first_win_sp >= 1.13
  and first_lose_sp >= 1.12
# and bet_time <= '2025-03-20'
  and bookmaker_id in (
    3,
        11,99,63,75,64,39,84,91,68,79,22,32,6,24,126,82,161,18,74,57,192,93,72,47,25,80,17,127,9,106,48,115,42,121,130,70,60,1000,
110

    )
order by r.bet_time, match_id

    """
    raw_df = pd.read_sql(query, engine)

    # 筛选覆盖度>=85%的机构
    total_matches = raw_df['match_id'].nunique()
    valid_agencies = raw_df.groupby('bookmaker_id').filter(
        lambda x: x['match_id'].nunique() >= 0.85 * total_matches
    )['bookmaker_id'].unique()
    df = raw_df[raw_df['bookmaker_id'].isin(valid_agencies)]
    # 按 match_time, match_id 排序
    df = df.sort_values(['bet_time', 'match_id'])
    return df

