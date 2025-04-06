import pandas as pd
import mysql.connector

# 数据库连接配置 
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'root',
    'database': 'football_guessing',
    'charset': 'utf8'
}

try:
    # 建立连接并读取数据 
    conn = mysql.connector.connect(**db_config)

    df = pd.read_sql(
        'select spf_result, league_id, first_win_odds_of24h, first_win_odds_of24h_bucket, first_draw_odds_of24h, first_lose_odds_of24h, last_win_odds_of24h, last_win_odds_of24h_bucket, last_draw_odds_of24h, last_lose_odds_of24h, win_odds_of24h_drop_num, win_odds_of24h_drop_num_bucket, draw_odds_of24h_drop_num, draw_odds_of24h_drop_num_bucket, lose_odds_of24h_drop_num, lose_odds_of24h_drop_num_bucket, win_odds_of24h_up_num, win_odds_of24h_up_num_bucket, draw_odds_of24h_up_num, draw_odds_of24h_up_num_bucket, lose_odds_of24h_up_num, lose_odds_of24h_up_num_bucket, win_odds_of24h_change_num, win_odds_of24h_change_num_bucket, draw_odds_of24h_change_num, draw_odds_of24h_change_num_bucket, lose_odds_of24h_change_num, lose_odds_of24h_change_num_bucket, win_odds_of24h_stable_num, win_odds_of24h_stable_num_bucket, draw_odds_of24h_stable_num, draw_odds_of24h_stable_num_bucket, lose_odds_of24h_stable_num, lose_odds_of24h_stable_num_bucket, total_num_of24h, total_num_of24h_bucket, win_odds_of24h_change_num_rank, draw_odds_of24h_change_num_rank, lose_odds_of24h_change_num_rank, win_odds_of24h_stable_num_rank, draw_odds_of24h_stable_num_rank, lose_odds_of24h_stable_num_rank, win_odds_of24h_final_change, draw_odds_of24h_final_change, lose_odds_of24h_final_change, win_odds_of24h_delta, draw_odds_of24h_delta, lose_odds_of24h_delta, win_odds_of24h_delta_bucket, draw_odds_of24h_delta_bucket, lose_odds_of24h_delta_bucket from match_factor',
        con=conn)
    print(df.head())  # 验证数据

except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals() and conn.is_connected():
        conn.close()


# 写一个方法 查询数据库，并且返回查询的数据
def query_database(sql):
    try:
        # 建立连接并读取数据
        conn = mysql.connector.connect(**db_config)
        df = pd.read_sql(sql, con=conn)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()


def query_allMatchFactor():
    try:
        # 建立连接并读取数据
        conn = mysql.connector.connect(**db_config)
        df = pd.read_sql(
            'select spf_result, league_id, first_win_odds_of24h, first_win_odds_of24h_bucket, first_draw_odds_of24h, first_lose_odds_of24h, last_win_odds_of24h, last_win_odds_of24h_bucket, last_draw_odds_of24h, last_lose_odds_of24h, win_odds_of24h_drop_num, win_odds_of24h_drop_num_bucket, draw_odds_of24h_drop_num, draw_odds_of24h_drop_num_bucket, lose_odds_of24h_drop_num, lose_odds_of24h_drop_num_bucket, win_odds_of24h_up_num, win_odds_of24h_up_num_bucket, draw_odds_of24h_up_num, draw_odds_of24h_up_num_bucket, lose_odds_of24h_up_num, lose_odds_of24h_up_num_bucket, win_odds_of24h_change_num, win_odds_of24h_change_num_bucket, draw_odds_of24h_change_num, draw_odds_of24h_change_num_bucket, lose_odds_of24h_change_num, lose_odds_of24h_change_num_bucket, win_odds_of24h_stable_num, win_odds_of24h_stable_num_bucket, draw_odds_of24h_stable_num, draw_odds_of24h_stable_num_bucket, lose_odds_of24h_stable_num, lose_odds_of24h_stable_num_bucket, total_num_of24h, total_num_of24h_bucket, win_odds_of24h_change_num_rank, draw_odds_of24h_change_num_rank, lose_odds_of24h_change_num_rank, win_odds_of24h_stable_num_rank, draw_odds_of24h_stable_num_rank, lose_odds_of24h_stable_num_rank, win_odds_of24h_final_change, draw_odds_of24h_final_change, lose_odds_of24h_final_change, win_odds_of24h_delta, draw_odds_of24h_delta, lose_odds_of24h_delta, win_odds_of24h_delta_bucket, draw_odds_of24h_delta_bucket, lose_odds_of24h_delta_bucket from match_factor',
            con=conn)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()
