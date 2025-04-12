import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
import sys

# 添加项目根目录到系统路径，确保可以导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入必要的函数
from service.util.spfTest import create_features, getSelf, preprocess_data, get_target_names
from service.spf.initData.data.mysql_data import load_europe_odds_not_handicap_data



def load_model(model_path):
    """
    加载保存的模型和相关组件
    """
    try:
        model = joblib.load(model_path)
        print(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None

def preprocess_new_data(new_data, feature_names, scaler):
    """
    对新数据进行预处理
    """
    # 创建特征
    features_df = create_features(new_data)

    # 确保特征列与训练时一致
    common_cols = list(set(features_df.columns) & set(feature_names))
    features_df = features_df[common_cols]

    # 标准化
    features_scaled = scaler.transform(features_df)

    return pd.DataFrame(features_scaled, columns=common_cols, index=features_df.index)

def predict_new_matches(model_path='best_model.pkl', scaler_path='scaler.pkl', feature_names_path='feature_names.pkl'):
    """
    主函数：预测新比赛结果
    """
    # 加载模型和相关组件
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)

    if model is None:
        print("无法加载模型，预测终止")
        return

    # 获取新比赛数据
    try:
        #导入 mysql_data.py 里面的fetch_new_matches
        from service.spf.initData.data.mysql_data import fetch_new_matches
        new_matches = fetch_new_matches()
        if new_matches.empty:
            print("没有找到新的比赛数据")
            return

        print(f"获取到 {len(new_matches)} 场新比赛")
    except Exception as e:
        print(f"获取新比赛数据失败: {str(e)}")
        # 如果无法从数据库获取，可以尝试从文件加载
        try:
            new_matches = pd.read_csv('new_matches.csv')
            print(f"从文件加载了 {len(new_matches)} 场新比赛")
        except:
            print("无法获取新比赛数据，预测终止")
            return

    # 数据预处理
    try:
        # 使用fetch_new_matches获取的数据
        y_column = 'nwdl_result'  # 目标变量
        guess_type = 'win_draw_loss'  # 预测类型
        useless_cols = ['bet_time']  # 无用列

        # 将原始数据转换为match维度的数据
        from service.util.spfTest import create_match_level_future_by_match_group
        # 先删除无用列
        new_matches_processed = new_matches.drop(useless_cols, axis=1, errors='ignore')
        # 使用create_match_level_future_by_match_group函数将数据打平成match维度
        match_level_df = create_match_level_future_by_match_group(new_matches_processed)

        # 处理新数据
        new_match_level_df = create_features(match_level_df, useless_cols)
        #
        # 确保特征列与训练时一致
        X_new = new_match_level_df[feature_names]

        # 标准化
        X_new_scaled = scaler.transform(new_match_level_df)

        # 预测
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)

        # 获取目标名称
        target_names = get_target_names(guess_type)

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'match_id': new_match_level_df.index,
            'prediction': [target_names[p] for p in predictions]
        })

        # 添加概率列
        for i, name in enumerate(target_names):
            results_df[f'prob_{name}'] = probabilities[:, i]

        # 如果有主客队名称，添加到结果中
        if 'host_name' in new_matches.columns and 'guest_name' in new_matches.columns:
            match_info = new_matches.groupby('match_id').first()[['host_name', 'guest_name','order_queue']]
            results_df = results_df.merge(match_info, left_on='match_id', right_index=True, how='left')

            # 重新排列列顺序
            cols = ['match_id', 'host_name', 'guest_name','order_queue', 'prediction'] + [c for c in results_df.columns if c.startswith('prob_')]
            results_df = results_df[cols]

        # 输出结果
        print("\n预测结果:")
        #  order_queue强制转为 int
        if 'order_queue' in results_df.columns:
            # 先处理可能的NaN值
            results_df['order_queue'] = results_df['order_queue'].fillna(9999)  # 给缺失值设置一个大数值，让它们排在最后
            results_df['order_queue'] = results_df['order_queue'].astype(int)
            # 使用inplace=True进行原地排序
            results_df.sort_values(['order_queue'], ascending=[True], inplace=True)
        print(results_df.to_string())

        # 保存结果
        results_df.to_csv('prediction_results.csv', index=False)
        print("\n预测结果已保存到 prediction_results.csv")

        return results_df

    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_trained_model(model, scaler, feature_names, prefix=''):
    """
    保存训练好的模型和相关组件，供预测使用
    """
    joblib.dump(model, f'{prefix}best_model.pkl')
    joblib.dump(scaler, f'{prefix}scaler.pkl')
    joblib.dump(feature_names, f'{prefix}feature_names.pkl')
    print(f"模型和相关组件已保存，可用于预测新数据")

if __name__ == '__main__':
    # 如果已有训练好的模型，直接预测
    if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl') and os.path.exists('feature_names.pkl'):
        predict_new_matches()
    else:
        # 否则，先训练模型
        print("未找到训练好的模型，请先运行 spfTest.py 训练模型")

        # 获取数据
        y_column, guess_type, useless_cols, match_level_df = getSelf()

        # 数据预处理
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(
            match_level_df, y_column, guess_type, useless_cols)

        # 导入模型训练函数
        from service.util.spfTest import get_models, get_param_grids, train_and_evaluate_models, compute_class_weights

        # 类别权重计算
        class_weights = compute_class_weights(y_train)

        # 获取模型和参数网格
        models = get_models()
        param_grids = get_param_grids()

        # 训练并评估模型
        best_models = train_and_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test, param_grids, models, feature_names)

        # 选择最佳模型
        best_model_name = max(best_models, key=lambda k: best_models[k]['best_score'])
        best_model = best_models[best_model_name]['best_estimator']

        # 保存模型
        save_trained_model(best_model, scaler, feature_names)

        # 预测新数据
        predict_new_matches()
