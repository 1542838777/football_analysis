import numpy as np
import pandas as pd
import joblib
import os
import sys

# 添加项目根目录到系统路径，确保可以导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入必要的函数
from service.util.spfTest import (
    getSelf, preprocess_data, get_models, get_param_grids,
    train_and_evaluate_models, compute_class_weights, plot_feature_importance
)

def train_and_save_model(model_dir='D:\\lqs\\codeAbout\\py\\guessingFootball\\service\\latest\\models', score_weights=None):
    """
        训练模型并保存到指定目录

    参数:
        model_dir (str): 模型保存目录
        score_weights (dict): 综合评分的权重字典，例如:
            {
                'best_score': 0.25,            # 交叉验证得分权重
                'test_balanced_accuracy': 0.25, # 测试集平衡准确率权重
                'recent_30_accuracy': 0.30,     # 最近30场准确率权重
                'recent_150_accuracy': 0.20     # 最近150场准确率权重
            }
            如果为None，则使用默认权重
    """
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)

    # 获取数据
    y_column, guess_type, useless_cols, match_level_df = getSelf()

    print(f"数据加载完成，共 {len(match_level_df)} 条记录")

    # 数据预处理
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(
        match_level_df, y_column, guess_type, useless_cols)

    print(f"数据预处理完成，训练集: {X_train_scaled.shape}, 测试集: {X_test_scaled.shape}")

    # 类别权重计算
    class_weights = compute_class_weights(y_train)

    # 获取模型和参数网格
    models = get_models()
    param_grids = get_param_grids()

    # 训练并评估模型
    best_models = train_and_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test, param_grids, models, feature_names)

    # 从 spfTest 模块导入计算综合评分的函数
    from service.util.spfTest import calculate_composite_score

    # 计算每个模型的综合评分
    model_scores = {}
    for model_name, model_info in best_models.items():
        # 计算综合评分
        composite_score = calculate_composite_score(model_info, score_weights)
        model_scores[model_name] = composite_score

        # 打印每个模型的评分详情
        print(f"\n{model_name} 模型的评分详情:")
        print(f"  交叉验证得分: {model_info['best_score']:.2%}")
        print(f"  测试集平衡准确率: {model_info['test_balanced_accuracy']:.2%}")
        print(f"  最近30场准确率: {model_info['recent_30_accuracy']:.2%}")
        print(f"  最近150场准确率: {model_info['recent_150_accuracy']:.2%}")
        print(f"  综合评分: {composite_score:.2%}")

    # 选择综合评分最高的模型
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = best_models[best_model_name]['best_estimator']

    print(f"\n最佳模型: {best_model_name}")
    #保存最佳模型名
    # 将标量值包裹在列表中，这样pandas可以正确创建DataFrame
    df = pd.DataFrame({'best_model_name':[best_model_name]})
    df.to_csv('best_model_name.csv', index=False)

    # 另一种更简单的方法，直接将模型名称写入文本文件
    with open('best_model_name.txt', 'w') as f:
        f.write(best_model_name)

    print(f"综合评分: {model_scores[best_model_name]:.2%}")
    print(f"交叉验证得分: {best_models[best_model_name]['best_score']:.2%}")
    print(f"测试集平衡准确率: {best_models[best_model_name]['test_balanced_accuracy']:.2%}")
    print(f"最近30场准确率: {best_models[best_model_name]['recent_30_accuracy']:.2%}")
    print(f"最近150场准确率: {best_models[best_model_name]['recent_150_accuracy']:.2%}")

    # 保存模型和相关组件
    joblib.dump(best_model, os.path.join(model_dir, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.pkl'))
    joblib.dump(guess_type, os.path.join(model_dir, 'guess_type.pkl'))

    # 保存模型元数据
    metadata = {
        'model_name': best_model_name,
        'composite_score': model_scores[best_model_name],
        'cross_validation_score': best_models[best_model_name]['best_score'],
        'test_balanced_accuracy': best_models[best_model_name]['test_balanced_accuracy'],
        'recent_30_accuracy': best_models[best_model_name]['recent_30_accuracy'],
        'recent_150_accuracy': best_models[best_model_name]['recent_150_accuracy'],
        'feature_count': len(feature_names),
        'train_samples': X_train_scaled.shape[0],
        'test_samples': X_test_scaled.shape[0],
        'class_distribution': {
            'train': np.bincount(y_train).tolist(),
            'test': np.bincount(y_test).tolist()
        },
        'score_weights': score_weights or {
            'best_score': 0.20,
            'test_balanced_accuracy': 0.25,
            'recent_30_accuracy': 0.33,
            'recent_150_accuracy': 0.22
        }
    }

    joblib.dump(metadata, os.path.join(model_dir, 'metadata.pkl'))

    print(f"\n模型和相关组件已保存到 {model_dir} 目录")

    # 特征重要性可视化
    # 获取最佳模型的完整信息，包括选定的特征
    best_model_info = best_models[best_model_name]

    # 检查是否有选定的特征
    selected_features = best_model_info.get('selected_features', feature_names)

    # 使用完整的模型信息进行特征重要性可视化
    plot_feature_importance({best_model_name: best_model_info}, feature_names)

    return best_model, scaler, feature_names, guess_type

if __name__ == '__main__':
    # 可以自定义权重，例如更重视最近的比赛结果
    custom_weights = {
        'best_score': 0.20,            # 交叉验证得分权重
        'test_balanced_accuracy': 0.20, # 测试集平衡准确率权重
        'recent_30_accuracy': 0.40,     # 最近30场准确率权重
        'recent_150_accuracy': 0.20     # 最近150场准确率权重
    }

    # 使用默认权重
    train_and_save_model()

    # 或者使用自定义权重
    # train_and_save_model(score_weights=custom_weights)
