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

def train_and_save_model(model_dir='models'):
    """
    训练模型并保存到指定目录
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

    # 选择最佳模型
    best_model_name = max(best_models, key=lambda k: best_models[k]['best_score'])
    best_model = best_models[best_model_name]['best_estimator']

    print(f"\n最佳模型: {best_model_name}")
    print(f"平衡准确率: {best_models[best_model_name]['best_score']:.2%}")

    # 保存模型和相关组件
    joblib.dump(best_model, os.path.join(model_dir, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.pkl'))
    joblib.dump(guess_type, os.path.join(model_dir, 'guess_type.pkl'))

    # 保存模型元数据
    metadata = {
        'model_name': best_model_name,
        'balanced_accuracy': best_models[best_model_name]['best_score'],
        'feature_count': len(feature_names),
        'train_samples': X_train_scaled.shape[0],
        'test_samples': X_test_scaled.shape[0],
        'class_distribution': {
            'train': np.bincount(y_train).tolist(),
            'test': np.bincount(y_test).tolist()
        }
    }

    joblib.dump(metadata, os.path.join(model_dir, 'metadata.pkl'))

    print(f"\n模型和相关组件已保存到 {model_dir} 目录")

    # 特征重要性可视化
    plot_feature_importance({best_model_name: {'best_estimator': best_model}}, feature_names)

    return best_model, scaler, feature_names, guess_type

if __name__ == '__main__':
    train_and_save_model()
