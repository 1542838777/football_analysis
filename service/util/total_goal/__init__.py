import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine
import joblib

# 数据库配置（请根据实际情况修改）
DB_CONFIG = {
    'user': 'root',
    'password': 'root',
    'host': '127.0.0.1',
    'database': 'football_guessing',
    'port': 3306
}

def load_data():
    """从数据库加载数据并进行初步处理"""
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8")

    # 读取赔率数据（仅取赛前6小时数据）
    odds_query = """
    SELECT o.*, m.total_goals 
    FROM t_g_detail_odds o
    JOIN match_result m ON o.match_id = m.match_id
    WHERE 
        o.bookmaker_id IN (1000) AND
        o.update_time_distance >= 360  -- 6小时=360分钟
    """
    df = pd.read_sql(odds_query, engine)

    # 将进球数转换为分类标签（0-7对应，7+合并到7）
    df['total_goals'] = df['total_goals'].apply(lambda x: min(x, 7))
    return df


def feature_engineering(df):
    """高级特征工程"""

    # 1. 机构特征分组计算
    bookmaker_features = []
    for bookmaker in [1000]:
        group = df[df['bookmaker_id'] == bookmaker].groupby('match_id')

        # 每个机构的特征
        features = pd.DataFrame({
            f'bookmaker_{bookmaker}_last_odds': group['odds_value'].last(),
            f'bookmaker_{bookmaker}_change_count': group['change_status'].apply(lambda x: (x != 0).sum()),
            f'bookmaker_{bookmaker}_kelly_mean': group['kelly_index'].mean(),
            f'bookmaker_{bookmaker}_proba_diff': group['proba'].apply(lambda x: x.iloc[-1] - x.iloc[0]),
            f'bookmaker_{bookmaker}_volatility': group['odds_value'].apply(lambda x: x.pct_change().std())
        })
        bookmaker_features.append(features)

    # 2. 时间维度特征
    time_features = df.groupby('match_id').agg({
        'update_time_distance': ['max', 'min', 'std'],
        'this_time': 'max'
    })
    time_features.columns = ['_'.join(col) for col in time_features.columns]

    # 3. 交叉机构特征
    cross_features = pd.DataFrame({
        'odds_range': df.groupby('match_id')['odds_value'].apply(lambda x: x.max() - x.min()),
        'kelly_range': df.groupby('match_id')['kelly_index'].apply(lambda x: x.max() - x.min())
    })

    # 合并所有特征
    feature_df = pd.concat([pd.concat(bookmaker_features, axis=1), time_features, cross_features], axis=1)

    # 添加标签
    labels = df.groupby('match_id')['total_goals'].first()
    return feature_df.join(labels).dropna()


def build_ensemble_model():
    """构建集成模型"""
    # 第一层模型
    rf = RandomForestClassifier(n_estimators=300,
                                max_depth=8,
                                class_weight='balanced',
                                random_state=42)

    lgbm = LGBMClassifier(n_estimators=500,
                          learning_rate=0.05,
                          max_depth=5,
                          objective='multiclass',
                          num_class=8,
                          random_state=42)

    # 第二层元分类器
    meta_model = LGBMClassifier(n_estimators=200,
                                learning_rate=0.1,
                                max_depth=3)

    return StackingClassifier(
        estimators=[('rf', rf), ('lgbm', lgbm)],
        final_estimator=meta_model,
        stack_method='predict_proba',
        passthrough=True
    )


def train_model(X, y):
    """模型训练与调优"""
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 模型参数网格搜索
    param_grid = {
        'rf__max_depth': [6, 8],
        'lgbm__num_leaves': [31, 63],
        'final_estimator__learning_rate': [0.05, 0.1]
    }

    model = build_ensemble_model()
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    # 最佳模型评估
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"Best Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    return best_model


def main():
    # 数据加载与处理
    df = load_data()
    feature_df = feature_engineering(df)

    # 准备训练数据
    X = feature_df.drop('total_goals', axis=1)
    y = feature_df['total_goals']

    # 训练模型
    model = train_model(X, y)

    # 保存模型
    joblib.dump(model, 'goal_prediction_model.pkl')


if __name__ == "__main__":
    main()