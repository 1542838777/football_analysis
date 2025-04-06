import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from service.FindBestFactorCombo2 import numeric_cols, df

# 使用决策树自动发现交互作用
dt_model = DecisionTreeClassifier(
    max_depth=3,  # 控制复杂度
    min_samples_leaf=50,  # 防止过拟合
    criterion='gini'
)
dt_model.fit(df[numeric_cols], df['spf_result'])

# 可视化决策树
plt.figure(figsize=(20, 24))
plot_tree(dt_model,
          feature_names=numeric_cols,
          class_names=['负', '平', '胜'],
          filled=True,
          rounded=True)
plt.show()


# 解析树结构中的交互规则
def extract_tree_rules(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]

    rules = []

    def recurse(node, depth, current_rule):
        if left[node] != right[node]:  # 非叶节点
            rule = f"{features[node]} <= {threshold[node]:.2f}"
            recurse(left[node], depth + 1, current_rule + [rule])
            rule = f"{features[node]} > {threshold[node]:.2f}"
            recurse(right[node], depth + 1, current_rule + [rule])
        else:  # 叶节点
            class_prob = tree.tree_.value[node][0]
            dominant_class = np.argmax(class_prob)
            if len(current_rule) >= 2:  # 至少两个条件的才算交互
                rules.append({
                    'rule': ' AND '.join(current_rule),
                    'class': dominant_class,
                    'confidence': class_prob[dominant_class] / class_prob.sum()
                })

    recurse(0, 1, [])
    return pd.DataFrame(rules)


# 获取并展示有效规则
interaction_rules = extract_tree_rules(dt_model, numeric_cols)
print("\n自动发现的交互规则：")
print(interaction_rules.sort_values('confidence', ascending=False).head(1000))


# 新增分组功能
def analyze_group(group_df, group_name, numeric_cols):
    # 训练模型
    dt_model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=30,
        criterion='gini'
    )
    dt_model.fit(group_df[numeric_cols], group_df['spf_result'])

    # 保存树可视化到文件
    plt.figure(figsize=(20, 24))
    plot_tree(dt_model,
              feature_names=numeric_cols,
              class_names=['负', '平', '胜'],
              filled=True,
              rounded=True)
    plt.savefig(f'decision_tree_{group_name}.png')  # 按组名保存图片
    plt.close()

    # 提取规则并添加组标识
    rules = extract_tree_rules(dt_model, numeric_cols)
    rules['group'] = group_name  # 添加分组列
    return rules


# 使用示例（假设按season列分组）
all_rules = pd.DataFrame()
group_column = 'league_id'  # 替换为实际的分组字段名

for group_name, group_df in df.groupby(group_column):
    # 过滤样本量不足的组（至少需要2*min_samples_leaf）
    if len(group_df) < 2 * 30:
        print(f"跳过分组 {group_name}，样本量不足")
        continue

    try:
        group_rules = analyze_group(group_df, group_name, numeric_cols)
        all_rules = pd.concat([all_ru2les, group_rules], ignore_index=True)
    except Exception as e:
        print(f"分组 {group_name} 处理失败: {str(e)}")

# 展示所有分组的规则（按置信度排序）
print("\n所有分组的交互规则：")
print(all_rules.sort_values(['group', 'confidence'], ascending=[True, False]))

# 保存结果到CSV
all_rules.to_csv('interaction_rules_by_group.csv', index=False)