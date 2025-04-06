import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import warnings


class RuleExtractor:
    def __init__(self, df, numeric_cols, target_col,
                 class_names):
        """
        初始化规则提取器

        参数：
        df: 包含特征和目标变量的DataFrame
        numeric_cols: 用于建模的数值特征列名列表
        target_col: 目标变量列名（默认'spf_result'）
        class_names: 目标类别标签（默认['负', '平', '胜']）
        """
        self.df = df.copy()
        self.numeric_cols = numeric_cols
        self.target_col = target_col
        self.class_names = class_names
        self._validate_inputs()

    def _validate_inputs(self):
        """验证输入数据有效性"""
        missing_cols = [col for col in self.numeric_cols + [self.target_col]
                        if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"缺失必要列：{missing_cols}")

        if len(self.class_names) != len(self.df[self.target_col].unique()):
            raise ValueError("类别标签数量与目标变量类别数不匹配")

    @staticmethod
    def extract_tree_rules(tree, feature_names):
        """解析决策树规则"""
        left = tree.tree_.children_left
        right = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features = [feature_names[i] for i in tree.tree_.feature]

        rules = []

        def recurse(node, current_rule):
            if left[node] != right[node]:  # 非叶节点
                rule = f"{features[node]} <= {threshold[node]:.2f}"
                recurse(left[node], current_rule + [rule])
                rule = f"{features[node]} > {threshold[node]:.2f}"
                recurse(right[node], current_rule + [rule])
            else:  # 叶节点
                class_prob = tree.tree_.value[node][0]
                dominant_class = np.argmax(class_prob)
                if len(current_rule) >= 2:  # 至少两个条件
                    rules.append({
                        'rule': ' AND '.join(current_rule),
                        'class': dominant_class,
                        'confidence': class_prob[dominant_class] / class_prob.sum()
                    })

        recurse(0, [])
        return pd.DataFrame(rules)

    def analyze_group(self, typeName, group_df, group_name,
                      model_params=None,
                      save_plot=True,
                      plot_dir='./'):
        """
        分析单个分组

        参数：
        group_df: 分组数据
        group_name: 分组标识名称
        model_params: 模型参数字典（默认None使用预设参数）
        save_plot: 是否保存决策树图片（默认True）
        plot_dir: 图片保存路径（默认当前目录）
        """
        # 设置默认模型参数
        default_params = {
            'max_depth': 3,
            'min_samples_leaf': 16,
            'criterion': 'gini'
        }
        if model_params:
            default_params.update(model_params)

        try:
            # 训练模型
            dt_model = DecisionTreeClassifier(**default_params)
            dt_model.fit(group_df[self.numeric_cols], group_df[self.target_col])

            # 保存决策树可视化
            if save_plot:
                plt.figure(figsize=(20, 24))
                plot_tree(dt_model,
                          feature_names=self.numeric_cols,
                          class_names=self.class_names,
                          filled=True,
                          rounded=True)
                plt.savefig(f'{typeName}_decision_tree_{group_name}.png',

                            bbox_inches='tight')

                plt.close()

            # 提取规则
            rules = self.extract_tree_rules(dt_model, self.numeric_cols)
            rules['group'] = group_name
            return rules

        except Exception as e:
            warnings.warn(f"分组 {group_name} 处理失败: {str(e)}")
            return pd.DataFrame()

    def run_group_analysis(self, typeName, group_column='league_id',
                           min_samples=16,
                           **analyze_kwargs):
        """
        执行分组分析

        参数：
        group_column: 分组列名（默认'league_id'）
        min_samples: 最小样本量阈值（默认60=2*min_samples_leaf）
        analyze_kwargs: 传递给analyze_group的参数
        """
        all_rules = pd.DataFrame()

        for group_name, group_df in self.df.groupby(group_column):
            # 过滤小样本分组
            if len(group_df) < min_samples:
                print(f"跳过分组 {group_name}，样本量不足（{len(group_df)} < {min_samples}）")
                continue

            try:
                group_rules = self.analyze_group(
                    typeName,
                    group_df=group_df,
                    group_name=group_name,
                    **analyze_kwargs
                )
                all_rules = pd.concat([all_rules, group_rules], ignore_index=True)
            except Exception as e:
                print(f"分组 {group_name} 处理异常: {str(e)}")
                continue

        # 后处理
        if not all_rules.empty:
            all_rules['class'] = all_rules['class'].astype(int)
            all_rules['class_label'] = all_rules['class'].apply(
                lambda x: self.class_names[x]
            )
            all_rules = all_rules.sort_values(['confidence', 'group'],
                                              ascending=[False, False])
        return all_rules

    def save_rules(self, typeName, rules_df, file_path='interaction_rules_by_group.csv'):
        """保存规则到CSV"""
        file_path = f'{typeName}_interaction_rules_by_group.csv'
        if not rules_df.empty:
            rules_df.to_csv(file_path, index=False)
            print(f"规则已保存至：{file_path}")
        else:
            print("无有效规则可保存")


def get_base_name_europe_groups():
    typeName = 'europeGroup'
    numeric_cols = ['first_win_odds_of24h',
                    'first_draw_odds_of24h',
                    'first_lose_odds_of24h',
                    'last_win_odds_of24h',
                    'last_draw_odds_of24h',
                    'last_lose_odds_of24h',
                    'win_odds_of24h_drop_num',
                    'draw_odds_of24h_drop_num',
                    'lose_odds_of24h_drop_num',
                    'win_odds_of24h_up_num',
                    'draw_odds_of24h_up_num',
                    'lose_odds_of24h_up_num',
                    'win_odds_of24h_change_num',
                    'draw_odds_of24h_change_num',
                    'lose_odds_of24h_change_num',
                    'win_odds_of24h_stable_num',
                    'draw_odds_of24h_stable_num',
                    'lose_odds_of24h_stable_num',
                    'total_num_of24h',
                    'win_odds_of24h_change_num_rank',
                    'draw_odds_of24h_change_num_rank',
                    'lose_odds_of24h_change_num_rank',
                    'win_odds_of24h_stable_num_rank',
                    'draw_odds_of24h_stable_num_rank',
                    'lose_odds_of24h_stable_num_rank',
                    'win_odds_of24h_final_change',
                    'draw_odds_of24h_final_change',
                    'lose_odds_of24h_final_change',
                    'win_odds_of24h_delta',
                    'draw_odds_of24h_delta',
                    'lose_odds_of24h_delta'

                    ]  # 补充其他实际字段名
    # 这里可以添加任何需要的逻辑来处理 numeric_cols 和 df
    df = pd.read_csv('D:/lqs/life/football/file/football_guessing_match_factor.csv')  # 加载数据
    target_col = 'spf_result'
    class_names = ['负', '平', '胜']
    return typeName, numeric_cols, df, target_col, class_names


def get_base_name_ah_12h():
    typeName = '亚盘明升12h凯利'
    numeric_cols = [
        '欧亚盘口差',
        '欧赔让球数',

        '初组上盘赔率',
        '初组下盘赔率',
        '初组上盘凯利',
        '初组下盘凯利',
        '初组盘口中值',
        '初组凯利中值',

        '二组上盘赔率',
        '二组下盘赔率',
        '二组上盘凯利',
        '二组下盘凯利',
        '二组盘口中值',
        '二组凯利中值',
        '初组上盘赔率变化',
        '初组下盘赔率变化',
        '二组上盘赔率变化',
        '二组下盘赔率变化',
        '盘口变化',
        '上盘凯利变化',
        '下盘凯利变化',
        '盘口中值变化'
    ]  # 补充其他实际字段名
    # 这里可以添加任何需要的逻辑来处理 numeric_cols 和 df
    df = pd.read_csv('D:/lqs/life/football/file/亚盘明升12h两组结合欧让球数量分析.csv')  # 加载数据
    target_col = '亚盘结果'  # -1:上盘 0走水 1：下盘
    class_names = ['上盘', '走水', '下盘']  # -1:上盘 0走水 1：下盘
    return typeName, numeric_cols, df, target_col, class_names


def get_base_europe_ah_handicap_diff_12h():
    typeName = '欧亚盘口差让球赛果分析'
    numeric_cols = [
        '欧亚盘口差',
        '欧赔让球数',

        '初组上盘赔率',
        '初组下盘赔率',
        '初组上盘凯利',
        '初组下盘凯利',
        '初组盘口中值',
        '初组凯利中值',

        '二组上盘赔率',
        '二组下盘赔率',
        '二组上盘凯利',
        '二组下盘凯利',
        '二组盘口中值',
        '二组凯利中值',
        '初组上盘赔率变化',
        '初组下盘赔率变化',
        '二组上盘赔率变化',
        '二组下盘赔率变化',
        '盘口变化',
        '上盘凯利变化',
        '下盘凯利变化',
        '盘口中值变化'
    ]  # 补充其他实际字段名
    # 这里可以添加任何需要的逻辑来处理 numeric_cols 和 df
    df = pd.read_csv('D:/lqs/life/football/file/亚盘明升12h两组结合欧让球数量分析.csv')  # 加载数据
    target_col = '欧赔让盘结果'  # -1:上盘 0走水 1：下盘
    class_names = ['负', '平', '胜']  #
    return typeName, numeric_cols, df, target_col, class_names


def get_base_欧亚让球差异合并欧赔12h24h变化组数():
    typeName = '欧亚让球差异合并欧赔12h24h变化组数'
    numeric_cols = [
        '欧亚盘口差',
        '欧赔让球数',

        '初组上盘赔率',
        '初组下盘赔率',
        '初组上盘凯利',
        '初组下盘凯利',
        '初组盘口中值',
        '初组凯利中值',

        '二组上盘赔率',
        '二组下盘赔率',
        '二组上盘凯利',
        '二组下盘凯利',
        '二组盘口中值',
        '二组凯利中值',
        '初组上盘赔率变化',
        '初组下盘赔率变化',
        '二组上盘赔率变化',
        '二组下盘赔率变化',
        '盘口变化',
        '上盘凯利变化',
        '下盘凯利变化',
        '盘口中值变化',
        'first_win_odds_of24h',
        'first_draw_odds_of24h',
        'first_lose_odds_of24h',
        'last_win_odds_of24h',
        'last_draw_odds_of24h',
        'last_lose_odds_of24h',
        'win_odds_of24h_drop_num',
        'draw_odds_of24h_drop_num',
        'lose_odds_of24h_drop_num',
        'win_odds_of24h_up_num',
        'draw_odds_of24h_up_num',
        'lose_odds_of24h_up_num',
        'win_odds_of24h_change_num',
        'draw_odds_of24h_change_num',
        'lose_odds_of24h_change_num',
        'win_odds_of24h_stable_num',
        'draw_odds_of24h_stable_num',
        'lose_odds_of24h_stable_num',
        'total_num_of24h',
        'win_odds_of24h_change_num_rank',
        'draw_odds_of24h_change_num_rank',
        'lose_odds_of24h_change_num_rank',
        'win_odds_of24h_stable_num_rank',
        'draw_odds_of24h_stable_num_rank',
        'lose_odds_of24h_stable_num_rank',
        'win_odds_of24h_final_change',
        'draw_odds_of24h_final_change',
        'lose_odds_of24h_final_change',
        'win_odds_of24h_delta',
        'draw_odds_of24h_delta',
        'lose_odds_of24h_delta'
    ]  # 补充其他实际字段名

    # 这里可以添加任何需要的逻辑来处理 numeric_cols 和 df


    df = pd.read_csv('D:/lqs/life/football/file/欧亚让球差异合并欧赔12h24h变化组数.csv')  # 加载数据
    target_col = '欧赔让盘结果'  # -1:上盘 0走水 1：下盘
    class_names = ['负', '平', '胜']  #
    return typeName, numeric_cols, df, target_col, class_names

# 使用示例
if __name__ == "__main__":
    # 从外部导入数据（示例）

    # 调用方法并接收返回值
    # typeName,numeric_cols, df,target_col,class_names = get_base_name_europe_groups()

    # typeName, numeric_cols, df, target_col, class_names = get_base_name_ah_12h()
    typeName, numeric_cols, df, target_col, class_names = get_base_欧亚让球差异合并欧赔12h24h变化组数()
    # typeName, numeric_cols, df, target_col, class_names = get_base_europe_ah_handicap_diff_12h()
    # 初始化规则提取器
extractor = RuleExtractor(
    df=df,
    numeric_cols=numeric_cols,
    target_col=target_col,
    class_names=class_names
)

# 执行分组分析
rules = extractor.run_group_analysis(
    typeName,
    group_column='league_id',
    min_samples=16,
    model_params={'max_depth': 9, 'min_samples_leaf': 16},
    save_plot=True,
    plot_dir='./decision_trees'
)

# 展示并保存结果
print("\n所有分组的交互规则：")
print(rules.head())

# reules 每行rule字段 追加上 and league_id = group的值
rules['rule'] = rules.apply(lambda row: row['rule'] + ' and league_id = ' + str(row['group']), axis=1)
extractor.save_rules(typeName, rules)
