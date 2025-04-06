# 第一组数据
# 原始数据（以换行符分隔）
raw_data = """
1
1
0
2
2
0
0
0
0
2

"""

# 将原始数据转换为列表
data1 = [int(num) for num in raw_data.strip().split('\n')]


# 第二组数据（字符串形式）
data2_str = "2202220002"
data2 = [int(char) for char in data2_str]  # 转换为整数列表

# 比对两组数据
comparison_results = []
for i in range(len(data1)):
    if i < len(data2):
        comparison_results.append((data1[i], data2[i], data1[i] == data2[i]))
    else:
        comparison_results.append((data1[i], None, False))  # data2没有对应值

# 输出比对结果
for index, (val1, val2, is_equal) in enumerate(comparison_results):
    print(f"Index {index}: data1 = {val1}, data2 = {val2}, Equal = {is_equal}")