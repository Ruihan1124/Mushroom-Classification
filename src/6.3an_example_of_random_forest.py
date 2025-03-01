import csv
import random
import math
from collections import Counter

# 导入数据
def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        headers = data[0]  # 获取标题行
        data = data[1:]  # 去除标题行
        for row in data:
            for i in range(len(row) - 1):  # 忽略标签列，处理特征
                row[i] = float(row[i]) if row[i].replace('.', '', 1).isdigit() else row[i]
            row[-1] = 1 if row[-1] == "p" else 0  # 将标签转换为1和0
        return headers, data

# 划分数据集
def split_data(data, ratio=0.7):
    random.shuffle(data)  # 随机打乱数据顺序
    train_size = int(len(data) * ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# 定义节点分裂
def split_node(data, feature_index, threshold):
    left = [row for row in data if row[feature_index] < threshold]
    right = [row for row in data if row[feature_index] >= threshold]
    return left, right

# 计算基尼不纯度
def gini_impurity(data):
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    impurity = 1.0 - sum((count / len(data)) ** 2 for count in label_counts.values())
    return impurity

# 获取最佳分割点
def get_best_split(data, num_features):
    best_feature, best_threshold, best_impurity, best_splits = None, None, float('inf'), None
    features = random.sample(range(len(data[0]) - 1), num_features)
    for feature_index in features:
        thresholds = set(row[feature_index] for row in data)
        for threshold in thresholds:
            left, right = split_node(data, feature_index, threshold)
            if not left or not right:
                continue
            p_left = len(left) / len(data)
            p_right = len(right) / len(data)
            impurity = p_left * gini_impurity(left) + p_right * gini_impurity(right)
            if impurity < best_impurity:
                best_feature, best_threshold, best_impurity, best_splits = feature_index, threshold, impurity, (left, right)
    if best_splits is not None:
        return best_feature, best_threshold, best_splits
    else:
        return None  # 如果没有有效的分割则返回 None

# 创建节点
def build_tree(data, max_depth, min_size, depth=0, num_features=None):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels) or depth >= max_depth or len(data) <= min_size:
        return Counter(labels).most_common(1)[0][0]

    split = get_best_split(data, num_features)
    if split is None:  # 检查是否找到有效的分割
        return Counter(labels).most_common(1)[0][0]

    feature, threshold, (left, right) = split
    if not left or not right:
        return Counter(labels).most_common(1)[0][0]

    node = {'feature': feature, 'threshold': threshold, 'left': None, 'right': None}
    node['left'] = build_tree(left, max_depth, min_size, depth + 1, num_features)
    node['right'] = build_tree(right, max_depth, min_size, depth + 1, num_features)
    return node

# 随机森林构建
def build_random_forest(train_data, num_trees, max_depth, min_size, num_features):
    forest = []
    for _ in range(num_trees):
        sample = [random.choice(train_data) for _ in range(len(train_data))]
        tree = build_tree(sample, max_depth, min_size, num_features=num_features)
        forest.append(tree)
    return forest

# 预测
def predict(tree, row):
    if isinstance(tree, dict):
        if row[tree['feature']] < tree['threshold']:
            return predict(tree['left'], row)
        else:
            return predict(tree['right'], row)
    else:
        return tree  # 如果是叶节点，直接返回类别

def random_forest_predict(forest, row):
    predictions = [predict(tree, row) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]

# 模型评估
def evaluate(forest, test_data):
    correct = sum(1 for row in test_data if random_forest_predict(forest, row) == row[-1])
    return correct / len(test_data)

# 使用代码
filename = r'C:\Users\pc\Desktop\4270assignment\dataset\mushrooms-full data.csv'  # 请替换为你的CSV文件路径
headers, data = load_data(filename)
train_data, test_data = split_data(data)

# 超参数调整
num_trees = 150
max_depth = 20
min_size = 15
num_features = int(math.sqrt(len(data[0]) - 1))

# 训练模型
forest = build_random_forest(train_data, num_trees, max_depth, min_size, num_features)

# 测试模型
accuracy = evaluate(forest, test_data)
print(f'Random Forest Accuracy: {accuracy * 100:.2f}%')
