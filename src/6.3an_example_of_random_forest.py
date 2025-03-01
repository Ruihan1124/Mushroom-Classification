import csv
import random
import math
from collections import Counter

# import data
def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        headers = data[0]  # get title
        data = data[1:]  # remove title
        for row in data:
            for i in range(len(row) - 1):  # ignore the title
                row[i] = float(row[i]) if row[i].replace('.', '', 1).isdigit() else row[i]
            row[-1] = 1 if row[-1] == "p" else 0  # change the label to 1 or 0
        return headers, data

# split data
def split_data(data, ratio=0.7):
    random.shuffle(data)  # 随机打乱数据顺序
    train_size = int(len(data) * ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# define the split node
def split_node(data, feature_index, threshold):
    left = [row for row in data if row[feature_index] < threshold]
    right = [row for row in data if row[feature_index] >= threshold]
    return left, right

# calculate the gini impurity
def gini_impurity(data):
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    impurity = 1.0 - sum((count / len(data)) ** 2 for count in label_counts.values())
    return impurity

# get the best split
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
        return None  # if no split is found

# create the tree
def build_tree(data, max_depth, min_size, depth=0, num_features=None):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels) or depth >= max_depth or len(data) <= min_size:
        return Counter(labels).most_common(1)[0][0]

    split = get_best_split(data, num_features)
    if split is None:  # check if no split is found
        return Counter(labels).most_common(1)[0][0]

    feature, threshold, (left, right) = split
    if not left or not right:
        return Counter(labels).most_common(1)[0][0]

    node = {'feature': feature, 'threshold': threshold, 'left': None, 'right': None}
    node['left'] = build_tree(left, max_depth, min_size, depth + 1, num_features)
    node['right'] = build_tree(right, max_depth, min_size, depth + 1, num_features)
    return node

# build the random forest
def build_random_forest(train_data, num_trees, max_depth, min_size, num_features):
    forest = []
    for _ in range(num_trees):
        sample = [random.choice(train_data) for _ in range(len(train_data))]
        tree = build_tree(sample, max_depth, min_size, num_features=num_features)
        forest.append(tree)
    return forest

# predict the class
def predict(tree, row):
    if isinstance(tree, dict):
        if row[tree['feature']] < tree['threshold']:
            return predict(tree['left'], row)
        else:
            return predict(tree['right'], row)
    else:
        return tree  # if it is a leaf node

def random_forest_predict(forest, row):
    predictions = [predict(tree, row) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]

# evaluate the model
def evaluate(forest, test_data):
    correct = sum(1 for row in test_data if random_forest_predict(forest, row) == row[-1])
    return correct / len(test_data)

# use the random forest
filename = r'C:\Users\pc\Desktop\4270assignment\dataset\mushrooms-full data.csv'  # replace with your own path
headers, data = load_data(filename)
train_data, test_data = split_data(data)

# parameters
num_trees = 150
max_depth = 20
min_size = 15
num_features = int(math.sqrt(len(data[0]) - 1))

# train the model
forest = build_random_forest(train_data, num_trees, max_depth, min_size, num_features)

# test the model
accuracy = evaluate(forest, test_data)
print(f'Random Forest Accuracy: {accuracy * 100:.2f}%')
