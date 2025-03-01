import math
import pandas as pd

# Load dataset from the provided file (recreate the table in Python)
data = pd.DataFrame({
    'id': [2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 18, 16, 17, 20],
    'class': ['edible', 'edible', 'poisonous', 'edible', 'edible', 'poisonous', 'edible', 'edible', 'edible',
              'poisonous', 'poisonous', 'edible', 'edible', 'poisonous'],
    'cap-shape': ['convex', 'bell', 'convex', 'convex', 'bell', 'convex', 'bell', 'convex', 'convex', 'convex',
                  'convex', 'sunken', 'flat', 'convex'],
    'cap-color': ['yellow', 'white', 'white', 'gray', 'white', 'white', 'yellow', 'yellow', 'yellow', 'white', 'brown',
                  'gray', 'white', 'brown'],
    'gill-size': ['broad', 'broad', 'narrow', 'broad', 'broad', 'narrow', 'broad', 'broad', 'broad', 'narrow', 'narrow',
                  'narrow', 'broad', 'narrow'],
    'gill-color': ['black', 'brown', 'brown', 'black', 'brown', 'pink', 'gray', 'gray', 'brown', 'black', 'brown',
                   'black', 'black', 'black'],
    'stalk-shape': ['enlarging', 'enlarging', 'enlarging', 'tapering', 'enlarging', 'enlarging', 'enlarging',
                    'enlarging', 'enlarging', 'enlarging', 'enlarging', 'enlarging', 'tapering', 'enlarging'],
    'spore-print-color': ['brown', 'brown', 'black', 'brown', 'brown', 'black', 'black', 'brown', 'black', 'brown',
                          'black', 'brown', 'brown', 'brown']
})


# Calculate entropy
def entropy(data):
    labels = data['class'].value_counts()
    total = len(data)
    ent = 0
    for count in labels:
        prob = count / total
        ent -= prob * math.log2(prob)
    return ent


# Calculate information gain
def info_gain(data, attribute):
    total_entropy = entropy(data)
    values = data[attribute].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[attribute] == value]
        prob = len(subset) / len(data)
        weighted_entropy += prob * entropy(subset)
    gain = total_entropy - weighted_entropy
    return gain


# Recursively build the decision tree
def build_tree(data, features):
    # Check if all examples have the same class
    if len(data['class'].unique()) == 1:
        return data['class'].iloc[0]

    # If no features are left, return the most common class
    if len(features) == 0:
        return data['class'].mode()[0]

    # Find the best attribute based on information gain
    gains = {feature: info_gain(data, feature) for feature in features}
    best_feature = max(gains, key=gains.get)

    # Create a subtree for each value of the best feature
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = build_tree(subset, remaining_features)
        tree[best_feature][value] = subtree

    return tree


# Define features and build the tree
features = ['cap-shape', 'cap-color', 'gill-size', 'gill-color', 'stalk-shape', 'spore-print-color']
decision_tree = build_tree(data, features)

# Print the resulting decision tree
import pprint

pprint.pprint(decision_tree)
