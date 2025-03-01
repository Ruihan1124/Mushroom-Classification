import pandas as pd
import numpy as np


# Define the calculation of entropy
def entropy(data):
    labels = data['class'].value_counts(normalize=True)
    return -sum(labels * np.log2(labels))


# Calculate information gain
def information_gain(data, feature):
    total_entropy = entropy(data)
    values = data[feature].value_counts(normalize=True)
    conditional_entropy = sum(values[val] * entropy(data[data[feature] == val]) for val in values.index)
    return total_entropy - conditional_entropy


# Select the best segmentation attributes
def best_feature_to_split(data, features):
    info_gains = {feature: information_gain(data, feature) for feature in features}
    best_feature = max(info_gains, key=info_gains.get)
    return best_feature, info_gains[best_feature]


# Recursive function to create decision tree
def create_decision_tree(data, features):
    # If all instances belong to the same class, return that class
    if len(data['class'].unique()) == 1:
        return data['class'].iloc[0]

    # If no features are available, return the class with the most
    if not features:
        return data['class'].mode().iloc[0]

    # Select the attribute with the highest information gain for segmentation
    best_feature, best_info_gain = best_feature_to_split(data, features)

    # Create tree nodes
    tree = {best_feature: {}}

    # Traverse each value of the attribute, recursively creating a subtree for each subset
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        if subset.empty:
            # If the subset is empty, return the class with the largest number in the current dataset.
            tree[best_feature][value] = data['class'].mode().iloc[0]
        else:
            # Recursively create subtrees
            subtree = create_decision_tree(subset, [feat for feat in features if feat != best_feature])
            tree[best_feature][value] = subtree

    return tree


# test data set
data = pd.DataFrame({
    'class': ['poisonous', 'edible', 'edible', 'poisonous', 'edible', 'edible', 'edible', 'edible', 'poisonous',
              'edible',
              'edible', 'edible', 'edible', 'poisonous', 'edible', 'edible', 'edible', 'poisonous', 'poisonous',
              'poisonous'],
    'cap-shape': ['convex', 'convex', 'bell', 'convex', 'convex', 'convex', 'bell', 'bell', 'convex', 'bell',
                  'convex', 'convex', 'bell', 'convex', 'convex', 'sunken', 'flat', 'convex', 'convex', 'convex'],
    'cap-color': ['brown', 'yellow', 'white', 'white', 'gray', 'yellow', 'white', 'white', 'white', 'yellow',
                  'yellow', 'yellow', 'yellow', 'white', 'brown', 'gray', 'white', 'brown', 'white', 'brown'],
    'gill-size': ['narrow', 'broad', 'broad', 'narrow', 'broad', 'broad', 'broad', 'broad', 'narrow', 'broad',
                  'broad', 'broad', 'broad', 'narrow', 'broad', 'narrow', 'broad', 'narrow', 'narrow', 'narrow'],
    'gill-color': ['black', 'black', 'brown', 'brown', 'black', 'brown', 'gray', 'brown', 'pink', 'gray',
                   'gray', 'brown', 'white', 'black', 'brown', 'black', 'black', 'brown', 'brown', 'black'],
    'stalk-shape': ['enlarging', 'enlarging', 'enlarging', 'enlarging', 'tapering', 'enlarging', 'enlarging',
                    'enlarging', 'enlarging', 'enlarging',
                    'enlarging', 'enlarging', 'enlarging', 'enlarging', 'tapering', 'enlarging', 'tapering',
                    'enlarging', 'enlarging', 'enlarging'],
    'spore-print-color': ['black', 'brown', 'brown', 'black', 'brown', 'black', 'black', 'brown', 'black', 'black',
                          'brown', 'black', 'brown', 'brown', 'black', 'brown', 'brown', 'black', 'brown', 'brown']
})

# Execute decision tree algorithm
features = list(data.columns.drop('class'))
decision_tree = create_decision_tree(data, features)

# Output decision tree
print("决策树结构：", decision_tree)