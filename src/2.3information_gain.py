import pandas as pd
import numpy as np
from math import log2

# Load the dataset
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


# Calculate entropy
def entropy(data):
    labels = data['class'].value_counts(normalize=True)
    return -sum(labels * np.log2(labels))


# Calculate information gain of features
def information_gain(data, feature):
    # Initial entropy of the data set
    total_entropy = entropy(data)

    # Calculate the conditional entropy of a feature
    values = data[feature].value_counts(normalize=True)
    conditional_entropy = sum(values[val] * entropy(data[data[feature] == val]) for val in values.index)

    # Information gain = total entropy - conditional entropy
    return total_entropy - conditional_entropy


# Calculate the information gain for each feature
features = data.columns.drop('class')
info_gain = {feature: information_gain(data, feature) for feature in features}


# Output the information gain of each feature
print('the information gain calculated for each attribute is as follows:')
for feature, gain in info_gain.items():
    print(f"Information Gain for {feature}: {gain:.3f}")

# Load the dataset(narrow)
data = pd.DataFrame({
    'class': ['poisonous', 'poisonous', 'poisonous', 'poisonous', 'edible', 'poisonous', 'poisonous', 'poisonous'],
    'cap-shape': ['convex', 'convex', 'convex', 'convex', 'sunken', 'convex', 'convex', 'convex'],
    'cap-color': ['brown', 'white', 'white', 'white', 'gray', 'brown', 'white', 'brown'],
    'gill-size': ['narrow', 'narrow', 'narrow', 'narrow', 'narrow', 'narrow', 'narrow', 'narrow'],
    'gill-color': ['black', 'brown', 'pink', 'black', 'black', 'brown', 'brown', 'black'],
    'stalk-shape': ['enlarging', 'enlarging', 'enlarging', 'enlarging', 'enlarging', 'enlarging', 'enlarging',
                    'enlarging'],
    'spore-print-color': ['black', 'black', 'black', 'brown', 'brown', 'black', 'brown', 'brown']
})

def entropy(data):
    labels = data['class'].value_counts(normalize=True)
    return -sum(labels * np.log2(labels))

def information_gain(data, feature):
    total_entropy = entropy(data)

    #Calculate the conditional entropy of a feature
    values = data[feature].value_counts(normalize=True)
    conditional_entropy = sum(values[val] * entropy(data[data[feature] == val]) for val in values.index)

    return total_entropy - conditional_entropy


# Calculate the information gain for each feature
features = data.columns.drop('class')
info_gain = {feature: information_gain(data, feature) for feature in features}

print('Based on gill-size,the information gain calculated for each attribute is as follows:')
# Output the information gain of each feature
for feature, gain in info_gain.items():
    print(f"Information Gain for {feature}: {gain:.3f}")
