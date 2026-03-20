import pandas as pd

data = {
    'Age': ['Young', 'Young', 'Middle', 'Old'],
    'BrowsingTime': ['High', 'Low', 'High', 'Low'],
    'PastPurchase': ['Yes', 'No', 'Yes', 'No'],
    'Buy': ['Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)
print(df)
import math
from collections import Counter

# Entropy calculation
def entropy(target):
    total = len(target)
    counts = Counter(target)
    return -sum((count/total) * math.log2(count/total) for count in counts.values())

# Information Gain
def info_gain(df, feature, target='Buy'):
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    
    for val in values:
        subset = df[df[feature] == val]
        weighted_entropy += (len(subset)/len(df)) * entropy(subset[target])
    
    return total_entropy - weighted_entropy

# ID3 Algorithm
def id3(df, features, target='Buy'):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    
    if len(features) == 0:
        return df[target].mode()[0]
    
    gains = {feature: info_gain(df, feature) for feature in features}
    best_feature = max(gains, key=gains.get)
    
    tree = {best_feature: {}}
    
    for val in df[best_feature].unique():
        subset = df[df[best_feature] == val]
        subtree = id3(subset, [f for f in features if f != best_feature])
        tree[best_feature][val] = subtree
    
    return tree

# Build tree
features = ['Age', 'BrowsingTime', 'PastPurchase']
tree = id3(df, features)
print("Decision Tree:", tree)
