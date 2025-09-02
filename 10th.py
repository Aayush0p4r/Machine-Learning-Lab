# Step 1: Import Libraries
import pandas as pd
import numpy as np
import math
from collections import Counter

# Step 2: Prepare Dataset
# Classic Play Tennis dataset
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain', 'Overcast','Sunny','Sunny','Rain','Sunny','Overcast', 'Overcast','Rain'],
    
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool', 'Cool','Mild','Cool','Mild','Mild','Mild', 'Hot','Mild'],
    
    'Humidity': ['High','High','High','High','Normal','Normal', 'Normal','High','Normal','Normal','Normal','High', 'Normal','High'],
    
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong', 'Strong','Weak','Weak','Weak','Strong','Strong', 'Weak','Strong'],
    
    'PlayTennis': ['No','No','Yes','Yes','Yes','No', 'Yes','No','Yes','Yes','Yes','Yes', 'Yes','No']
}

df = pd.DataFrame(data)
print(df)

# Define Entropy Function 
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = 0
    for i in range(len(elements)):
        prob = counts[i] / np.sum(counts)
        entropy_value -= prob * math.log2(prob)
    return entropy_value

# Define Information Gain Function
def info_gain(data, split_attribute, target_attribute):
    # Calculate the total entropy before the split
    total_entropy = entropy(data[target_attribute])
    
    # Get the unique values of the split attribute
    values, counts = np.unique(data[split_attribute], return_counts=True)
    
    # Calculate the weighted entropy after the split
    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[split_attribute] == values[i]]
        weighted_entropy += (counts[i]/np.sum(counts)) * entropy(subset[target_attribute])

    # Information Gain is the difference
    gain = total_entropy - weighted_entropy
    return gain

# Build ID3 Algorithm 
def ID3(data, originaldata, features, target_attribute="PlayTennis", parent_node_class=None):
    # If all target values are the same, return a leaf node
    if len(np.unique(data[target_attribute])) <= 1:
        return np.unique(data[target_attribute])[0]

    # If Dataset is empty, return the class label of the parent node
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute])[np.argmax(np.unique(originaldata[target_attribute], return_counts=True)[1])]
    
    # If the feature space is empty, return the parent node class
    elif len(features) == 0:
        return parent_node_class
    
    # Otherwise, grow the tree
    else:
        parent_node_class = np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]

        # Select feature with maximum Information Gain
        gains = [info_gain(data, feature, target_attribute) for feature in features]

        # Select the feature with the maximum gain
        best_feature = features[np.argmax(gains)]

        # Create a new node for the tree
        tree = Node(name=best_feature)

        # Split the dataset on the best feature
        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value]
            # Recursively grow the tree
            sub_features = [f for f in features if f != best_feature]
            subtree = ID3(subset, originaldata, sub_features, target_attribute, parent_node_class)

            tree[best_feature][value] = subtree

        return tree
