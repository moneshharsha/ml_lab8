import numpy as np
import pandas as pd

class DecisionTree:
    def _init_(self):
        pass

    def calculate_entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def calculate_information_gain(self, feature_values, labels):
        unique_values, counts = np.unique(feature_values, return_counts=True)
        weighted_entropy = 0
        for value, count in zip(unique_values, counts):
            subset_labels = labels[feature_values == value]
            weighted_entropy += (count / len(feature_values)) * self.calculate_entropy(subset_labels)
        information_gain = self.calculate_entropy(labels) - weighted_entropy
        return information_gain

    def select_root_feature(self, features, labels):
        max_information_gain = -np.inf
        best_feature = None
        for feature in features:
            information_gain = self.calculate_information_gain(features[feature], labels)
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_feature = feature
        return best_feature

class Binning:
    def _init_(self):
        pass

    def equal_width_binning(self, data, num_bins):
        min_val = min(data)
        max_val = max(data)
        bin_width = (max_val - min_val) / num_bins
        bins = [min_val + i * bin_width for i in range(num_bins)]
        bins.append(max_val)  # Include the maximum value in the last bin
        return bins

    def frequency_binning(self, data, num_bins):
        sorted_data = sorted(data)
        bin_size = len(data) // num_bins
        bins = [sorted_data[i * bin_size: (i + 1) * bin_size] for i in range(num_bins)]
        bins[-1].extend(sorted_data[num_bins * bin_size:])  # Add remaining elements to the last bin
        return [bin[0] for bin in bins]  # Return the first element of each bin as bin boundaries

class CustomDecisionTree:
    def _init_(self):
        pass

    def fit(self, X, y):
        self.features = X.columns
        self.labels = y
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        # Implement your decision tree building algorithm here
        pass

    def predict(self, X):
        # Implement your prediction logic using the trained decision tree
        pass


data = pd.read_excel(r"C:\Users\mones\OneDrive\Documents\AISP 1.xlsx")

# Separate features and labels
features = data.drop(columns=['Label'])
labels = data['Label']

# A1: Detecting the root node feature using information gain
decision_tree = DecisionTree()
root_feature = decision_tree.select_root_feature(features, labels)
print("Root feature selected using information gain:", root_feature)

# A2: Converting continuous-valued feature to categorical using binning
binning = Binning()
# Assuming 'ContinuousFeature' is a continuous-valued feature column in your dataset
continuous_feature_values = data['Score'].values  # Example continuous feature values
num_bins = 3  # Number of bins to create
equal_width_bins = binning.equal_width_binning(continuous_feature_values, num_bins)
frequency_bins = binning.frequency_binning(continuous_feature_values, num_bins)

print("Equal width bins:", equal_width_bins)
print("Frequency bins:", frequency_bins)

# A3: Building a custom decision tree module
custom_decision_tree = CustomDecisionTree()
custom_decision_tree.fit(features, labels)