from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Keep track of the selected features and the best accuracy
selected_features = []
unordered_features = [*range(X.shape[1])]
incrementalAccuracies = []
# Loop through each feature and evaluate its performance with KNN
while(len(unordered_features) > 0):
    accuracies = []

    for i in unordered_features:
        # Create a temporary list of the selected features
        temp = selected_features.copy()
        
        # Add the current feature to the list
        temp.append(i)
        
        # Create a KNN model with K=1 using the selected features
        knn = KNeighborsClassifier(n_neighbors=1)
        
        t = X[:, temp]
        # Use cross-validation to evaluate the model's accuracy
        scores = cross_val_score(knn, X[:, temp], y, cv=5)
        accuracies.append(scores.mean())

        # If the model's accuracy is better than the current best, update the best accuracy and the selected features
    bestId = accuracies.index(max(accuracies))
    incrementalAccuracies.append((max(accuracies)))
    selected_features.append(unordered_features[bestId])  
    unordered_features.remove(unordered_features[bestId])


# Print the selected features
[ print(iris.feature_names[selected_features[id]], incrementalAccuracies[id]) for id in range(len(selected_features))  ]