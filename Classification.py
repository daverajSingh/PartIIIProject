import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

data = np.load('system/features.npy', allow_pickle=True) 

print("loaded data")

labels = np.array([item[0] for item in data])  # Extracting the class IDs
features = np.stack([item[1] for item in data])  # Stack arrays to form a 2D feature matrix

# Initialize KFold Cross-Validation
KFold = StratifiedKFold(n_splits=4, random_state=1, shuffle=True)
# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=1, metric='hamming')  

# To collect the true and predicted labels
y_true, y_pred = [], []

print("performing classification")
# Perform LOO cross-validation
for train_index, test_index in KFold.split(features, labels):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Fit KNN classifier
    knn.fit(X_train, y_train)

    # Predict the class for the test set
    prediction = knn.predict(X_test)
    probabilities = knn.predict_proba(X_test)

    # Store results
    y_true.extend(y_test)
    y_pred.extend(prediction)


# Calculate the accuracy
accuracy = accuracy_score(y_true, y_pred, normalize=True)
