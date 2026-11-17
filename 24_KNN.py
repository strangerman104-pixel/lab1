import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv('iris.data', header=None, names=cols).dropna()

X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

np.random.seed(10)
indices = np.random.permutation(len(X))
split = int(0.70 * len(X))
X_train, y_train = X[indices[:split]], y[indices[:split]]
X_test, y_test = X[indices[split:]], y[indices[split:]]

def knn_predict(X_tr, y_tr, X_te, k):
    preds = []
    for test_pt in X_te:
        dists = [(np.sqrt(np.sum((test_pt - X_tr[i]) ** 2)), y_tr[i]) for i in range(len(X_tr))]
        dists.sort(key=lambda x: x[0])
        nearest = [label for _, label in dists[:k]]
        preds.append(Counter(nearest).most_common(1)[0][0])
    return np.array(preds)

for k in [3, 5, 7, 9]:
    y_pred = knn_predict(X_train, y_train, X_test, k)
    print(f"\nK = {k}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=np.unique(y)))
    print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F-Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))