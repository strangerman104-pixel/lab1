import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = pd.read_csv('clustering_data.csv')[['x', 'y']]

plt.figure(figsize=(6, 5))
plt.scatter(X['x'], X['y'])
plt.title("Data Points Before K-Means")
plt.show()

for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    plt.figure(figsize=(6, 5))
    plt.scatter(X['x'], X['y'], c=labels, cmap='tab10')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='X', s=150, color='black', label='Centroids')
    plt.title(f"K-Means Clustering - {k} Clusters")
    plt.legend()
    plt.show()