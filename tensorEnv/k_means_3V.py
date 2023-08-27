import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Heart_Attack.csv", sep=",")
print(df)
df.dropna(inplace=True)

# Remove the 'class' column from the DataFrame (K Means is unsupervised algorithm, we don't need to label)
df.drop('class', axis=1, inplace=True)
print(df)

print(df.describe())
# Data transform standardization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# round two decimals after comma
df_scaled = np.round(df_scaled, 2)
print(df_scaled)

# in case we need to get some of the features, better to use PCA and reduce the dimensionality
selected_columns = ["age", "troponin", "glucose", "pressurehight"]
df_selected = df[selected_columns]


# Apply PCA for dimensionality reduction from 4 dimension to two dimension
pca = PCA(n_components=2)  # Choose the number of components you want
df_pca = pca.fit_transform(df_scaled)

# Apply KMeans clustering with the desired number of clusters
num_clusters = 3  # You can specify the number of clusters you want
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
kmeans.fit(df_pca)

# Get the cluster centers and labels
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

# Visualize the clusters
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title("KMeans Clustering with PCA")
plt.show()


# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
ax.scatter(df_scaled[:, 0], df_scaled[:, 1], df_scaled[:, 2], c=cluster_labels, cmap='viridis')


# Set labels for the axes
ax.set_xlabel("Age")
ax.set_ylabel("troponin")
ax.set_zlabel("glucose")

# Add a legend
for i in range(num_clusters):
    ax.scatter([], [], [], c=plt.cm.viridis(i / num_clusters), label=f"Cluster {i + 1}")

ax.legend()

plt.title("KMeans Clustering in 3D")
plt.title("KMeans Clustering in 3D Axis: X Age, Y Troponin, Z Glucose")
# Show the 3D plot
plt.show()