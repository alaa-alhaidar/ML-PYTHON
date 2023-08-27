import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

df = pd.read_csv("processed_data.csv", sep=",")

df.dropna(inplace=True)

print(df.describe())

# get the name of the attributes
print(df.columns)
# Data transform standardization

# select the column with numeric value
selected_columns = ["price", "number_of_reviews", "rating"]
df_selected = df[selected_columns]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)
df_scaled = np.round(df_scaled, 2)

# Apply PCA for dimensionality reduction from 3 dimension to two dimension
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

cmpn = pca.components_
cmpn_df = pd.DataFrame(cmpn, columns=["PC1","PC2","PC3"], index=["price",  "rating"])
print(cmpn_df)

variance_explained = pca.explained_variance_
print("Variance explained by each component:", variance_explained)

variance_explained_ratio = pca.explained_variance_ratio_
print("Percentage of variance explained by each component:", variance_explained_ratio)

pca_score = pca.transform(df_scaled)
pca_score_df = pd.DataFrame(pca_score, columns=["PC1","PC2"])
print(pca_score_df)


# Apply KMeans clustering with the desired number of clusters
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
kmeans.fit(df_pca)

# Get the cluster centers and labels
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

# Visualize the clusters
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.title("KMeans Clustering with PCA")
plt.show()

# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
ax.scatter(df_scaled[:, 0], df_scaled[:, 1], df_scaled[:, 2], c=cluster_labels, cmap='viridis')

# Set labels for the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# Add a legend
for i in range(num_clusters):
    ax.scatter([], [], [], c=plt.cm.viridis(i / num_clusters), label=f"Cluster {i + 1}")

ax.legend()
plt.title("KMeans Clustering in 3D")
plt.show()

# Inverse transform the cluster centers back to the original feature space
cluster_centers_original = pca.inverse_transform(cluster_centers)

# Create a DataFrame to display the cluster centers in the original feature space
cluster_centers_df = pd.DataFrame(cluster_centers_original, columns=df_selected.columns)

# Print the cluster centers for each cluster
# magnitude in each cluster,  The higher the value of a feature for a cluster,
# the more that feature contributes to that cluster's characteristics.
for i in range(num_clusters):
    print(f"Cluster {i + 1} Center:")
    print(cluster_centers_df.iloc[i])
    print("\n")
