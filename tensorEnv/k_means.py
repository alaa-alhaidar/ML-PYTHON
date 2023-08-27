import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

df = pd.read_csv("Heart_Attack.csv", sep=",")
print(df.drop_duplicates())
print(df)
df.dropna(inplace=True)

# Remove the 'class' column from the DataFrame
df.drop('class', axis=1, inplace=True)
print(df)

print(df.describe())
# Data transform standardization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
print(df_scaled)


# Assuming 'df' is your original DataFrame containing all the data
selected_columns = ["age", "troponin"]
df_selected = df[selected_columns]

# Data transform standardization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Apply KMeans clustering with the desired number of clusters
num_clusters = 3  # You can specify the number of clusters you want
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
clusters = kmeans.fit(df_scaled)

# Predict the clusters for each data point
cluster_assignments = kmeans.predict(df_scaled)


# Initialize lists for each cluster
cluster_0_data = []
cluster_1_data = []
cluster_2_data = []

# Separate the data points based on cluster assignments
for i, label in enumerate(cluster_assignments):
    if label == 0:
        cluster_0_data.append(df_scaled[i])
    elif label == 1:
        cluster_1_data.append(df_scaled[i])
    elif label == 2:
        cluster_2_data.append(df_scaled[i])

# Convert lists to numpy arrays (if needed)
cluster_0_data = np.array(cluster_0_data)
cluster_1_data = np.array(cluster_1_data)
cluster_2_data = np.array(cluster_2_data)



# Get the cluster centers and labels
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_


# Visualize the clusters
plt.scatter(df_selected["age"], df_selected["troponin"], c=cluster_labels, cmap='viridis')
plt.xlabel("Age")
plt.ylabel("troponin")
plt.title("KMeans Clustering")
plt.show()
