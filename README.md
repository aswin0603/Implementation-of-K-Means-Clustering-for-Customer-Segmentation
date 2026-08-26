# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries
2. Load the dataset (Mall_Customers.csv)
3. Check dataset info,Check for missing values
4. Select features: Annual Income and Spending Score
5. Standardize the selected features
6. Apply Elbow method for k = 1 to 10 and Plot
7. Apply Silhouette Score for k = 2 to 10 and Plot 
8. Fit K-Means model
9. Predict cluster labels and add cluster labels to the dataset
10.Compute cluster centers
11. Visualize clusters using scatter plot


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: ASWIN B
RegisterNumber:  212224110007
*/
```
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------
# 1. Load the dataset
# ---------------------------------------
df = pd.read_csv("Mall_Customers.csv")  # UPDATE PATH IF NEEDED
print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
display(df.head())

# ---------------------------------------
# 2. Check info and missing values
# ---------------------------------------
print("\nDataset Info:")
display(df.info())
print("\nMissing Values:")
display(df.isnull().sum())

# ---------------------------------------
# 3. Select features for clustering
# Using Income & Spending Score
# ---------------------------------------
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

print("\nFeatures Used:", features)

# ---------------------------------------
# 4. Standardize the data
# ---------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------
# 5. Elbow Method to choose k
# ---------------------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia / SSE")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# ---------------------------------------
# 6. Silhouette Scores
# ---------------------------------------
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(6,4))
plt.plot(range(2, 11), sil_scores, marker='o', color="orange")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.grid(True)
plt.show()

# ---------------------------------------
# 7. Apply KMeans (Choose k=5 by elbow)
# ---------------------------------------
k_final = 5
kmeans = KMeans(n_clusters=k_final, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels
print("\nCluster Counts:")
print(df["Cluster"].value_counts())

# ---------------------------------------
# 8. Cluster Centers in original units
# ---------------------------------------
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(centers_original, columns=features)
centers_df["Cluster"] = range(k_final)

print("\nCluster Centers (Original Values):")
display(centers_df.round(2))

# ---------------------------------------
# 9. Visualization of Clusters
# ---------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    s=70
)

# Show cluster centers
plt.scatter(
    centers_df["Annual Income (k$)"],
    centers_df["Spending Score (1-100)"],
    s=250,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means (k=5)")
plt.legend()
plt.grid(True)
plt.show()




```

## Output:

<img width="682" height="665" alt="image" src="https://github.com/user-attachments/assets/4b0acfe9-5f7a-450c-a16e-5e98df204bac" />


<img width="540" height="391" alt="image" src="https://github.com/user-attachments/assets/d5c830ac-245b-4dba-97ab-4cf0fe515792" />


<img width="545" height="391" alt="image" src="https://github.com/user-attachments/assets/938e3566-04ab-4cf3-a9d9-2549a77d3abc" />


<img width="871" height="372" alt="image" src="https://github.com/user-attachments/assets/aaa6ac94-196a-4a35-b130-843ccc03dcfd" />


<img width="695" height="545" alt="image" src="https://github.com/user-attachments/assets/27193fad-e8a7-4f15-9b05-3bc727f85392" />




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
