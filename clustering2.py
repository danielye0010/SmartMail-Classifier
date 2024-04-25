import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
cleaned_input_path = "C:/Users/Daniel Ye/Desktop/no auto cleaned.csv"
df = pd.read_csv(cleaned_input_path, encoding="ISO-8859-1")
df = df.dropna(subset=['Cleaned_Text'])  # Drop rows where 'Cleaned_Text' is NaN
texts = df['Cleaned_Text'].tolist()

# Convert texts to TF-IDF vectors
vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000)
X = vectorizer.fit_transform(texts)

 # Use the Elbow method to find a good number of clusters using WCSS (Within-Cluster-Sum-of-Squares)
 # wcss = []
 # max_clusters = 20  # or another number depending on your preference
 # for i in range(1, max_clusters + 1):
 #     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
 #     kmeans.fit(X)
 #     wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
# plt.figure(figsize=(10,5))
# plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# # Apply K-means clustering with k = 12
kmeans = KMeans(n_clusters=12, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# # Save the dataframe with cluster labels to a new CSV file
df.to_csv('C:/Users/Daniel Ye/Desktop/clustered_output.csv', index=False)

# # Convert texts to TF-IDF vectors
# vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000)
# X = vectorizer.fit_transform(texts).todense()
# # Perform hierarchical clustering
# Z = linkage(X, method='ward')  # Using the "ward" method for linkage
#
# # Plot the dendrogram
# plt.figure(figsize=(10, 5))
# dendrogram(Z, truncate_mode='level', p=3)  # 'p' controls the number of levels in the dendrogram
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Email Index')
# plt.ylabel('Euclidean Distance')
# plt.show()
