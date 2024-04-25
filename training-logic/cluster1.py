import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load Cleaned Data
cleaned_input_path = "C:/Users/Daniel Ye/Desktop/no auto cleaned.csv"
df_cleaned = pd.read_csv(cleaned_input_path, encoding="ISO-8859-1")

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english', use_idf=True)
X = vectorizer.fit_transform(df_cleaned['Cleaned_Text'])

# Finding the optimal number of clusters using the Elbow Method
wcss = []  # Within-Cluster-Sum-of-Squares
k_values = range(10, 30)  # for instance, testing k values from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 5))
plt.plot(k_values, wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

