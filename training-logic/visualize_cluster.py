import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

cleaned_input_path = "C:/Users/Daniel Ye/Desktop/clustered_output.csv"
df = pd.read_csv(cleaned_input_path, encoding="ISO-8859-1")

# Initialize an empty dictionary to store the most common words for each cluster
most_common_words_by_cluster = {}

# Group the data by cluster label
grouped = df.groupby('Cluster')

# For each cluster, compute the most common words
for cluster_label, group_data in grouped:
    # Combine all emails in the cluster into one text
    combined_text = ' '.join(group_data['Cleaned_Text'])

    # Tokenize the text
    vectorizer = CountVectorizer()
    words = vectorizer.fit_transform([combined_text])
    word_list = vectorizer.get_feature_names_out()

    # Calculate word frequencies using Counter
    word_freq_dict = Counter(dict(zip(word_list, words.toarray()[0])))

    # Get the three most common words and their counts
    most_common_words = word_freq_dict.most_common(5)

    most_common_words_by_cluster[cluster_label] = most_common_words

print(most_common_words_by_cluster)
