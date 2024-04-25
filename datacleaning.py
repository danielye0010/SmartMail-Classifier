import pandas as pd
import re
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


# Functions

def extract_subject_and_body(email):
    subject_match = re.search(r'Subject: (.*?)\n', email)
    subject = subject_match.group(1) if subject_match else ""
    parts = re.split(r'(?:Subject:|From:|To:|Date:)', email)
    body = parts[-1].strip()
    combined_text = subject + " " + body
    return combined_text


def clean_text(text):
    # Remove emails
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    custom_stopwords = ['thank', 'thanks', 'wisc', 'edu', 'email', 'address', 'research', 'drive', 'researchdrive', 're',
                        're:', 'university', 'wisc', 'madison', 'wisconsin', '***', 'data', 'please', 'Madison', 'Wi',
                        "Tel", 'Hi', 'wisc.edu','college','department','engineer', 'consultant', 'specialist', 'cell',
                        'office', 'fwd:', 'fw:', 'professor','am','pm','lab', 'technician','john']
    stop_words.update(custom_stopwords)
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Joining back
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Load Data
file_path = "C:/Users/Daniel Ye/Desktop/classify/Data1.csv"
df = pd.read_csv(file_path)

# Cleaning
df['Combined_Text'] = df['Email'].apply(extract_subject_and_body)
df['Cleaned_Text'] = df['Combined_Text'].apply(clean_text)
df_cleaned = df[df['Cleaned_Text'].notna() & (df['Cleaned_Text'] != '')].copy()

# Save Cleaned Data to CSV
cleaned_output_path = "C:/Users/Daniel Ye/Desktop/Cleaned_Data.csv"
df_cleaned.to_csv(cleaned_output_path, index=False)

cleaned_input_path = "C:/Users/Daniel Ye/Desktop/Cleaned_Data.csv"
df = pd.read_csv(cleaned_input_path, encoding="ISO-8859-1")
df = df.dropna(subset=['Cleaned_Text'])  # Drop rows where 'Cleaned_Text' is NaN
texts = df['Cleaned_Text'].tolist()

# Convert texts to TF-IDF vectors
vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000)
X = vectorizer.fit_transform(texts)

# Apply K-means clustering with k = 12
kmeans = KMeans(n_clusters=12, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Save the dataframe with cluster labels to a new CSV file
df.to_csv('C:/Users/Daniel Ye/Desktop/labeling.csv', index=False)

# stratify code later after labeling
# data = pd.read_csv("C:/Users/Daniel Ye/Desktop/Cleaned_Data.csv", encoding="ISO-8859-1")
# train, test = train_test_split(data, test_size=0.20, stratify=data['Cleaned_Text'], random_state=42)
# train.to_csv("C:/Users/Daniel Ye/Desktop/train.csv", index=False)
# test.to_csv("C:/Users/Daniel Ye/Desktop/test.csv", index=False)