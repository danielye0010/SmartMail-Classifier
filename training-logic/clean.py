import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def extract_body(email):
    subject_match = re.search(r'Subject: (.*?)\n', email)
    subject = subject_match.group(1) if subject_match else ""

    # Extract body by splitting on metadata headers
    parts = re.split(r'(?:Subject:|From:|To:|Date:)', email)
    body = parts[-1].strip()

    # Combine subject and body
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
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Joining back
    cleaned_text = ' '.join(tokens)

    return cleaned_text


# Load the CSV
file_path = "C:/Users/Daniel Ye/Desktop/Data1.csv"
df = pd.read_csv(file_path)

# Cleaning
df['Combined_Text'] = df['Email'].apply(extract_subject_and_body)
df['Cleaned_Text'] = df['Combined_Text'].apply(clean_text)
df = df[df['Cleaned_Text'].notna() & (df['Cleaned_Text'] != '')]

# Save Cleaned Data to CSV
cleaned_output_path = "C:/Users/Daniel Ye/Desktop/Cleaned_Data1.csv"
df_cleaned.to_csv(cleaned_output_path, index=False)

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english', use_idf=True)
X = vectorizer.fit_transform(df_cleaned['Cleaned_Text'])

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_cleaned['Cluster'] = kmeans.fit_predict(X)

# Save Clustering Results to CSV
clustered_output_path = "C:/Users/Daniel Ye/Desktop/Clustered_Data1.csv"
df_cleaned.to_csv(clustered_output_path, index=False)

