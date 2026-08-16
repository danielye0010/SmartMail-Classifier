import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the datasets
file_path1 = "C:/Users/Daniel Ye/Desktop/labeling.csv"
labeled_data = pd.read_csv(file_path1, encoding="ISO-8859-1")
file_path2 = "C:/Users/Daniel Ye/Desktop/active_label.csv"
unlabeled_data = pd.read_csv(file_path2, encoding="ISO-8859-1")

# Replace NaN values in the 'Cleaned_Text' column with empty strings
labeled_data.dropna(subset=['label'], inplace=True)
labeled_data['Cleaned_Text'].fillna("", inplace=True)
unlabeled_data['Cleaned_Text'].fillna("", inplace=True)

# Preprocess the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
combined_text = pd.concat([labeled_data['Cleaned_Text'], unlabeled_data['Cleaned_Text']])
tfidf_vectorizer.fit(combined_text)

X_labeled = tfidf_vectorizer.transform(labeled_data['Cleaned_Text'])
X_unlabeled = tfidf_vectorizer.transform(unlabeled_data['Cleaned_Text'])
y_labeled = labeled_data['label']

# Train a logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_labeled, y_labeled)

# Predict the labels for the unlabeled data
predicted_labels = logreg.predict(X_unlabeled)
predicted_probs = logreg.predict_proba(X_unlabeled)

# Adding predicted labels to the unlabeled data
unlabeled_data['Predicted_Label'] = predicted_labels

# add the probabilities for each class
for i, col in enumerate(logreg.classes_):
    unlabeled_data[f"Prob_{col}"] = predicted_probs[:, i]

# Save the augmented unlabeled data with predictions
unlabeled_data.to_csv("C:/Users/Daniel Ye/Desktop/active_label_with_predictions.csv", index=False)
