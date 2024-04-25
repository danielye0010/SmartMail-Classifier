import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random
from sklearn.metrics import classification_report

np.random.seed(42)
random.seed(42)

# Load the data
file_path = "C:/Users/Daniel Ye/Desktop/classify/labeling_final.csv"
data = pd.read_csv(file_path, encoding="ISO-8859-1")

# find out there are some blank rows in the csv, need be dropped
# na_rows = data[data['Cleaned_Text'].isna() | data['label'].isna()]
# print(na_rows)

data.dropna(subset=['Cleaned_Text', 'label'], inplace=True)

# Prepare the data using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['Cleaned_Text'])
y = data['label']

# Define models, choose , class_weight='balanced'
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'SVM': SVC(probability=True, class_weight='balanced'),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(class_weight='balanced')
}

# Perform 10-fold stratified cross-validation
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

# run each model
for name, model in models.items():
    y_pred = cross_val_predict(model, X, y, cv=stratified_kfold)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    rmse = mean_squared_error(y, y_pred, squared=False)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'RMSE': rmse
    })

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)

print(results_df)

# focus on the Random Forest model with class weights:
RFmodel = RandomForestClassifier(class_weight='balanced')
RFmodel.fit(X, y)
y_pred = cross_val_predict(RFmodel, X, y, cv=stratified_kfold)

# Get the classification report
report = classification_report(y, y_pred, output_dict=True)

# Extracting metrics for each class
for label in np.unique(y):
    class_metrics = report[str(label)]
    print(f"Metrics for class {label}:")
    print(f"Precision: {class_metrics['precision']:.3f}")
    print(f"Recall: {class_metrics['recall']:.3f}")
    print(f"F1 Score: {class_metrics['f1-score']:.3f}")
    print("----------")

# export the model and vectorizer
import pickle

with open('RFmodel.pkl', 'wb') as model_file:
    pickle.dump(RFmodel, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)
