import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Load the data
file_path = "C:/Users/Daniel Ye/Desktop/labeling_final.csv"
data = pd.read_csv(file_path, encoding="ISO-8859-1")

# Drop NA values
data.dropna(subset=['Cleaned_Text', 'label'], inplace=True)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['Cleaned_Text'])
y = data['label'].values

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier()
}

# 10-fold stratified cross-validation
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    fold_metrics = []
    for train_index, val_index in stratified_kfold.split(X, y):
        # Split the data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Apply SMOTE only to the training data
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Train the model
        model_clone = model
        model_clone.fit(X_train_smote, y_train_smote)

        # Make predictions
        y_pred = model_clone.predict(X_val)

        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        fold_metrics.append({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'RMSE': rmse
        })

    # Average metrics over all folds
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0]}
    avg_metrics['Model'] = name
    results.append(avg_metrics)

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)

print(results_df)
