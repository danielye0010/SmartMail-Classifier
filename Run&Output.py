import datetime
import json
import logging
import os
import pickle
import re
import string

import gspread
import nltk
from dotenv import load_dotenv
from google.oauth2 import service_account
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

load_dotenv()

logging.basicConfig(
    filename="app.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

KEY_FILE_PATH = os.getenv("KEY_FILE_PATH")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
MODEL_PATH = os.getenv("MODEL_PATH", "RFmodel.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "tfidf_vectorizer.pkl")


def clean_text(text):
    text = re.sub(r"\S*@\S*\s?", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)

    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    custom_stopwords = {
        "thank", "thanks", "email", "address", "data", "please", "university",
        "college", "department", "engineer", "consultant", "specialist", "cell",
        "office", "professor", "lab", "technician", "fwd", "fw",
    }
    stop_words.update(custom_stopwords)

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Trained model artifacts are not included in the public repository. "
            "Provide authorized local artifacts via MODEL_PATH and VECTORIZER_PATH."
        )

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


loaded_RFmodel, loaded_vectorizer = load_artifacts()

label_mapping = {
    1: "Welcome / onboarding",
    2: "Account or storage modification",
    3: "Automatic out-of-office message",
    4: "Quota or threshold alert",
    5: "Setup / configuration / mounting issue",
    6: "Spam",
    7: "Storage increase request",
    8: "Special case / recovery / error consultation",
    9: "General information request",
    10: "File transfer / removal / copy request",
    11: "Connection / login / access / sharing issue",
    12: "Performance issue",
    13: "Permission or access-control issue",
}


def predict_email_classification(email_text):
    cleaned_email = clean_text(email_text)
    features = loaded_vectorizer.transform([cleaned_email])
    prediction = loaded_RFmodel.predict(features)[0]
    confidence = max(loaded_RFmodel.predict_proba(features)[0])
    return label_mapping.get(prediction, f"Category {prediction}"), confidence


def parse_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return None


def classify_emails_and_write_to_sheet(filename):
    if not KEY_FILE_PATH or not GOOGLE_SHEET_ID:
        raise RuntimeError("KEY_FILE_PATH and GOOGLE_SHEET_ID must be configured locally.")

    with open("date_range.json", "r", encoding="utf-8") as file:
        date_range = json.load(file)

    start_date_obj = parse_date(date_range["start_date"])
    end_date_obj = parse_date(date_range["end_date"])

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = service_account.Credentials.from_service_account_file(
        KEY_FILE_PATH,
        scopes=scopes,
    )
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet("Sheet1")

    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []
    for inner_list in data:
        for item in inner_list:
            if "body" not in item:
                continue
            created = item.get("created")
            created_date = parse_date(created)
            if created_date and start_date_obj <= created_date <= end_date_obj:
                label, confidence = predict_email_classification(item["body"])
                results.append(
                    [
                        item["body"],
                        label,
                        f"{confidence:.2f}",
                        item.get("createdBy", "N/A"),
                        created,
                    ]
                )

    if results:
        first_empty_row = len(sheet.col_values(1)) + 1
        sheet.update(
            f"A{first_empty_row}:E{first_empty_row + len(results) - 1}",
            results,
        )


if __name__ == "__main__":
    classify_emails_and_write_to_sheet("journal_history.json")
