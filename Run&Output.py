import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
from google.oauth2 import service_account
import gspread
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# debugging
logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variable for the key file path
KEY_FILE_PATH = os.getenv('KEY_FILE_PATH')

# Clean email text by removing unwanted characters and lemmatizing
def clean_text(text):
    text = re.sub(r'\S*@\S*\s?', '', text)  # Remove emails
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    custom_stopwords = ['thank', 'thanks', 'wisc', 'edu', 'email', 'address', 'research', 'drive', 'researchdrive',
                        're', 're:', 'university', 'wisc', 'madison', 'wisconsin', '***', 'data', 'please', 'Madison', 'Wi',
                        "Tel", 'Hi', 'wisc.edu', 'college', 'department', 'engineer', 'consultant', 'specialist',
                        'cell', 'office', 'fwd:', 'fw:', 'professor', 'am', 'pm', 'lab', 'technician', 'john']
    stop_words.update(custom_stopwords)
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()  # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = ' '.join(tokens)  # Joining back
    return cleaned_text

# Load the trained Random Forest model
with open('RFmodel.pkl', 'rb') as model_file:
    loaded_RFmodel = pickle.load(model_file)

# Load the saved tfidf_vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Define the label mapping
label_mapping = {
    1: "Auto email - Welcome to Research Drive",
    2: "Auto email - ResearchDrive account/storage Modification/Request",
    3: "Auto email - Customer not in office",
    4: "Auto email - Reach quota limit/Alert threshold violation",
    5: "Setup / configuration / mounting issues",
    6: "Spam",
    7: "Quota increase/adding storage request",
    8: "Special User case/Error issue consult/Recover missing file",
    9: "Interested in applying/Want to learn more about ResearchDrive",
    10: "Transfer / remove/copy files",
    11: "Connect/VPN/firewall/Login/accessing/sharing issues",
    12: "Slow speed/stuck - need help",
    13: "Manifest/permission issue/grant access"
}

# Predict the classification of an email
def predict_email_classification(email_text):
    cleaned_email = clean_text(email_text)  # Clean the email text
    features = loaded_vectorizer.transform([cleaned_email])  # Vectorize the cleaned email
    prediction = loaded_RFmodel.predict(features)[0]  # Predict using the loaded model
    confidence = max(loaded_RFmodel.predict_proba(features)[0])
    label_description = label_mapping[prediction]
    return label_description, confidence

# Change date format
def parse_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        return None

# Classify emails and output results to a Google Sheet
def classify_emails_and_write_to_sheet(filename, sheet_id):

    with open('date_range.json', 'r') as f:
        date_range = json.load(f)
    start_date = date_range['start_date']
    end_date = date_range['end_date']

    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    credentials = service_account.Credentials.from_service_account_file(KEY_FILE_PATH, scopes=scopes)
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(sheet_id).worksheet('Sheet1')

    # Load and classify emails
    with open(filename, 'r') as file:
        data = json.load(file)

    start_date_obj = parse_date(start_date)
    end_date_obj = parse_date(end_date)

    results = []
    for inner_list in data:
        for item in inner_list:
            if "body" in item:
                created = item.get("created", "N/A")
                created_date = parse_date(created)
                if created_date and start_date_obj <= created_date <= end_date_obj:
                    email_text = item["body"]
                    label, confidence = predict_email_classification(email_text)
                    created_by = item.get("createdBy", "N/A")
                    results.append([email_text, label, "{:.2f}".format(confidence), created_by, created])

    # Find first blank row
    first_empty_row = len(sheet.col_values(1)) + 1  # Go to the first blank row

    # Write in
    if results:  # Make sure it is not empty
        sheet.update(f'A{first_empty_row}:E{first_empty_row + len(results) - 1}', results)

# Run
if __name__ == "__main__":
    classify_emails_and_write_to_sheet('journal_history.json', 'your_google_sheet_id')
