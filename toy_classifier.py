import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import simpledialog, messagebox


# Data Cleaning Function
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
    custom_stopwords = ['thank', 'thanks', 'wisc', 'edu', 'email', 'address', 'research', 'drive', 'researchdrive',
                        're',
                        're:', 'university', 'wisc', 'madison', 'wisconsin', '***', 'data', 'please', 'Madison', 'Wi',
                        "Tel", 'Hi', 'wisc.edu', 'college', 'department', 'engineer', 'consultant', 'specialist',
                        'cell',
                        'office', 'fwd:', 'fw:', 'professor', 'am', 'pm', 'lab', 'technician', 'john']
    stop_words.update(custom_stopwords)
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Joining back
    cleaned_text = ' '.join(tokens)
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


def predict_email_classification(email_text):
    # Clean the email text
    cleaned_email = clean_text(email_text)

    # Vectorize the cleaned email
    features = loaded_vectorizer.transform([cleaned_email])

    # Predict using the loaded model
    prediction = loaded_RFmodel.predict(features)[0]
    confidence = max(loaded_RFmodel.predict_proba(features)[0])
    label_description = label_mapping[prediction]

    return label_description, confidence


def ask_for_email_and_predict():
    # Use simpledialog to get the email text
    email_text = simpledialog.askstring("Input", "Enter the email text:")

    # Check if the user didn't cancel the dialog
    if email_text:
        label, confidence_score = predict_email_classification(email_text)

        # Interpret the confidence score
        if confidence_score >= 0.9:
            confidence_description = "Trust me, I am very confident about this classification."
        elif confidence_score >= 0.7:
            confidence_description = "I am pretty confident about this classification but cannot say for certain"
        elif confidence_score >= 0.5:
            confidence_description = "I am not really sure about my answer."
        else:
            confidence_description = "I am guessing! Please review this manually."

        # Use messagebox to show the result
        messagebox.showinfo("Prediction",
                            f"Predicted Label: {label}\nConfidence: {confidence_score:.2f}\n{confidence_description}")


def main_app():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Show the welcome message
    welcome_text = ("Hi, I'm the email-classifier.V1 developed by the UW-Madison ResearchDrive support team, "
                    "designed to help identify the main idea of incoming emails from ResearchDrive customers."
                    "I learned 2000+ real emails by the Random Forest algorithm. Please enter the email content, "
                    "and I will help you identify what it is talking about!")
    messagebox.showinfo("Welcome!", welcome_text)

    while True:
        ask_for_email_and_predict()

        # Ask the user if they want to continue or quit
        continue_or_quit = messagebox.askquestion("Continue?", "Would you like to classify another email?")

        if continue_or_quit == "no":
            break


if __name__ == "__main__":
    main_app()