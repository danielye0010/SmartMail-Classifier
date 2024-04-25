# Email Classifier

## Project Resources
- **Support Files**: [Google Drive Folder](https://drive.google.com/drive/u/1/folders/1Z4pYzs5GwdmJ45PrAhQ3ljAhrYO1jXDC)
- **Looker Studio Dashboard**: [View Dashboard](https://lookerstudio.google.com/u/1/reporting/a83f9c32-c6f1-44b0-a2bc-31e258d943e1/page/oIrvD)
- **Monthly Emails Google Sheet**: [View Google Sheet](https://docs.google.com/spreadsheets/u/1/d/1YzhjixsGiazThC5dxD1d6tE7TpeF1NcV0fRusegtOgQ/edit?usp=drive_web&ouid=116239248667180049470)
- **Detailed Model Description**: [Google Docs](https://docs.google.com/document/d/1J7x77BfacRxQpYYlaHieyMWikch25IBx/edit?rtpof=true)
- **GCP VM Details**: Rocky Linux; Instance ID: 1102777180463610226; Name: doit-rci-vm01

## Introduction
The Email Classifier project automates the classification of emails from ResearchDrive customers into predefined categories. Originally trained on over 2000 archived emails, the classifier outputs results to a Google Sheet and visualizes them in Looker Studio. This process is managed through scripts executed monthly on a Google Cloud Platform (GCP) Virtual Machine (VM), ensuring continuous data integration and refinement.

## Model Overview
- **Data Collection**: Utilizes the WiscIT system to collect 2007 support-related emails, filtering out duplicates to ensure quality.
- **Data Cleaning**: Applies advanced text processing techniques using the Natural Language Toolkit (NLTK) for email preprocessing.
- **Exploratory Data Analysis**: Employs K-Means and Hierarchical Clustering to understand the data structure and identify optimal cluster numbers.
- **Model Training and Evaluation**: Tests various models including Logistic Regression, SVM, Naive Bayes, and Random Forest. Uses class weighting and SMOTE to handle class imbalances.
- **Testing**: A toy_classifier with GUI is developed to enable text input to test the classify process.


## Dependencies
Python 3.10
Required Python packages to pip install: requests, nltk, gspread, google-auth, pickle, re, string

## Usage

### Cron Scheduling
First, enable Autostart: In "Automation" section, check the box for "Autostart" to enable automatic startup for the VM.
Then go and edit cron in vm:
`crontab -e`  
Then add follwing to set to run automatically at midnight on the first day of each month:  
`0 0 1 * * /path/to/run_scripts.sh`  
shutdown the vm  
`sudo shutdown now`  
Then save:  
`:wq`  
Double check authority :  
`crontab -l`  
`chmod +x /path/to/run_scripts.sh`

## Debugging

There will be an app.log file generated with 1st_api, api error will show here if any.  
There will be a date_range.json file generated with 1st_api, check if the time range matches your desire in api serching load.

## Potential Improvements
1. Quality Control   
Regularly perform quality checks on the data. For emails where automatic classification results with low confident rate, perform manual annotation.

2. Data Preprocessing Optimization  
Update Stopwords List: Update the custom stopwords list based on the newly collected data.  
Replace model: Implement more advanced models such as BERT, transformer or Neural Network.

3. Model Retraining
Regularly retrain the model using the latest annotated dataset to adapt to changes in email content and emerging categories.

4. Model Evaluation  
Assess the performance of the updated model on the new dataset using metrics such as accuracy, recall, and F1 score.

5. Active Learning  
Automatically flag emails for review when the model's confidence in classification results is low, requesting manual review. The data from this review can be used for further training of the model.


## Contributors

- **Daniel Ye**
- **Casey Schacher** 
- **Bernie Bernstein**
- **ZEKAI OTLES** 
- **Sam Fosler**





