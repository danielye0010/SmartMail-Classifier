# Email Classifier  


## Sensitive Information Disclaimer
In the interest of security and privacy for UW-Madison, all sensitive information have been either hidden or modified. The provided code and configuration files are designed to demonstrate the functionality of the project without exposing any actual sensitive data. For actual deployment, please replace the placeholders with your real credentials and ensure that they are securely stored using environment variables or secret management tools.

## Introduction
This classifier system automates the entire pipeline for handling emails from ResearchDrive customers, including API downloading, pre-processing, feature engineering, classification, outputting, and visualization. The core classifier, trained on over 2,000 labeled archived emails using SMOTE Random Forest, categorizes emails into 13 predefined labels. Automated scripts, executed monthly on a GCP VM via shell, manage this process, ensuring continuous data integration and refinement. The system seamlessly outputs categorized emails and their labels to a Google Sheet and provides visualizations through Looker Studio.



## Automatic System Pipeline
- **Download**: Utilizes the WiscIT API query search to collect RD support-related emails for the last 30 days.
- **Pre-process**: Cleans and preprocesses the raw email data to remove unnecessary information.
- **Classify**: Emails are cleaned, vectorized, and categorized using the trained model.
- **Output**: Emails are output to a Google Sheet with their metadata and labels.
- **Visualization**: Results are visualized in Looker Studio as a monthly summary report.
- **Automation**: The entire pipeline is managed through scripts executed monthly by shell on a GCP VM, ensuring continuous data integration and refinement.

## Model Overview
- **Data Collection**: Utilizes the WiscIT API query search to collect 2007 support-related emails for a specified period.
- **Data Cleaning**: Applies text processing techniques using NLTK and manual cleaning, including filtering out duplicates and auto-replies.
- **Feature Engineering**: Extracts relevant features from the cleaned email data for model training.
- **Labeling**: Combines semi-supervised learning and active learning. Trains a logistic regression model with labeled data, predicts labels for unlabeled data, and manually corrects low-confidence predictions to improve performance.
- **Exploratory Data Analysis**: Employs K-Means and Hierarchical Clustering to understand the data structure and identify optimal cluster numbers.
- **Model Training and Evaluation**: Tests various ML models, using class weighting and SMOTE to handle class imbalances. Evaluates model performance using metrics such as accuracy, precision, recall, and F1-score.
- **Deployment**: Implements the trained model into the automated system pipeline for real-time classification.
- **Testing**: Develops a toy classifier with a GUI to enable text input for testing the classification process.


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

## Continuous Model Improvement
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
Hosted by team of Research Cyberinfrastructure at UW-Madison CTO
- **Daniel Ye**
- **Casey Schacher** 
- **Bernie Bernstein**
- **ZEKAI OTLES** 
- **Sam Fosler**

## Project Resources
- **Support Files**: [Google Drive Folder](https://drive.google.com/drive/u/1/folders/1Z4pYzs5GwdmJ45PrAhQ3ljAhrYO1jXDC)
- **Looker Studio Dashboard**: [View Dashboard](https://lookerstudio.google.com/u/1/reporting/a83f9c32-c6f1-44b0-a2bc-31e258d943e1/page/oIrvD)
- **Monthly Emails Google Sheet**: [View Google Sheet](https://docs.google.com/spreadsheets/u/1/d/1YzhjixsGiazThC5dxD1d6tE7TpeF1NcV0fRusegtOgQ/edit?usp=drive_web&ouid=116239248667180049470)
- **Detailed Model Description**: [Google Docs](https://docs.google.com/document/d/1J7x77BfacRxQpYYlaHieyMWikch25IBx/edit?rtpof=true)
- **GCP VM Details**: Rocky Linux; Instance ID: 1102777180463610226; Name: doit-rci-vm01



