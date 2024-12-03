# Email Classifier  


## Sensitive Info Disclaimer
All sensitive information have been either hidden or modified. The provided code and configuration files are only designed to demonstrate the functionality of this project.

## Introduction
This classifier system automates the entire pipeline for handling emails from ResearchDrive customers, including API downloading, pre-processing, feature engineering, classification, outputting, and visualization. Trained on over 2,000 labeled archived emails using SMOTE Random Forest, the core classifier categorizes emails into 13 predefined labels. The system outputs categorized emails to a Google Sheet and provides visualizations through Looker Studio. Containerized with Docker for consistency and reproducibility, the process is automated via a monthly cron job on a GCP VM.



## Automatic System Pipeline
- **Download**: Utilizes the WiscIT API query search to collect RD support-related emails for the last 30 days.
- **Pre-process**: Cleans and preprocesses the raw email data to remove unnecessary information.
- **Classify**: Emails are cleaned, vectorized, and categorized using the trained model.
- **Output**: Emails are output to a Google Sheet with their metadata and labels.
- **Visualization**: Results are visualized in Looker Studio as a monthly summary report.
- **Automation**: The pipeline is containerized using Docker, ensuring a consistent and reproducible environment. A cron job within the Docker container is scheduled to run monthly on a GCP VM, managing the entire process and ensuring continuous data integration and refinement.

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
`0 0 1 * * docker run --rm -v /path/to/project:/app myproject-image`  

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
- **GCP VM Details**: Rocky Linux; Instance ID: 1102777180463610226; Name: doit-rci-vm01



