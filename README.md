# Email Classifier

## Privacy and Security Notice

This repository is a sanitized demonstration of an internal email-classification workflow. Credentials, participant/customer data, infrastructure identifiers, service-account files, trained production artifacts, and other restricted operational details are intentionally not distributed here.

## Introduction

This project demonstrates an automated pipeline for handling support emails: API ingestion, preprocessing, feature engineering, classification, output, visualization, and scheduled execution. The original system was trained on more than 2,000 labeled archived emails and used a Random Forest classifier with TF-IDF features and class-imbalance handling.

## System Pipeline

- **Download**: Retrieve recent support records through an authenticated service API.
- **Pre-process**: Clean and normalize email text.
- **Classify**: Vectorize text and apply a trained classifier.
- **Output**: Write categorized results to a Google Sheet.
- **Visualization**: Support downstream reporting and dashboarding.
- **Automation**: Run the workflow in Docker on a scheduled host.

## Model Development

The development workflow included:

- text cleaning and feature engineering;
- semi-supervised and active-learning experiments;
- clustering for exploratory analysis;
- comparison of multiple classifiers;
- class-imbalance handling;
- evaluation using accuracy, precision, recall, and F1 score.

## Public Repository Scope

The public repository contains demonstration code only. It does **not** include:

- archived or production email data;
- production credentials or service-account files;
- organization-specific infrastructure identifiers;
- fitted vectorizers or trained production model binaries.

To run the workflow, provide your own authorized data, trained model artifacts, and environment configuration.

## Configuration

Sensitive values should be supplied through environment variables, for example:

```text
CHERWELL_URL
CLIENT_ID
USERNAME
PASSWORD
KEY_FILE_PATH
GOOGLE_SHEET_ID
BUSINESS_OBJECT_ID
SERVICE_FILTER_FIELD_ID
CREATED_FIELD_ID
SERVICE_FILTER_VALUE
```

Do not commit a real `.env` file, credentials, access tokens, customer data, or service-account JSON files.

## Dependencies

The project uses Python, `requests`, `nltk`, `gspread`, `google-auth`, `python-dotenv`, and scikit-learn-compatible model artifacts.

## Automation

The workflow can be containerized with Docker and scheduled externally (for example, with cron). Deployment-specific host names, instance IDs, and production paths are intentionally omitted from this public repository.

## Project Context

Developed as part of work with the UW-Madison Research Cyberinfrastructure team. This repository is intended to demonstrate the system architecture and implementation approach rather than reproduce the production environment.
