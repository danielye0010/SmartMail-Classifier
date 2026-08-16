import json
import logging
import os
from datetime import datetime, timedelta, timezone
from time import sleep

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename="app.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Credentials and deployment-specific identifiers must be supplied locally.
CHERWELL_URL = os.getenv("CHERWELL_URL")
CLIENT_ID = os.getenv("CLIENT_ID")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
BUSINESS_OBJECT_ID = os.getenv("BUSINESS_OBJECT_ID")
SERVICE_FILTER_FIELD_ID = os.getenv("SERVICE_FILTER_FIELD_ID")
CREATED_FIELD_ID = os.getenv("CREATED_FIELD_ID")
SERVICE_FILTER_VALUE = os.getenv("SERVICE_FILTER_VALUE")

required = {
    "CHERWELL_URL": CHERWELL_URL,
    "CLIENT_ID": CLIENT_ID,
    "USERNAME": USERNAME,
    "PASSWORD": PASSWORD,
    "BUSINESS_OBJECT_ID": BUSINESS_OBJECT_ID,
    "SERVICE_FILTER_FIELD_ID": SERVICE_FILTER_FIELD_ID,
    "CREATED_FIELD_ID": CREATED_FIELD_ID,
    "SERVICE_FILTER_VALUE": SERVICE_FILTER_VALUE,
}
missing = [name for name, value in required.items() if not value]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# Authenticate and obtain an access token.
token_url = f"{CHERWELL_URL}/CherwellAPI/token"
payload = (
    f"grant_type=password&client_id={CLIENT_ID}"
    f"&username={USERNAME}&password={PASSWORD}"
)
headers = {"Content-Type": "application/x-www-form-urlencoded"}
response = requests.post(token_url, headers=headers, data=payload)
response.raise_for_status()
response_json = response.json()

access_token = response_json["access_token"]
refresh_token_value = response_json["refresh_token"]
expires = response_json[".expires"]
logging.debug("Authentication successful")

auth_header = {
    "Accept": "application/json",
    "Authorization": f"Bearer {access_token}",
}

# Query the previous calendar month.
now = datetime.now(timezone.utc)
last_day_of_last_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc) - timedelta(days=1)
first_day_of_last_month = datetime(
    last_day_of_last_month.year,
    last_day_of_last_month.month,
    1,
    tzinfo=timezone.utc,
)
start_date = first_day_of_last_month.strftime("%Y-%m-%dT00:00:00")
end_date = last_day_of_last_month.strftime("%Y-%m-%dT23:59:59")

search_payload = {
    "associationName": "Incident",
    "busObId": BUSINESS_OBJECT_ID,
    "filters": [
        {
            "fieldId": SERVICE_FILTER_FIELD_ID,
            "operator": "eq",
            "value": SERVICE_FILTER_VALUE,
        },
        {
            "fieldId": CREATED_FIELD_ID,
            "operator": "gt",
            "value": start_date,
        },
        {
            "fieldId": CREATED_FIELD_ID,
            "operator": "lte",
            "value": end_date,
        },
    ],
    "includeAllFields": True,
    "pageSize": 50,
}

search_url = f"{CHERWELL_URL}/CherwellAPI/api/V1/getsearchresults"
search_response = requests.post(search_url, headers=auth_header, json=search_payload)
search_response.raise_for_status()
search_data = search_response.json()


def token_needs_refresh(expires_text):
    expires_datetime = datetime.strptime(expires_text, "%a, %d %b %Y %H:%M:%S %Z")
    remaining = expires_datetime - datetime.utcnow()
    return remaining.total_seconds() < 960


def refresh_token(current_refresh_token):
    if not token_needs_refresh(expires):
        return None, current_refresh_token

    refresh_payload = (
        f"grant_type=refresh_token&client_id={CLIENT_ID}"
        f"&refresh_token={current_refresh_token}"
    )
    refresh_response = requests.post(token_url, headers=headers, data=refresh_payload)
    refresh_response.raise_for_status()
    refreshed = refresh_response.json()
    return refreshed["access_token"], refreshed["refresh_token"]


history_records = []
current_refresh_token = refresh_token_value

for item in search_data.get("businessObjects", []):
    new_token, current_refresh_token = refresh_token(current_refresh_token)
    if new_token:
        auth_header["Authorization"] = f"Bearer {new_token}"

    sleep(0.5)
    busobid = item["busObId"]
    recid = item["busObRecId"]

    history_url = (
        f"{CHERWELL_URL}/CherwellAPI/api/V1/getactivities/"
        f"busobid/{busobid}/busobrecid/{recid}/pagesize/100"
    )
    history_response = requests.get(history_url, headers=auth_header)
    history_response.raise_for_status()
    history_records.append(history_response.json())

# Generated files may contain restricted records and must remain local.
with open("journal_history.json", "w", encoding="utf-8") as file:
    json.dump(history_records, file)

with open("date_range.json", "w", encoding="utf-8") as file:
    json.dump({"start_date": start_date, "end_date": end_date}, file)
