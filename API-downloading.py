import requests
import datetime
from time import sleep
import os
import json
from datetime import datetime, timedelta, timezone
import logging
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# debugging
logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables for sensitive information
CHERWELL_URL = os.getenv('CHERWELL_URL')
CLIENT_ID = os.getenv('CLIENT_ID')
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')

# Authenticate and get access token from API
url = f"{CHERWELL_URL}/CherwellAPI/token"
payload = f'grant_type=password&client_id={CLIENT_ID}&username={USERNAME}&password={PASSWORD}'
headers = {'Content-Type': 'application/x-www-form-urlencoded'}

response = requests.request("POST", url, headers=headers, data=payload)

if response.status_code == 200:
    responsejson = response.json()
    token = f"Bearer {responsejson['access_token']}"
    refresh_token_value = responsejson['refresh_token']
    expires = responsejson['.expires']
    logging.debug("Authentication successful")
else:
    logging.error(f"Authentication failed: {response.status_code} {response.text}")

now = datetime.now(timezone.utc)

# convert expires string into a datetime object
expired_dt = datetime.strptime(expires, '%a, %d %b %Y %H:%M:%S %Z')
expired_dt_aware = expired_dt.astimezone(timezone.utc)

# set search time
last_day_of_last_month = datetime(now.year, now.month, 1) - timedelta(days=1)
first_day_of_last_month = datetime(last_day_of_last_month.year, last_day_of_last_month.month, 1)

# Adjust date format
start_date = first_day_of_last_month.strftime('%Y-%m-%dT00:00:00')
end_date = last_day_of_last_month.strftime('%Y-%m-%dT23:59:59')

# JSON payload for the get search results request
search_payload = {
    "associationName": "Incident",
    "busObId": "6dd53665c0c24cab86870a21cf6434ae",
    "filters": [
        {
            "fieldId": "9339fc404e8d5299b7a7c64de79ab81a1c1ff4306c",
            "operator": "eq",
            "value": "SharedDrive"
        },
        {
            "fieldId": "93543557882ad94503745843c9a380aa0c380935c8",
            "operator": "gt",
            "value": start_date
        },
        {
            "fieldId": "93543557882ad94503745843c9a380aa0c380935c8",
            "operator": "lte",
            "value": end_date
        }
    ],
    "includeAllFields": True,
    "pageSize": 50
}

# URL for the getsearchresults endpoint
get_search_results_url = f"{CHERWELL_URL}/CherwellAPI/api/V1/getsearchresults"

authHeader = {
    "Accept": "application/json",
    "Authorization": f"Bearer {responsejson['access_token']}"
}

# POST request to perform the search with the JSON payload
search_response = requests.post(get_search_results_url, headers=authHeader, json=search_payload)

if search_response.ok:
    search_data = search_response.json()
else:
    logging.error(f"Failed to perform the search. Status code: {search_response.status_code}")

# Check and refresh token expiration
def check_token_expiration(expires):
    now = datetime.now()
    expires_datetime = datetime.strptime(expires, '%a, %d %b %Y %H:%M:%S %Z')
    timediff = expires_datetime - now
    if timediff.seconds < 960:
        return True
    else:
        return False

# Refresh the access token using the refresh token
def refresh_token(current_refresh_token):
    if check_token_expiration(expires):
        url = f"{CHERWELL_URL}/CherwellAPI/token"
        payload = f'grant_type=refresh_token&client_id={CLIENT_ID}&refresh_token={current_refresh_token}'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        if response.ok:
            response_json = response.json()
            new_access_token = f"Bearer {response_json['access_token']}"
            new_refresh_token = response_json['refresh_token']
            return new_access_token, new_refresh_token
        else:
            logging.error("Failed to refresh token")
            return None, current_refresh_token  # Return None if the refresh fails
    else:
        return None, current_refresh_token  # Return None if the token does not need to be refreshed


original_ticket = []
history_dictionary = []

if search_response.ok:
    new_search_data = search_response.json()

    # search
    for object in new_search_data['businessObjects']:
        refresh_token(refresh_token_value)
        sleep(0.5)  #

        # get id
        busobid = object['busObId']
        recid = object['busObRecId']

        ticketURL_full = f'{CHERWELL_URL}/CherwellAPI/api/V1/getbusinessobject/busobid/{busobid}/busobrecid/{recid}'
        original_ticket_response = requests.get(ticketURL_full, headers=authHeader)
        if original_ticket_response.ok:
            original_ticket_dict = original_ticket_response.json()
            original_ticket.append(original_ticket_dict)

        historiesURL = f'{CHERWELL_URL}/CherwellAPI/api/V1/getactivities/busobid/{busobid}/busobrecid/{recid}/pagesize/100'
        history_response = requests.get(historiesURL, headers=authHeader)
        if history_response.ok:
            history_dict = history_response.json()
            history_dictionary.append(history_dict)

# Save the collected history data to a JSON file
with open("journal_history.json", 'w') as file:
    json.dump(history_dictionary, file)

# export date range
now = datetime.now(timezone.utc)
last_day_of_last_month = datetime(now.year, now.month, 1) - timedelta(days=1)
first_day_of_last_month = datetime(last_day_of_last_month.year, last_day_of_last_month.month, 1)

start_date = first_day_of_last_month.strftime('%Y-%m-%dT00:00:00')
end_date = last_day_of_last_month.strftime('%Y-%m-%dT23:59:59')

# save date range
date_range = {'start_date': start_date, 'end_date': end_date}
with open('date_range.json', 'w') as f:
    json.dump(date_range, f)
