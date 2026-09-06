import json
import os
import time
import requests

BASE_URL = "https://raw.githubusercontent.com/hudl/open-data/master/data"

# Competition / season we want
COMPETITION_ID = 11
SEASON_ID = 27

# Where event files will be saved
EVENTS_DIR = "data/raw/events/laliga_2015_16"

os.makedirs(EVENTS_DIR, exist_ok=True)

# Get all matches in this competition / season
matches = requests.get(
    f"{BASE_URL}/matches/{COMPETITION_ID}/{SEASON_ID}.json"
).json()

match_ids = [m["match_id"] for m in matches]

print(f"Found {len(match_ids)} matches")

# Download event data for every match
for match_id in match_ids:

    out_path = f"{EVENTS_DIR}/{match_id}.json"

    # Skip matches already downloaded
    if os.path.exists(out_path):
        continue

    response = requests.get(
        f"{BASE_URL}/events/{match_id}.json"
    )

    if response.status_code == 200:

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(response.json(), f)

        print(f"Saved match {match_id}")

    else:

        print(
            f"FAILED match {match_id}: "
            f"status {response.status_code}"
        )

    time.sleep(0.2)

print("Done.")