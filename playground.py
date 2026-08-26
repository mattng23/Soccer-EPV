# Load in libraries and match data

import json
import pandas as pd

with open("Single Match Data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)

# Index to make sure we have correct order of events

df = df.sort_values("index").reset_index(drop=True)

# Don't want location like this [61.0, 40.1]

df["x"] = df["location"].apply(
    lambda loc: loc[0] if isinstance(loc, list) else None
)

df["y"] = df["location"].apply(
    lambda loc: loc[1] if isinstance(loc, list) else None
)

# See what shots look like

shots = df[df["type.name"] == "Shot"].copy()

shots[[
    "team.name",
    "player.name",
    "possession",
    "shot.statsbomb_xg",
    "shot.outcome.name"
]]