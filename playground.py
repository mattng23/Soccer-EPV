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

#Extract columns we want

cols = [
    "index",
    "possession",
    "possession_team.name",
    "team.name",
    "minute",
    "second",
    "player.name",
    "type.name",
    "x",
    "y",
    "shot.statsbomb_xg",
    "shot.outcome.name"
]

df[cols].head(80)

# Take out usesless values

df = df[
    ~df["type.name"].isin([
        "Half Start",
        "Starting XI",
        "Half End",
        "Substitution",
        "Tactical Shift"
    ])
].copy()

# Only want actions done by the team in possession for now

df = df[
    df["team.name"] == df["possession_team.name"]
].copy()

# Half Start and Starting XI counted as possession 1 (which we have now taken out). We want to make the first true possession = 1 

df["possession"] = df["possession"] - 1

# Create location and info for where current event ends (i.e where the next event starts)

df["next_x"] = (
    df.groupby("possession")["x"].shift(-1)
)

df["next_y"] = (
    df.groupby("possession")["y"].shift(-1)
)

df["next_type"] = (
    df.groupby("possession")["type.name"].shift(-1)
)

df["next_player"] = (
    df.groupby("possession")["player.name"].shift(-1)
)

# Incorporate new columns

sequence_cols = [
    "possession",
    "possession_event_num",
    "player.name",
    "type.name",
    "x",
    "y",
    "next_player",
    "next_type",
    "next_x",
    "next_y"
]

df[df["possession"] == 1][sequence_cols]