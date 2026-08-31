# Load in libraries and match data

import json
import pandas as pd
import numpy as np

with open("Single Match Data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)

# Index to make sure we have correct order of events

df = df.sort_values("index").reset_index(drop=True)

# Don't want location like this [61.0, 40.1]. Want a separate x and y column

df["x"] = df["location"].apply(
    lambda loc: loc[0] if isinstance(loc, list) else None
)

df["y"] = df["location"].apply(
    lambda loc: loc[1] if isinstance(loc, list) else None
)

# Take out useless values

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

# Half Start and Starting XI counted as possession 1.
# Make the first true possession = 1

df["possession"] = df["possession"] - 1

# Number events within each possession performed by team in possession

df["possession_event_num"] = (
    df.groupby("possession").cumcount() + 1
)

# Get each action's OWN end location - where the ball actually ended up
# because of this action - rather than inferring it from wherever the
# next logged row happens to start.
# e.g. if Messi passes from (20, 20) to (50, 50), we want (50, 50)
# attributed to Messi's pass directly, not borrowed from the next event.

end_location_cols = [
    "pass.end_location",
    "carry.end_location",
    "shot.end_location",
    "goalkeeper.end_location"
]

end_x = pd.Series(np.nan, index=df.index)
end_y = pd.Series(np.nan, index=df.index)

for col in end_location_cols:
    end_x = end_x.fillna(df[col].str[0])
    end_y = end_y.fillna(df[col].str[1])

# No end location for this type -> nothing moved

df["end_x"] = end_x.fillna(df["x"]).astype(float)
df["end_y"] = end_y.fillna(df["y"]).astype(float)


# Shot -> reward = StatsBomb xG
# Everything else -> reward = 0

# We use shot quality (xG), NOT whether the shot actually went in.

# A 0.70 xG shot is still a valuable opportunity
# even if the player misses.

df["reward"] = (
    df["shot.statsbomb_xg"]
    .fillna(0)
)

# future_xg = total xG generated from the current event
# through the remainder of the possession.

# future_xg is NOT the player's value.

# It is the observed outcome we will use to teach a model
# what dangerous states tend to look like.

# Eventually: V(s) = expected future attacking value from state s

df["future_xg"] = (
    df.groupby(
        ["possession", "possession_team.name"]
    )["reward"]
    .transform(
        lambda s: s.iloc[::-1].cumsum().iloc[::-1]
    )
)


df["distance_to_goal"] = np.sqrt(
    (120 - df["x"])**2 +
    (40 - df["y"])**2
)

df["end_distance_to_goal"] = np.sqrt(
    (120 - df["end_x"])**2 +
    (40 - df["end_y"])**2
)

# Approximate goalposts in StatsBomb coordinates:
# (120, 36) and (120, 44)
# Calculate angle to goal - vectorized, no row-wise apply needed
# since arithmetic + np.arctan2 already work on whole Series at once

def goal_angle(x, y):
    v1_x = 120 - x
    v1_y = 36 - y

    v2_x = 120 - x
    v2_y = 44 - y

    cross = np.abs(v1_x * v2_y - v1_y * v2_x)
    dot = v1_x * v2_x + v1_y * v2_y

    return np.arctan2(cross, dot)

df["goal_angle"] = goal_angle(df["x"], df["y"])
df["end_goal_angle"] = goal_angle(df["end_x"], df["end_y"])