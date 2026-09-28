# Data came from Statsbomb

# Link: https://github.com/hudl/open-data/tree/master/data/events

# LIBRARIES

import json
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, train_test_split


# SETTINGS

# Minimum number of actions a player needs to appear in the final ranking
MIN_ACTIONS = 500


# FUNCTION: CALCULATE GOAL ANGLE

# Approximate StatsBomb goalposts:
# left post  = (120, 36)
# right post = (120, 44)

# Goal angle represents how much of the goal mouth is visible from the ball's current location.

def goal_angle(x, y):

    # Vector from ball to left goalpost
    v1_x = 120 - x
    v1_y = 36 - y

    # Vector from ball to right goalpost
    v2_x = 120 - x
    v2_y = 44 - y

    # Cross product and dot product
    cross = np.abs(v1_x * v2_y - v1_y * v2_x)
    dot = v1_x * v2_x + v1_y * v2_y

    # Angle between the two vectors
    return np.arctan2(cross, dot)


# FUNCTION: PREPARE ONE MATCH

# This function takes the raw StatsBomb JSON for one match and performs all of our preprocessing.

# The reason for putting this into a function is that we run the exact same process on many matches.

def prepare_match(data, match_id):

    df = pd.json_normalize(data)

    # This lets us identify which match every event came from

    df["match_id"] = match_id

    # Put events in chronological order

    df = df.sort_values("index").reset_index(drop=True)

    # Separate starting x and y coordinates

    # StatsBomb location looks like:
    # [61.0, 40.1]
    # We want:
    # x = 61.0
    # y = 40.1

    df["x"] = df["location"].apply(
        lambda loc: loc[0] if isinstance(loc, list) else None
    )

    df["y"] = df["location"].apply(
        lambda loc: loc[1] if isinstance(loc, list) else None
    )

    # Remove event types we do not want to value

    remove_types = [
        "Half Start",
        "Starting XI",
        "Half End",
        "Substitution",
        "Tactical Shift"
    ]

    df = df[
        ~df["type.name"].isin(remove_types)
    ].copy()

    # Only keep actions by the team in possession

    # StatsBomb possessions can contain events from both teams.
    # For now, we only want to value actions performed by the team
    # StatsBomb identifies as being in possession.

    df = df[
        df["team.name"] == df["possession_team.name"]
    ].copy()

    # Reset pandas row numbers after filtering
    df = df.reset_index(drop=True)

    # Renumber possessions
    # Keep the original StatsBomb possession number too.

    df["original_possession"] = df["possession"]

    # Half Start / Starting XI were counted in possession 1.
    # Since those were removed, make the first true possession = 1.

    df["possession"] = df["possession"] - 1

    # Number events within each possession

    df["possession_event_num"] = (
        df.groupby("possession").cumcount() + 1
    )

    # Get each action's own ending location

    # We want movement to be attributed to the player/action that
    # actually caused it.
    # Example:
    # Messi passes:
    # (20, 20) -> (50, 50)
    # We want (50, 50) recorded as the endpoint of Messi's pass,
    # rather than borrowing the location from the receiver's next event.

    end_location_cols = [
        "pass.end_location",
        "carry.end_location",
        "shot.end_location",
        "goalkeeper.end_location"
    ]

    end_x = pd.Series(np.nan, index=df.index)
    end_y = pd.Series(np.nan, index=df.index)

    for col in end_location_cols:

        # Some matches may not contain every type of event,
        # so make sure the column exists before using it.
        if col in df.columns:

            end_x = end_x.fillna(df[col].str[0])
            end_y = end_y.fillna(df[col].str[1])

    # Point events with no separately recorded endpoint
    # keep the same start/end location.

    df["end_x"] = end_x.fillna(df["x"]).astype(float)
    df["end_y"] = end_y.fillna(df["y"]).astype(float)

    # Immediate attacking reward

    # Shot -> reward = StatsBomb xG
    # Everything else -> reward = 0
    # We care about shot QUALITY rather than whether the shot happened to result in a goal.

    df["reward"] = df["shot.statsbomb_xg"].fillna(0)

    # Future xG

    # future_xg = total xG generated from the current event through the remainder of the possession.
    # IMPORTANT: future_xg is NOT player value.
    # It is an observed outcome used to teach the model what kinds of states tend to lead to attacking value.
    # Eventually:
    # V(s) = expected future attacking value from state s

    df["future_xg"] = (
        df.groupby(
            ["match_id", "possession", "possession_team.name"]
        )["reward"]
        .transform(
            lambda s: s.iloc[::-1].cumsum().iloc[::-1]
        )
    )

    # Distance to goal
    # Opponent goal is centered approximately at: (120, 40)

    df["distance_to_goal"] = np.sqrt(
        (120 - df["x"])**2 +
        (40 - df["y"])**2
    )

    df["end_distance_to_goal"] = np.sqrt(
        (120 - df["end_x"])**2 +
        (40 - df["end_y"])**2
    )

    df["goal_angle"] = goal_angle(df["x"], df["y"])

    df["end_goal_angle"] = goal_angle(df["end_x"], df["end_y"])

    return df


# LOAD AND PREPARE ALL MATCHES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EVENTS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "events",
    "laliga_2015_16"
)

all_matches = []

for filename in sorted(os.listdir(EVENTS_DIR)):

    # Only load JSON files
    if not filename.endswith(".json"):
        continue

    # Example:
    # 3773369.json -> match_id = 3773369
    file_match_id = filename.replace(".json", "")

    file_path = os.path.join(EVENTS_DIR, filename)

    # Load raw StatsBomb events
    with open(file_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    # Run the same preprocessing on this match
    match_df = prepare_match(match_data, match_id=file_match_id)

    all_matches.append(match_df)

# Combine all matches into one dataframe
df = pd.concat(all_matches, ignore_index=True)

# Check that everything loaded correctly
print("Matches loaded:", df["match_id"].nunique())
print("Total events:", len(df))
print(
    "Total possessions:",
    df.groupby(["match_id", "possession"]).ngroups
)


# FEATURE ENGINEERING FOR V(s)

# V(s) should describe the value of the STATE before the player chooses their action.

# Therefore we only use information that is already known before the action occurs.
#
# We intentionally do NOT include things such as:
#
# type.name
# pass.length
# pass.angle
# pass.end_location
# pass.outcome.name
# dribble.outcome.name
#
# because those describe the action or its result.

# Under pressure
# StatsBomb usually records True when under pressure and leaves
# the value blank otherwise.

df["under_pressure"] = (
    df["under_pressure"]
    .fillna(False)
    .astype(int)
)

# Categorical state feature: play pattern

# Examples:
# Regular Play, From Corner, From Free Kick, From Throw In

play_pattern_dummies = pd.get_dummies(
    df["play_pattern.name"],
    prefix="play_pattern",
    dtype=int
)

# Numeric state features

numeric_features = [
    "x",
    "y",
    "distance_to_goal",
    "goal_angle",
    "possession_event_num",
    "minute",
    "period",
    "under_pressure"
]

# Build feature matrix X

X = pd.concat(
    [
        df[numeric_features].reset_index(drop=True),
        play_pattern_dummies.reset_index(drop=True)
    ],
    axis=1
)

# Target y

# Observed future xG from this point onward in the possession.

y = df["future_xg"].reset_index(drop=True)

# Match IDs

# Needed because we split by entire matches rather than
# randomly splitting individual rows.

match_ids = df["match_id"].reset_index(drop=True)

print("Feature matrix shape:", X.shape)
print("Features:", list(X.columns))


# TRAIN / TEST SPLIT BY MATCH (for evaluating V(s))

unique_matches = match_ids.unique()

train_matches, test_matches = train_test_split(
    unique_matches,
    test_size=0.25,
    random_state=42
)

train_mask = match_ids.isin(train_matches)
test_mask = match_ids.isin(test_matches)

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print()
print(f"Train: {X_train.shape[0]} rows across {len(train_matches)} matches")
print(f"Test: {X_test.shape[0]} rows across {len(test_matches)} matches")


# FIT FIRST V(s) MODEL

# future_xg is non-negative and heavily concentrated at zero,
# so Poisson loss is a reasonable first model to experiment with.

model = HistGradientBoostingRegressor(
    loss="poisson",
    max_iter=200,
    random_state=42
)

model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# MODEL EVALUATION

train_mae = mean_absolute_error(y_train, pred_train)
test_mae = mean_absolute_error(y_test, pred_test)

print()
print("Train MAE:", round(train_mae, 4))
print("Test MAE:", round(test_mae, 4))

# Baseline 1: predict mean training future_xg for everything
mean_baseline = np.full(len(y_test), y_train.mean())
print("Mean baseline test MAE:", round(mean_absolute_error(y_test, mean_baseline), 4))

# Baseline 2: predict zero future_xg for everything
# (especially important because most possessions never produce a shot)
zero_baseline = np.zeros(len(y_test))
print("Zero baseline test MAE:", round(mean_absolute_error(y_test, zero_baseline), 4))


# CHECK WHETHER PREDICTED V(s) RANKS DANGER CORRECTLY

# Bin 1 = lowest predicted V(s), Bin 10 = highest predicted V(s)

value_check = pd.DataFrame({
    "predicted_value": pred_test,
    "actual_future_xg": y_test.to_numpy()
})

value_check["value_bin"] = pd.qcut(
    value_check["predicted_value"],
    q=10,
    labels=False,
    duplicates="drop"
) + 1

value_summary = (
    value_check
    .groupby("value_bin")
    .agg(
        states=("actual_future_xg", "size"),
        mean_predicted_value=("predicted_value", "mean"),
        mean_actual_future_xg=("actual_future_xg", "mean"),
        percent_positive_future_xg=(
            "actual_future_xg",
            lambda s: (s > 0).mean() * 100
        )
    )
    .reset_index()
)

print()
print("Predicted State Value Check")
print(value_summary)


# TARGET DISTRIBUTION CHECK

print()
print("Total events:", len(df))
print("Events with future_xg > 0:", (df["future_xg"] > 0).sum())
print(
    "Percent of events with future_xg > 0:",
    round((df["future_xg"] > 0).mean() * 100, 2),
    "%"
)
print(
    "Possessions producing xG:",
    df.groupby(["match_id", "possession"])["reward"].sum().gt(0).sum()
)


# ===============================================================
# ACTION VALUE SCORING
#
# action_value = V(state AFTER the action) - V(state BEFORE it)
#
# The value is credited to the player who performed the action.
# ===============================================================

# BUILD AFTER-STATE FEATURES

# Same columns as X, but built from where each action ENDED.
# minute / period / under_pressure / play_pattern carry over unchanged,
# since we do not observe an "after" version of those.
# possession_event_num moves forward one step.

X_after = X.copy()

X_after["x"] = df["end_x"].reset_index(drop=True)
X_after["y"] = df["end_y"].reset_index(drop=True)
X_after["distance_to_goal"] = df["end_distance_to_goal"].reset_index(drop=True)
X_after["goal_angle"] = df["end_goal_angle"].reset_index(drop=True)
X_after["possession_event_num"] = X["possession_event_num"] + 1


# OUT-OF-FOLD SCORING

# Every match is scored by a model that never trained on it.
# If we scored matches with the model that trained on them, V would
# partly memorize their possessions and create fake action values.

V_before = np.zeros(len(X))
V_after = np.zeros(len(X))

gkf = GroupKFold(n_splits=5)

for fold_train_idx, fold_test_idx in gkf.split(X, y, groups=match_ids):

    fold_model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=200,
        random_state=42
    )

    fold_model.fit(
        X.iloc[fold_train_idx],
        y.iloc[fold_train_idx]
    )

    V_before[fold_test_idx] = fold_model.predict(X.iloc[fold_test_idx])
    V_after[fold_test_idx] = fold_model.predict(X_after.iloc[fold_test_idx])


# TERMINAL OUTCOMES

# When an action's result is already known, V(after) should not come
# from a hypothetical location.

# A shot is worth its own xG.
is_shot = (df["type.name"] == "Shot").to_numpy()
V_after[is_shot] = df.loc[is_shot, "reward"].to_numpy()

# A failed pass loses the ball. pass.end_location records where the pass
# was AIMED, so without this an incomplete through ball would be credited
# for the dangerous spot it was aimed at.
if "pass.outcome.name" in df.columns:
    failed_pass = df["pass.outcome.name"].notna().to_numpy()
    V_after[failed_pass] = 0

df["V_before"] = V_before
df["V_after"] = V_after
df["action_value"] = df["V_after"] - df["V_before"]

print()
print("action_value summary:")
print(df["action_value"].describe())


# AGGREGATE TO PLAYER LEVEL

player_summary = (
    df.groupby(["player.name", "team.name"])
    .agg(
        actions=("action_value", "size"),
        total_value=("action_value", "sum"),
        value_per_action=("action_value", "mean")
    )
    .reset_index()
)

ranked = player_summary[player_summary["actions"] >= MIN_ACTIONS]

print()
print(f"Top 20 players by total attacking value (min {MIN_ACTIONS} actions):")
print(ranked.sort_values("total_value", ascending=False).head(20))

print()
print(f"Top 20 players by value per action (min {MIN_ACTIONS} actions):")
print(ranked.sort_values("value_per_action", ascending=False).head(20))