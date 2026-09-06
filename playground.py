# Data came from Statsbomb

# Link: https://github.com/hudl/open-data/tree/master/data/events

# LIBRARIES

import json
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


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

# The reason for putting this into a function is that eventually we want to run the exact same process on many matches.

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
    
    # For now, we only want to value actions performed by the team StatsBomb identifies as being in possession.

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

            end_x = end_x.fillna(
                df[col].str[0]
            )

            end_y = end_y.fillna(
                df[col].str[1]
            )


    # Point events with no separately recorded endpoint
    # keep the same start/end location.

    df["end_x"] = (
        end_x
        .fillna(df["x"])
        .astype(float)
    )

    df["end_y"] = (
        end_y
        .fillna(df["y"])
        .astype(float)
    )


    # Immediate attacking reward

    # Shot -> reward = StatsBomb xG
    # Everything else -> reward = 0
    # We care about shot QUALITY rather than whether the shot happened to result in a goal.

    df["reward"] = (
        df["shot.statsbomb_xg"]
        .fillna(0)
    )

    # Future xG

    # future_xg = total xG generated from the current event through the remainder of the possession.
    # IMPORTANT: future_xg is NOT player value.
    # It is an observed outcome used to teach the model what kinds of states tend to lead to attacking value.
    # Eventually:
    # V(s) = expected future attacking value from state s

    df["future_xg"] = (
        df.groupby(
            ["match_id","possession", "possession_team.name"]
        )["reward"]
        .transform(
            lambda s:
            s.iloc[::-1].cumsum().iloc[::-1]
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

    df["goal_angle"] = goal_angle(
        df["x"],
        df["y"]
    )

    df["end_goal_angle"] = goal_angle(
        df["end_x"],
        df["end_y"]
    )


    return df


# LOAD CURRENT MATCH

with open(
    "Single Match Data.json", 
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# Run all preprocessing above on this match

df = prepare_match(
    data,
    match_id="match_1"
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
#
# Regular Play
# From Corner
# From Free Kick
# From Throw In

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

y = (
    df["future_xg"]
    .reset_index(drop=True)
)


# Possession IDs

# Needed because we are splitting by possession rather than
# randomly splitting individual rows.

possession_id = (
    df["possession"]
    .reset_index(drop=True)
)


print(
    "Feature matrix shape:",
    X.shape
)

print(
    "Features:",
    list(X.columns)
)


# TRAIN / TEST SPLIT

# IMPORTANT:
# We split by POSSESSION, not by row.

# Events from the same possession are highly related and often share nearly identical future_xg labels.

# If we randomly split rows, events from one possession could appear in both train and test, creating leakage.

# Once we have many matches, we will change this to split by MATCH instead.

unique_poss = possession_id.unique()


train_poss, test_poss = train_test_split(
    unique_poss,
    test_size=0.25,
    random_state=42
)


train_mask = possession_id.isin(
    train_poss
)

test_mask = possession_id.isin(
    test_poss
)


X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]


print()

print(
    f"Train: {X_train.shape[0]} rows "
    f"across {len(train_poss)} possessions"
)

print(
    f"Test: {X_test.shape[0]} rows "
    f"across {len(test_poss)} possessions"
)


# FIT FIRST V(s) MODEL

# future_xg is:
# - non-negative
# - heavily concentrated at zero

# Poisson loss is a reasonable first model to experiment with.
# It is NOT necessarily the final/best model.
#
# Once we have more data, we should compare different loss
# functions and modeling approaches.

model = HistGradientBoostingRegressor(
    loss="poisson",
    max_iter=200,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


# PREDICTIONS

pred_train = model.predict(
    X_train
)

pred_test = model.predict(
    X_test
)


# MODEL EVALUATION

train_mae = mean_absolute_error(
    y_train,
    pred_train
)

test_mae = mean_absolute_error(
    y_test,
    pred_test
)


print()

print(
    "Train MAE:",
    round(train_mae, 4)
)

print(
    "Test MAE:",
    round(test_mae, 4)
)


# Baseline 1: predict mean training future_xg for everything

mean_baseline = np.full(
    len(y_test),
    y_train.mean()
)

mean_baseline_mae = mean_absolute_error(
    y_test,
    mean_baseline
)


print(
    "Mean baseline test MAE:",
    round(mean_baseline_mae, 4)
)


# Baseline 2: predict zero future_xg for everything

# This baseline is especially important because most soccer
# possessions do not result in a shot.

zero_baseline = np.zeros(
    len(y_test)
)

zero_baseline_mae = mean_absolute_error(
    y_test,
    zero_baseline
)


print(
    "Zero baseline test MAE:",
    round(zero_baseline_mae, 4)
)


# TARGET DISTRIBUTION CHECK

# Helpful for understanding how sparse future_xg is.

print()

print(
    "Total events:",
    len(df)
)

print(
    "Events with future_xg > 0:",
    (df["future_xg"] > 0).sum()
)

print(
    "Percent of events with future_xg > 0:",
    round(
        (df["future_xg"] > 0).mean() * 100,
        2
    ),
    "%"
)

print(
    "Total possessions:",
    df["possession"].nunique()
)

print(
    "Possessions producing xG:",
    df.groupby("possession")["reward"]
      .sum()
      .gt(0)
      .sum()
)