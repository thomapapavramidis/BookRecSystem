# Methods for splitting dataset into train and test bins

import numpy as np
import pandas as pd

# Produces train/test split so that every user has data points in both
# Returns a train dataframe and a test dataframe
def per_user_train_test_split(
    ratings,
    test_size=0.2,
    random_state=42,
    timestamp_col="timestamp",
):
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []

    # Indicates whether timestamps are useable or not
    use_time = timestamp_col in ratings.columns and ratings[timestamp_col].notna().any()

    for _, user_rows in ratings.groupby("user_id", sort=False):
        # If we have only one rating, add to training set
        if len(user_rows) < 2:
            train_parts.append(user_rows)
            continue

        # Calculate index for test split
        n_test = int(round(len(user_rows) * test_size))
        n_test = max(1, min(n_test, len(user_rows) - 1))

        if use_time:
            # If using timestamps, test on most recent books
            # This mimics the nature of book recommendations
            ordered = _sort_by_time(user_rows, timestamp_col)
            train_parts.append(ordered.iloc[:-n_test])
            test_parts.append(ordered.iloc[-n_test:])
        else:
            # If not using timestamps, split randomly
            indices = user_rows.index.to_numpy().copy()
            rng.shuffle(indices)
            test_idx = set(indices[:n_test])
            test_parts.append(user_rows.loc[list(test_idx)])
            train_parts.append(user_rows.drop(index=test_idx))

    # Collect train and test splits for each user into two dataframes
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# Re-orders list of timestamps from oldest to most recent
def _sort_by_time(rows, timestamp_col):
    rows = rows.copy()

    # If timestamps are numeric, just sort
    if pd.api.types.is_numeric_dtype(rows[timestamp_col]):
        return rows.sort_values(timestamp_col)

    # Otherwise convert to datetimes, then sort
    rows["_parsed_time"] = pd.to_datetime(rows[timestamp_col], errors="coerce")
    rows = rows.sort_values("_parsed_time", na_position="first")
    return rows.drop(columns=["_parsed_time"])
