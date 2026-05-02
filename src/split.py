import numpy as np
import pandas as pd


def per_user_train_test_split(
    ratings,
    test_size=0.2,
    random_state=42,
    timestamp_col="timestamp",
):
    """Split each user's ratings so every possible user has train and test rows."""
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []
    use_time = timestamp_col in ratings.columns and ratings[timestamp_col].notna().any()

    for _, user_rows in ratings.groupby("user_id", sort=False):
        if len(user_rows) < 2:
            train_parts.append(user_rows)
            continue

        n_test = int(round(len(user_rows) * test_size))
        n_test = max(1, min(n_test, len(user_rows) - 1))

        if use_time:
            ordered = _sort_by_time(user_rows, timestamp_col)
            train_parts.append(ordered.iloc[:-n_test])
            test_parts.append(ordered.iloc[-n_test:])
        else:
            indices = user_rows.index.to_numpy().copy()
            rng.shuffle(indices)
            test_idx = set(indices[:n_test])
            test_parts.append(user_rows.loc[list(test_idx)])
            train_parts.append(user_rows.drop(index=test_idx))

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _sort_by_time(rows, timestamp_col):
    rows = rows.copy()
    if pd.api.types.is_numeric_dtype(rows[timestamp_col]):
        return rows.sort_values(timestamp_col)

    rows["_parsed_time"] = pd.to_datetime(rows[timestamp_col], errors="coerce")
    rows = rows.sort_values("_parsed_time", na_position="first")
    return rows.drop(columns=["_parsed_time"])
