import ast
import re
from pathlib import Path

import pandas as pd

# Standardizes column name (no whitespace, lowercase, underscores)
def _clean_column_name(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")

# Cleans all columns in dataframe
def _standardize_columns(df):
    df = df.copy()
    df.columns = [_clean_column_name(col) for col in df.columns]
    return df

# Corrects first candidate column name in dataframe to target label
def _rename_first_match(df, target, candidates):
    if target in df.columns:
        return df
    for col in candidates:
        if col in df.columns:
            return df.rename(columns={col: target})
    return df

# Collects genres from potentially different formats, standardizes genre names and removes duplicates
# Returns a list of genres
def parse_genres(value):
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return []

    # Handle different input formats
    if isinstance(value, dict):
        raw_genres = list(value.keys())
    elif isinstance(value, (list, tuple, set)):
        raw_genres = list(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return []

        try:
            parsed = ast.literal_eval(text)
            if parsed != text:
                return parse_genres(parsed)
        except (ValueError, SyntaxError):
            pass

        text = text.strip("[](){}")
        raw_genres = re.split(r"[,;|/]+", text)

    # Standardize genre names
    cleaned = []
    for genre in raw_genres:
        genre = str(genre).strip().strip("'\"").lower()
        if genre and genre not in {"nan", "none", "null"}:
            cleaned.append(genre)

    # Removes duplicates
    return list(dict.fromkeys(cleaned))

# Extracts and cleans wanted data from csv file
# Returns a ratings dataframe with user, book, and ratings columns (and timestamps if available)
def load_ratings_csv(path):
    path = Path(path)
    ratings = pd.read_csv(path)
    ratings = _standardize_columns(ratings)

    # Replace varying column labels with user_id, book_id, rating, and timestamp
    ratings = _rename_first_match(
        ratings,
        "user_id",
        ["userid", "user", "reader_id", "readerid", "customer_id"],
    )
    ratings = _rename_first_match(
        ratings,
        "book_id",
        ["bookid", "book", "item_id", "itemid", "isbn", "isbn13", "work_id"],
    )
    ratings = _rename_first_match(
        ratings,
        "rating",
        ["rate", "score", "stars", "user_rating"],
    )
    ratings = _rename_first_match(
        ratings,
        "timestamp",
        ["time", "date", "created_at", "rated_at"],
    )

    # Make sure we have essential columns (user, book, and given rating)
    required = ["user_id", "book_id", "rating"]
    missing = [col for col in required if col not in ratings.columns]
    if missing:
        raise ValueError(f"Ratings file is missing required columns: {missing}")

    # Keep timestamps if we have them
    keep_cols = required + (["timestamp"] if "timestamp" in ratings.columns else [])
    ratings = ratings[keep_cols].copy()

    # Drop any entries with missing values for user, book, or rating
    ratings = ratings.dropna(subset=required)

    # Drop entries with empty strings for user or book
    ratings["user_id"] = ratings["user_id"].astype(str).str.strip()
    ratings["book_id"] = ratings["book_id"].astype(str).str.strip()
    ratings = ratings[(ratings["user_id"] != "") & (ratings["book_id"] != "")]

    # Drop entries with invalid ratings
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])
    ratings = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)]

    return ratings.reset_index(drop=True)

# Extracts book metadata from csv file
# Returns a books dataframe with book_id, title, author, and genres_list columns
def load_books_csv(path):
    path = Path(path)
    books = pd.read_csv(path)
    books = _standardize_columns(books)

    # Replace varying column labels with book_id, title, author, and genres
    books = _rename_first_match(
        books,
        "book_id",
        ["bookid", "book", "item_id", "itemid", "isbn", "isbn13", "work_id"],
    )
    books = _rename_first_match(books, "title", ["book_title", "name"])
    books = _rename_first_match(books, "author", ["authors", "book_author"])
    books = _rename_first_match(
        books,
        "genres",
        ["genre", "categories", "category", "shelves", "tags"],
    )

    # Make sure we have essential column (book_id)
    if "book_id" not in books.columns:
        raise ValueError("Books file is missing required column: book_id")

    # Fill with empty value if info not available
    for col in ["title", "author", "genres"]:
        if col not in books.columns:
            books[col] = pd.NA

    # Select wanted columns, drop entries missing valid book_id, remove duplicates, and clean genres
    books = books[["book_id", "title", "author", "genres"]].copy()
    books = books.dropna(subset=["book_id"])
    books["book_id"] = books["book_id"].astype(str).str.strip()
    books = books[books["book_id"] != ""]
    books = books.drop_duplicates(subset=["book_id"])
    books["genres_list"] = books["genres"].apply(parse_genres)

    return books.reset_index(drop=True)

# To help with array sparsity, remove users who have rated too few books and book which have to few ratings
# Default threshold of 5 ratings
# Returns filtered ratings dataframe
def filter_sparse_ratings(ratings, min_user_ratings=5, min_book_ratings=5):
    filtered = ratings.copy()

    while not filtered.empty:
        start_rows = len(filtered)

        # Select users with enough ratings
        user_counts = filtered["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_ratings].index
        filtered = filtered[filtered["user_id"].isin(keep_users)]

        # Select books with enough ratings
        book_counts = filtered["book_id"].value_counts()
        keep_books = book_counts[book_counts >= min_book_ratings].index
        filtered = filtered[filtered["book_id"].isin(keep_books)]

        # We are done if the filtering process did not remove anything
        if len(filtered) == start_rows:
            break

    return filtered.reset_index(drop=True)
