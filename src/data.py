import ast
import re
from pathlib import Path

import pandas as pd


def _clean_column_name(name):
    """Normalize common CSV column naming styles."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _standardize_columns(df):
    df = df.copy()
    df.columns = [_clean_column_name(col) for col in df.columns]
    return df


def _rename_first_match(df, target, candidates):
    if target in df.columns:
        return df
    for col in candidates:
        if col in df.columns:
            return df.rename(columns={col: target})
    return df


def parse_genres(value):
    """Return a clean list of genre strings from common CSV encodings."""
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return []

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

    cleaned = []
    for genre in raw_genres:
        genre = str(genre).strip().strip("'\"").lower()
        if genre and genre not in {"nan", "none", "null"}:
            cleaned.append(genre)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(cleaned))


def load_ratings_csv(path):
    """Load ratings and keep the columns needed by the recommender."""
    path = Path(path)
    ratings = pd.read_csv(path)
    ratings = _standardize_columns(ratings)

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

    required = ["user_id", "book_id", "rating"]
    missing = [col for col in required if col not in ratings.columns]
    if missing:
        raise ValueError(f"Ratings file is missing required columns: {missing}")

    keep_cols = required + (["timestamp"] if "timestamp" in ratings.columns else [])
    ratings = ratings[keep_cols].copy()
    ratings = ratings.dropna(subset=required)

    ratings["user_id"] = ratings["user_id"].astype(str).str.strip()
    ratings["book_id"] = ratings["book_id"].astype(str).str.strip()
    ratings = ratings[(ratings["user_id"] != "") & (ratings["book_id"] != "")]

    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])
    ratings = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)]

    return ratings.reset_index(drop=True)


def load_books_csv(path):
    """Load book metadata. Missing optional fields are filled with blanks."""
    path = Path(path)
    books = pd.read_csv(path)
    books = _standardize_columns(books)

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

    if "book_id" not in books.columns:
        raise ValueError("Books file is missing required column: book_id")

    for col in ["title", "author", "genres"]:
        if col not in books.columns:
            books[col] = pd.NA

    books = books[["book_id", "title", "author", "genres"]].copy()
    books = books.dropna(subset=["book_id"])
    books["book_id"] = books["book_id"].astype(str).str.strip()
    books = books[books["book_id"] != ""]
    books = books.drop_duplicates(subset=["book_id"])
    books["genres_list"] = books["genres"].apply(parse_genres)

    return books.reset_index(drop=True)


def filter_sparse_ratings(ratings, min_user_ratings=5, min_book_ratings=5):
    """Iteratively remove sparse users/books until both constraints hold."""
    filtered = ratings.copy()

    while not filtered.empty:
        start_rows = len(filtered)

        user_counts = filtered["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_ratings].index
        filtered = filtered[filtered["user_id"].isin(keep_users)]

        book_counts = filtered["book_id"].value_counts()
        keep_books = book_counts[book_counts >= min_book_ratings].index
        filtered = filtered[filtered["book_id"].isin(keep_books)]

        if len(filtered) == start_rows:
            break

    return filtered.reset_index(drop=True)
