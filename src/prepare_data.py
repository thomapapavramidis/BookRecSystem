"""
Merge books.csv + book_tags.csv + tags.csv into books_with_genres.csv.
Run once before main.py to enable the genre and hybrid models.
"""
import argparse
from pathlib import Path

import pandas as pd

SHELF_TAGS = {
    "to-read", "currently-reading", "read", "favorites", "owned",
    "books-i-own", "default", "maybe", "wish-list", "library",
    "kindle", "audiobook", "ebook", "re-read", "did-not-finish",
    "dnf", "abandoned", "borrowed", "own-it", "owned-books",
}


def prepare(books_path, book_tags_path, tags_path, out_path, min_tag_count=100, top_n=5):
    books = pd.read_csv(books_path)
    book_tags = pd.read_csv(book_tags_path)
    tags = pd.read_csv(tags_path)

    # Keep only tags that pass the count threshold and aren't shelf tags
    tags["tag_name"] = tags["tag_name"].str.strip().str.lower()
    genre_tags = tags[~tags["tag_name"].isin(SHELF_TAGS)]

    merged = book_tags.merge(genre_tags, on="tag_id")
    merged = merged[merged["count"] >= min_tag_count]

    # Top-N tags per goodreads_book_id by count
    merged = merged.sort_values("count", ascending=False)
    top_tags = (
        merged.groupby("goodreads_book_id")["tag_name"]
        .apply(lambda names: list(names.head(top_n)))
        .reset_index()
        .rename(columns={"tag_name": "genres"})
    )

    # Map goodreads_book_id -> book_id via books.csv
    id_map = books[["book_id", "goodreads_book_id"]].drop_duplicates()
    top_tags = top_tags.merge(id_map, on="goodreads_book_id", how="inner")

    result = books[["book_id", "title", "authors"]].copy()
    result = result.rename(columns={"authors": "author"})
    result = result.merge(top_tags[["book_id", "genres"]], on="book_id", how="left")
    result["genres"] = result["genres"].apply(
        lambda g: ", ".join(g) if isinstance(g, list) else ""
    )

    result.to_csv(out_path, index=False)
    n_with_genres = (result["genres"] != "").sum()
    print(f"Wrote {len(result)} books ({n_with_genres} with genres) to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books_path", default="data/books.csv")
    parser.add_argument("--book_tags_path", default="data/book_tags.csv")
    parser.add_argument("--tags_path", default="data/tags.csv")
    parser.add_argument("--out_path", default="data/books_with_genres.csv")
    parser.add_argument("--min_tag_count", type=int, default=100)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()
    prepare(
        args.books_path,
        args.book_tags_path,
        args.tags_path,
        args.out_path,
        args.min_tag_count,
        args.top_n,
    )


if __name__ == "__main__":
    main()
