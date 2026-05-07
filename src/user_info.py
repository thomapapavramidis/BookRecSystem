import argparse
import sys
from collections import defaultdict

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def main():
    parser = argparse.ArgumentParser(description="Show rating info for a user.")
    parser.add_argument("--user_id", required=True, type=int)
    parser.add_argument("--ratings_path", default="data/ratings.csv")
    parser.add_argument("--books_path", default="data/books.csv")
    parser.add_argument("--books_genres_path", default="data/books_with_genres.csv")
    args = parser.parse_args()

    ratings = pd.read_csv(args.ratings_path)
    books = pd.read_csv(args.books_path)
    books_genres = pd.read_csv(args.books_genres_path)

    user_ratings = ratings[ratings["user_id"] == args.user_id]

    if user_ratings.empty:
        print(f"No ratings found for user {args.user_id}.")
        return

    merged = user_ratings.merge(books[["book_id", "title", "authors"]], on="book_id", how="left")
    merged = merged.merge(books_genres[["book_id", "genres"]], on="book_id", how="left")

    print(f"User {args.user_id} — {len(merged)} books rated")
    print(f"Mean rating:   {merged['rating'].mean():.2f}")
    print(f"Rating breakdown: { {r: int((merged['rating']==r).sum()) for r in range(1,6)} }")

    # Genre preference: average rating per genre vs user mean
    user_mean = merged["rating"].mean()
    genre_sums = defaultdict(float)
    genre_counts = defaultdict(int)
    for row in merged.itertuples(index=False):
        if pd.isna(row.genres) or not row.genres:
            continue
        for genre in [g.strip() for g in str(row.genres).split(",")]:
            if genre:
                genre_sums[genre] += row.rating - user_mean
                genre_counts[genre] += 1

    if genre_counts:
        genre_scores = {g: genre_sums[g] / genre_counts[g] for g in genre_counts}
        genre_df = pd.DataFrame([
            {"genre": g, "avg_deviation": round(s, 2), "books_read": genre_counts[g]}
            for g, s in sorted(genre_scores.items(), key=lambda x: -x[1])
        ])
        print("\nGenre Preferences (deviation from your mean rating)")
        print(genre_df.to_string(index=False))

    print()
    print(merged[["title", "authors", "rating", "genres"]].sort_values("rating", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
