# Method to calculate and save user rating counts

import argparse
import pandas as pd

# Sums user rating counts and saves to file
def main():
    # Handle command-line input
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_path", required=True)
    parser.add_argument("--out_path", default="data/user_rating_counts.csv")
    parser.add_argument("--top_n", type=int, default=None, help="Save only top N users by count")
    args = parser.parse_args()

    # Get rating counts for each user
    ratings = pd.read_csv(args.ratings_path)
    counts = (
        ratings.groupby("user_id")["book_id"]
        .count()
        .reset_index()
        .rename(columns={"book_id": "books_rated"})
        .sort_values("user_id")
    )

    # Select desired number of most prolific users
    if args.top_n:
        counts = counts.head(args.top_n)

    counts.to_csv(args.out_path, index=False)
    print(f"Saved {len(counts)} users to {args.out_path}")


if __name__ == "__main__":
    main()
