import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data import filter_sparse_ratings, load_books_csv, load_ratings_csv
    from .evaluation import evaluate_model
    from .models import GenreModel, HybridModel, MFModel, PopularityModel
    from .recommend import recommend_for_user
    from .split import per_user_train_test_split
except ImportError:  # Allows running python src/main.py
    from data import filter_sparse_ratings, load_books_csv, load_ratings_csv
    from evaluation import evaluate_model
    from models import GenreModel, HybridModel, MFModel, PopularityModel
    from recommend import recommend_for_user
    from split import per_user_train_test_split


def main():
    args = parse_args()

    ratings = load_ratings_csv(args.ratings_path)
    books = load_books(args.books_path)

    ratings = filter_sparse_ratings(
        ratings,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
    )
    if ratings.empty:
        raise ValueError("No ratings remain after sparse user/book filtering.")

    train_df, test_df = per_user_train_test_split(
        ratings,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Ratings after filtering: {len(ratings):,}")
    print(f"Train ratings: {len(train_df):,}")
    print(f"Test ratings: {len(test_df):,}")

    models = {}
    results = []

    popularity = PopularityModel(smoothing=args.popularity_smoothing).fit(train_df, books)
    add_result("Popularity", popularity, models, results, train_df, test_df, books, args.random_state)

    genre_model = None
    if has_genre_metadata(books):
        genre_model = GenreModel().fit(train_df, books)
        add_result("Genre", genre_model, models, results, train_df, test_df, books, args.random_state)
    else:
        print("Skipping genre and hybrid models because usable genre metadata was not provided.")

    mf_scores = {}
    for n_factors in [20, 50, 100]:
        name = f"MF-{n_factors}"
        mf_model = MFModel(
            n_factors=n_factors,
            n_epochs=args.n_epochs,
            lr_all=args.lr_all,
            reg_all=args.reg_all,
            random_state=args.random_state,
        ).fit(train_df, books)
        if mf_model.backend == "sgd" and hasattr(mf_model, "surprise_error"):
            print(f"{name}: Surprise unavailable, using SGD fallback.")
        add_result(name, mf_model, models, results, train_df, test_df, books, args.random_state)
        mf_scores[name] = results[-1]["RMSE"]

    best_mf_name = min(mf_scores, key=lambda name: _nan_to_inf(mf_scores[name]))
    best_mf = models[best_mf_name]

    if genre_model is not None:
        for alpha in [0.6, 0.7, 0.8, 0.9]:
            name = f"Hybrid-{alpha:.1f}"
            hybrid = HybridModel(best_mf, genre_model, alpha=alpha)
            add_result(name, hybrid, models, results, train_df, test_df, books, args.random_state)

    results_df = pd.DataFrame(results)
    print("\nResults")
    print(format_results(results_df))

    best_model_name = pick_best_model(results_df)
    print(f"\nExample recommendations using {best_model_name}")
    for user_id in train_df["user_id"].drop_duplicates().head(3):
        print(f"\nUser {user_id}")
        recs = recommend_for_user(
            user_id,
            best_model_name,
            top_k=args.top_k,
            models=models,
            train_df=train_df,
            books_df=books,
        )
        print(recs.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Run book recommendation experiments.")
    parser.add_argument("--ratings_path", required=True, help="Path to ratings CSV.")
    parser.add_argument("--books_path", default=None, help="Optional path to book metadata CSV.")
    parser.add_argument("--min_user_ratings", type=int, default=5)
    parser.add_argument("--min_book_ratings", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--popularity_smoothing", type=float, default=10)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr_all", type=float, default=0.005)
    parser.add_argument("--reg_all", type=float, default=0.02)
    return parser.parse_args()


def load_books(path):
    if not path:
        return None

    path = Path(path)
    if not path.exists():
        print(f"Books metadata file not found: {path}. Continuing without metadata.")
        return None

    return load_books_csv(path)


def has_genre_metadata(books):
    if books is None or "genres_list" not in books.columns:
        return False
    return books["genres_list"].map(len).sum() > 0


def add_result(name, model, models, results, train_df, test_df, books, random_state):
    print(f"Evaluating {name}...")
    models[name] = model
    row = {"Model": name}
    row.update(evaluate_model(model, train_df, test_df, books_df=books, random_state=random_state))
    results.append(row)


def format_results(results_df):
    metric_cols = ["RMSE", "MAE", "HitRate@10", "Precision@10", "Recall@10", "NDCG@10"]
    ordered_cols = ["Model"] + metric_cols
    formatted = results_df[ordered_cols].copy()
    for col in metric_cols:
        formatted[col] = formatted[col].map(lambda value: "nan" if pd.isna(value) else f"{value:.4f}")
    return formatted.to_string(index=False)


def pick_best_model(results_df):
    usable = results_df.dropna(subset=["RMSE"])
    if usable.empty:
        return results_df.iloc[0]["Model"]
    return usable.sort_values("RMSE").iloc[0]["Model"]


def _nan_to_inf(value):
    return np.inf if pd.isna(value) else value


if __name__ == "__main__":
    main()
