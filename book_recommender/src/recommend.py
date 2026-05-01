import pandas as pd


def recommend_for_user(user_id, model_name, top_k=10, *, models, train_df, books_df=None):
    """Return top-k recommendations for one user from a fitted model dictionary."""
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {list(models)}")

    model = models[model_name]
    user_id = str(user_id).strip()
    rated_books = set(
        train_df.loc[train_df["user_id"].astype(str) == user_id, "book_id"].astype(str)
    )

    candidate_books = set(train_df["book_id"].astype(str))
    if books_df is not None:
        candidate_books.update(books_df["book_id"].astype(str))
    if hasattr(model, "all_book_ids"):
        candidate_books.update(str(book_id) for book_id in model.all_book_ids)

    candidate_books = sorted(candidate_books - rated_books)
    if not candidate_books:
        return pd.DataFrame(columns=["book_id", "predicted_score"])

    scores = model.score_items(user_id, candidate_books)
    recs = pd.DataFrame({"book_id": candidate_books, "predicted_score": scores})
    recs = recs.sort_values("predicted_score", ascending=False).head(top_k)

    if books_df is not None:
        metadata_cols = [
            col for col in ["book_id", "title", "author", "genres"] if col in books_df.columns
        ]
        recs = recs.merge(books_df[metadata_cols], on="book_id", how="left")

    display_cols = ["book_id"]
    for col in ["title", "author", "genres"]:
        if col in recs.columns:
            display_cols.append(col)
    display_cols.append("predicted_score")

    return recs[display_cols].reset_index(drop=True)
