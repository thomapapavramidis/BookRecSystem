# Methods to evaluate accuracy of model and helpfulness of ranked book prediction scores

import numpy as np

# Root mean squared error loss function
def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Mean absolute error loss function
def mae(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

# Makes predictions on test data and executes scoring
# Returns RMSE and MAE values
def evaluate_rating_predictions(model, test_df):
    if test_df.empty:
        return {"RMSE": np.nan, "MAE": np.nan}

    y_true = test_df["rating"].to_numpy(dtype=float)
    y_pred = [
        model.predict(row.user_id, row.book_id)
        for row in test_df.itertuples(index=False)
    ]
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred)}

# Predicts on curated book set and calculates several metrics and to determine model effectiveness
def evaluate_ranking(
    model,
    train_df,
    test_df,
    books_df=None,
    top_k=10,
    liked_threshold=4,
    negative_sample_size=100,
    random_state=42,
):
    rng = np.random.default_rng(random_state)
    
    # Make set of all books
    all_books = set(train_df["book_id"].astype(str))
    if books_df is not None:
        all_books.update(books_df["book_id"].astype(str))
    if hasattr(model, "all_book_ids"):
        all_books.update(str(book_id) for book_id in model.all_book_ids)

    # Group samples by user
    train_by_user = {
        str(user): set(group["book_id"].astype(str))
        for user, group in train_df.groupby("user_id")
    }

    # Calculate evaluation metrics for each user
    metrics = []
    for user, user_test in test_df.groupby("user_id"):
        # Get user and collect books user likes (rating passes liked_threshold)
        user = str(user)
        positives = set(
            user_test.loc[user_test["rating"] >= liked_threshold, "book_id"].astype(str)
        )
        if not positives:
            continue

        # Remove books in training set from prediction candidates
        rated_train = train_by_user.get(user, set())
        candidates = all_books - rated_train
        candidates.update(positives)

        # Choose set of non-liked books of specified size
        negatives = sorted(candidates - positives)
        if negative_sample_size is not None and len(negatives) > negative_sample_size:
            sample_idx = rng.choice(len(negatives), size=negative_sample_size, replace=False)
            negatives = [negatives[idx] for idx in sample_idx]

        # Predict on combined liked and non-liked books and sort by score
        candidate_books = list(positives) + negatives
        scores = model.score_items(user, candidate_books)
        ranked_books = [
            book_id
            for book_id, _ in sorted(
                zip(candidate_books, scores),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]

        # Calculate metrics for top k books
        top_books = ranked_books[:top_k]
        hits = sum(1 for book_id in top_books if book_id in positives)
        precision = hits / top_k
        recall = hits / len(positives)
        hit_rate = 1.0 if hits > 0 else 0.0
        ndcg = _ndcg_at_k(top_books, positives, top_k)

        metrics.append(
            {
                "HitRate@10": hit_rate,
                "Precision@10": precision,
                "Recall@10": recall,
                "NDCG@10": ndcg,
            }
        )

    if not metrics:
        return {
            "HitRate@10": np.nan,
            "Precision@10": np.nan,
            "Recall@10": np.nan,
            "NDCG@10": np.nan,
        }

    return {
        key: float(np.mean([row[key] for row in metrics]))
        for key in metrics[0]
    }

# Normalized discounted cumulative gain: rewards when positive items are ranked higher
def _ndcg_at_k(top_books, positives, top_k):
    # Calculate discounted cumulative gain
    dcg = 0.0
    for rank, book_id in enumerate(top_books[:top_k], start=1):
        if book_id in positives:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(positives), top_k)
    if ideal_hits == 0:
        return 0.0

    # Calculate maximum discounted cumulative gain and return normalized value
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg)

# Evaluates loss and ranking effectiveness for a given model, training set, and testing set
# Returns values of losses and ranking metrics
def evaluate_model(model, train_df, test_df, books_df=None, random_state=42):
    results = evaluate_rating_predictions(model, test_df)
    results.update(
        evaluate_ranking(
            model,
            train_df,
            test_df,
            books_df=books_df,
            random_state=random_state,
        )
    )
    return results
