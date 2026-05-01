# Book Recommender

This project builds a simple book recommendation system for an intro machine
learning class. Given historical user-book ratings, it predicts ratings and
ranks books a user might like.

## Expected Data

Ratings CSV is required and should contain at least:

| user_id | book_id | rating |
| --- | --- | --- |
| 1 | 101 | 5 |
| 1 | 205 | 3 |

Ratings should be in the range 1 to 5. If a timestamp/date column is present,
the train/test split uses it so earlier ratings are used for training and later
ratings are used for testing.

Book metadata CSV is optional and may contain:

| book_id | title | author | genres |
| --- | --- | --- | --- |
| 101 | The Hobbit | J.R.R. Tolkien | fantasy, adventure |

Genre values can be comma-separated strings, lists, or similar simple formats.
Review text is ignored even if it appears in the dataset.

## Install

```bash
cd book_recommender
pip install -r requirements.txt
```

`scikit-surprise` is listed because the matrix factorization model prefers the
Surprise SVD implementation. If Surprise is not available in your environment,
the code automatically falls back to a small SGD matrix factorization model.

## Run

```bash
python src/main.py --ratings_path data/ratings.csv --books_path data/books.csv
```

`--books_path` is optional:

```bash
python src/main.py --ratings_path data/ratings.csv
```

Useful options:

```bash
python src/main.py \
  --ratings_path data/ratings.csv \
  --books_path data/books.csv \
  --min_user_ratings 5 \
  --min_book_ratings 5 \
  --random_state 42
```

The script loads and filters the data, creates a per-user train/test split,
trains several models, prints an evaluation table, and prints example
recommendations for three users.

## Models

**Popularity baseline** ranks books by a Bayesian weighted average rating. This
checks whether personalization helps.

**Genre baseline** is an interpretable content-based model. It learns whether a
user rates each genre above or below that user's average rating.

**Matrix factorization** learns latent user and book vectors from ratings. It
captures collaborative patterns that are not explicitly present in metadata.

**Hybrid model** combines matrix factorization and genre preferences:

```text
final_score = alpha * MF_score + (1 - alpha) * genre_score
```

## Metrics

**RMSE** and **MAE** measure rating prediction error. Lower is better.

**HitRate@10** is 1 for a user if at least one liked held-out book appears in
the top 10 recommendations.

**Precision@10** measures what fraction of the top 10 recommendations are liked
held-out books.

**Recall@10** measures what fraction of a user's liked held-out books were
retrieved in the top 10.

**NDCG@10** rewards putting liked books higher in the ranked list.

A liked book is defined as a test rating of 4 or higher.

## Project Structure

```text
book_recommender/
  README.md
  requirements.txt
  src/
    data.py
    split.py
    models.py
    evaluation.py
    recommend.py
    main.py
```

## Known Limitations

- Cold-start users and books are filtered out instead of handled directly.
- Review text is ignored.
- Genre labels are crude and may be noisy.
- Matrix factorization needs rating history to learn useful user/book vectors.
- This is an academic ML project, not a production recommender service.
