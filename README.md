# Overview

This project builds a book recommendation system using machine learning. The goal is to predict which books a user would like based on their past ratings.

Recommender systems are widely used in platforms like Amazon and Netflix to suggest items based on user behavior. We implement and compare several approaches, from simple baselines to more advanced models.

## How to Run

### Prerequisites

- Python 3.8–3.12 (`scikit-surprise` does not support Python 3.13+)
- pip

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

The three required packages are `pandas`, `numpy`, and `scikit-surprise`.

### Step 2 — Run the full pipeline

```bash
python src/main.py --ratings_path data/ratings.csv --books_path data/books_with_genres.csv
```

This single command will:
1. Load and preprocess the data
2. Filter out users and books with fewer than 5 ratings
3. Split each user's ratings 80% train / 20% test
4. Train all models: Popularity, Genre, MF-20, MF-50, MF-100, and Hybrid variants
5. Evaluate every model on the held-out test set
6. Print a results table with RMSE, MAE, HitRate@10, Precision@10, Recall@10, and NDCG@10
7. Print example top-10 recommendations for three users using the best model

Note: `books_with_genres.csv` must be used (not `books.csv`) because it contains the genre column needed for the Genre and Hybrid models.

### Step 3 — (Optional) Run with hyperparameter tuning

```bash
python src/main.py --ratings_path data/ratings.csv --books_path data/books_with_genres.csv --tune
```

Adding `--tune` runs a grid search over SVD hyperparameters and includes a tuned MF model in the comparison. This takes longer to run.

### All available arguments

| Argument | Default | Description |
|---|---|---|
| `--ratings_path` | required | Path to ratings CSV |
| `--books_path` | required | Path to book metadata CSV with genres |
| `--min_user_ratings` | 5 | Minimum ratings a user must have to be included |
| `--min_book_ratings` | 5 | Minimum ratings a book must have to be included |
| `--test_size` | 0.2 | Fraction of each user's ratings held out for testing |
| `--random_state` | 42 | Random seed for reproducibility |
| `--top_k` | 10 | Number of recommendations to generate per user |
| `--n_epochs` | 20 | Training epochs for matrix factorization |
| `--lr_all` | 0.005 | Learning rate for matrix factorization |
| `--reg_all` | 0.02 | L2 regularization for matrix factorization |
| `--tune` | False | Run SVD hyperparameter grid search |

## Problem Definition

Given:

- A dataset of user-book ratings
- Book metadata with genre tags

We want to:

- Predict how much a user would like a book
- Recommend the top-N books they are most likely to enjoy

## Dataset Format

Ratings dataset:

| user_id | book_id | rating |
| --- | --- | --- |
| 1 | 101 | 5 |
| 1 | 205 | 3 |

Ratings are between 1 and 5.

Book metadata:

| book_id | title | author | genres |
| --- | --- | --- | --- |
| 101 | ... | ... | fantasy, adventure |

A books CSV with a genres column is required to run the Genre and Hybrid models. The provided `books_with_genres.csv` already has genres populated — it was generated from the raw `books.csv`, `book_tags.csv`, and `tags.csv` files using `src/prepare_data.py`.

## Approach

We implement three main models + one hybrid model.

### 1. Popularity Baseline

- Recommends the highest-rated books overall
- Same recommendations for all users

Why: simple baseline that checks whether personalization actually helps.

### 2. Genre-Based Model (Content-Based)

- Builds a genre preference profile per user
- Uses ratings, not just counts

For example, if a user rates fantasy books higher than their average, fantasy gets a high score. The model then recommends books with genres the user tends to rate highly.

Why: simple and interpretable, and works even for new books since it relies on metadata.

### 3. Matrix Factorization (Collaborative Filtering)

Learns latent features for users and books. Predicts ratings using:

```
predicted_rating = user_vector · book_vector
```

This captures hidden patterns in user preferences that are not explicitly labeled. Collaborative filtering works by finding patterns across many users rather than relying on item metadata.

Why: strong performance and the standard approach in recommender systems.

### 4. Hybrid Model

Combines both approaches:

```
final_score = α * matrix_factorization + (1 - α) * genre_score
```

Why: matrix factorization captures hidden patterns while the genre model adds interpretable content information. Together they typically improve results over either alone.

## Data Preprocessing

To simplify the problem and improve model performance:

- Remove users with very few ratings
- Remove books with very few ratings

This avoids cold-start issues where there is not enough data to make good predictions.

## Evaluation

We evaluate models in two ways.

**Rating Prediction**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Measures how close predictions are to actual ratings.

**Recommendation Quality**
- HitRate@10 — did we recommend at least one good book?
- Precision@10
- Recall@10
- NDCG@10 (ranking quality)

A "liked" book is defined as a test rating of 4 or higher.

## Project Structure

```
BookRecSystem/
  README.md
  requirements.txt
  src/
    main.py               — full pipeline: load, train, evaluate, recommend
    data.py               — data loading, cleaning, and sparse filtering
    split.py              — per-user train/test split
    models.py             — PopularityModel, GenreModel, MFModel, HybridModel
    evaluation.py         — RMSE, MAE, HitRate, Precision, Recall, NDCG
    recommend.py          — top-k recommendation generation
    tuning.py             — SVD hyperparameter grid search
    prepare_data.py       — merges raw CSVs into books_with_genres.csv
    user_rating_counts.py — utility to compute per-user rating counts
  data/
    ratings.csv           — user-book ratings (required)
    books_with_genres.csv — book metadata with genre tags (required)
    books.csv             — raw book metadata
    book_tags.csv         — raw book-to-tag mappings
    tags.csv              — raw tag definitions
    user_rating_counts.csv — precomputed user rating counts
```

## Key Ideas

- Collaborative filtering learns from patterns in user behavior across the whole dataset
- Content-based filtering uses item features like genres
- Hybrid systems combine both and typically outperform either approach alone

## Limitations

- Cold-start problem: new users and books are filtered out rather than handled directly
- No use of review text
- Genre labels are coarse and can be noisy
- Assumes past ratings reflect future preferences
