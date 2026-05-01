# Overview

This project builds a book recommendation system using machine learning. The goal is to predict which books a user would like based on their past ratings.

Recommender systems are widely used in platforms like Amazon and Netflix to suggest items based on user behavior. We implement and compare several approaches, from simple baselines to more advanced models.

## Problem Definition

Given:

- A dataset of user-book ratings
- Optional book metadata (genres, author, etc.)

We want to:

- Predict how much a user would like a book
- Recommend the top-N books they are most likely to enjoy

## Dataset Format

Ratings dataset (required):

| user_id | book_id | rating |
| --- | --- | --- |
| 1 | 101 | 5 |
| 1 | 205 | 3 |

Ratings are between 1 and 5.

Book metadata (optional):

| book_id | title | author | genres |
| --- | --- | --- | --- |
| 101 | ... | ... | fantasy, adventure |

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

## How to Run

```bash
pip install -r requirements.txt

python src/main.py \
  --ratings_path data/ratings.csv \
  --books_path data/books.csv
```

The script will load and preprocess the data, train all models, evaluate performance, and print results along with example recommendations for a few users.

## Key Ideas

- Collaborative filtering learns from patterns in user behavior across the whole dataset
- Content-based filtering uses item features like genres
- Hybrid systems combine both and typically outperform either approach alone

## Limitations

- Cold-start problem: new users and books are filtered out rather than handled directly
- No use of review text
- Genre labels are coarse and can be noisy
- Assumes past ratings reflect future preferences
