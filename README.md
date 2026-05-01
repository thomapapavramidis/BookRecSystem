Overview

This project builds a book recommendation system using machine learning. The goal is to predict which books a user would like based on their past ratings.

Recommender systems are widely used in platforms like Amazon and Netflix to suggest items based on user behavior .

We implement and compare several approaches, from simple baselines to more advanced models.

Problem Definition

Given:

A dataset of user-book ratings
Optional book metadata (genres, author, etc.)

We want to:

Predict how much a user would like a book
Recommend the top-N books they are most likely to enjoy
Dataset Format
Ratings dataset (required)
user_id	book_id	rating
1	101	5
1	205	3
Ratings are between 1 and 5
Book metadata (optional)
book_id	title	author	genres
101	...	...	fantasy, adventure
Approach

We implement three main models + one hybrid model.

1. Popularity Baseline
Recommends the highest-rated books overall
Same recommendations for all users

Why:

Simple baseline
Checks if personalization actually helps
2. Genre-Based Model (Content-Based)
Builds a genre preference profile per user
Uses ratings (not just counts)

Example idea:

If a user rates fantasy books higher than their average → fantasy gets a high score

Prediction:

Recommend books with genres the user tends to rate highly

Why:

Simple and interpretable
Works even for new books (uses metadata)
3. Matrix Factorization (Collaborative Filtering)
Learns latent features for users and books
Predicts ratings using:
predicted_rating = user_vector ⋅ book_vector

This method captures hidden patterns in user preferences that are not explicitly labeled.

Collaborative filtering works by learning from user behavior and finding patterns across many users .

Why:

Strong performance
Standard approach in recommender systems
4. Hybrid Model

Combines both approaches:

final_score = α * matrix_factorization + (1 - α) * genre_score

Why:

Matrix factorization captures hidden patterns
Genre model adds interpretable content information
Typically improves results
Data Preprocessing

To simplify the problem and improve performance:

Remove users with very few ratings
Remove books with very few ratings

This avoids cold-start issues, where there is not enough data to make predictions.

Evaluation

We evaluate models in two ways:

1. Rating Prediction
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)

Measures how close predictions are to actual ratings.

2. Recommendation Quality

We evaluate how good the recommendations are:

HitRate@10 → Did we recommend at least one good book?
Precision@10
Recall@10
NDCG@10 (ranking quality)

We define a “liked” book as:

rating ≥ 4
Project Structure
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
How to Run
pip install -r requirements.txt

python src/main.py \
  --ratings_path data/ratings.csv \
  --books_path data/books.csv

The script will:

Load and preprocess data
Train all models
Evaluate performance
Print results and example recommendations
Key Ideas
Collaborative filtering: learn from user behavior patterns
Content-based filtering: use item features (genres, etc.)
Hybrid systems combine both for better performance

Recommender systems generally fall into these categories or combinations of them .

Limitations
Cold-start problem (new users/books)
No use of review text (simplification)
Genre labels are coarse
Assumes past ratings reflect future preferences