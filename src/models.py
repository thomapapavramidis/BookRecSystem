from collections import defaultdict

import numpy as np

try:
    from .data import parse_genres
except ImportError:  # Allows running python src/main.py
    from data import parse_genres

# Make sure we only output valid ratings between 1 and 5 inclusive
def clamp_rating(value):
    return float(np.clip(value, 1.0, 5.0))

# Extracts value as string
def _as_id(value):
    return str(value).strip()

# Scores books based on Bayesian average, provides baseline performance with no user personalization
class PopularityModel:
    def __init__(self, smoothing=10):
        self.smoothing = smoothing

    def fit(self, train_df, books_df=None):
        train = train_df.copy()
        train["book_id"] = train["book_id"].map(_as_id)

        # Calculate Bayesian average to minimize impact of outliers
        self.global_mean = float(train["rating"].mean()) if len(train) else 3.0
        stats = train.groupby("book_id")["rating"].agg(["mean", "count"])
        m = self.smoothing
        stats["score"] = (stats["count"] / (stats["count"] + m)) * stats["mean"]
        stats["score"] += (m / (stats["count"] + m)) * self.global_mean

        # Save score as dictionary
        self.book_scores = stats["score"].to_dict()

        # Save set of all books trained on
        self.all_book_ids = set(train["book_id"])
        if books_df is not None:
            self.all_book_ids.update(books_df["book_id"].map(_as_id))
        self.all_book_ids = sorted(self.all_book_ids)
        return self

    # Return score for given user and book (no personalization in this model, all users share same scores)
    def predict(self, user_id, book_id):
        return clamp_rating(self.book_scores.get(_as_id(book_id), self.global_mean))

    # Get predictions for a given user and a list of books
    def score_items(self, user_id, book_ids):
        return np.array([self.predict(user_id, book_id) for book_id in book_ids])

# Makes personalized book enjoyment predictions by calculating how much a user enjoys the genres in a given book
class GenreModel:
    def fit(self, train_df, books_df=None):
        train = train_df.copy()
        train["user_id"] = train["user_id"].map(_as_id)
        train["book_id"] = train["book_id"].map(_as_id)
        self.global_mean = float(train["rating"].mean()) if len(train) else 3.0

        # Calculate average user ratings and book genres
        self.user_means = train.groupby("user_id")["rating"].mean().to_dict()
        self.book_genres = self._build_book_genres(books_df)
        self.has_genres = any(len(genres) > 0 for genres in self.book_genres.values())

        # Save set of all books trained on
        self.all_book_ids = set(train["book_id"])
        if books_df is not None:
            self.all_book_ids.update(books_df["book_id"].map(_as_id))
        self.all_book_ids = sorted(self.all_book_ids)

        # Create nested dictionaries to hold values to be computed
        sums = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(lambda: defaultdict(int))

        # Calculate how highly users rate and how many times users rate books of each genre
        for row in train.itertuples(index=False):
            # Get user, book, and genres for current row
            user = _as_id(row.user_id)
            book = _as_id(row.book_id)
            genres = self.book_genres.get(book, [])
            if not genres:
                continue

            # Calculate difference from mean rating and increment count for that genre 
            deviation = float(row.rating) - self.user_means[user]
            for genre in genres:
                sums[user][genre] += deviation
                counts[user][genre] += 1

        # Calculate scores using average rating above mean
        self.user_genre_scores = {}
        for user, genre_sums in sums.items():
            self.user_genre_scores[user] = {
                genre: genre_sum / counts[user][genre]
                for genre, genre_sum in genre_sums.items()
            }

        return self

    # Returns dictionary with book_ids as keys and genre lists as values
    def _build_book_genres(self, books_df):
        if books_df is None:
            return {}

        # Pull list of genres from each book and store in dictionary
        book_genres = {}
        for row in books_df.itertuples(index=False):
            book_id = _as_id(row.book_id)
            if hasattr(row, "genres_list"):
                genres = list(row.genres_list) if isinstance(row.genres_list, list) else []
            elif hasattr(row, "genres"):
                genres = parse_genres(row.genres)
            else:
                genres = []
            book_genres[book_id] = genres
        return book_genres

    # Return score for a given user and book by seeing how much the user enjoys the genres of the book
    def predict(self, user_id, book_id):
        user = _as_id(user_id)
        book = _as_id(book_id)
        base = self.user_means.get(user, self.global_mean)

        # Select user's average rating above mean for every genre the book has
        user_scores = self.user_genre_scores.get(user, {})
        genres = self.book_genres.get(book, [])
        matching_scores = [user_scores[genre] for genre in genres if genre in user_scores]

        # Average the average rating above mean for selected genres and add to user's mean rating
        if matching_scores:
            base += float(np.mean(matching_scores))
        return clamp_rating(base)

    # Get predictions for a given user and a list of books
    def score_items(self, user_id, book_ids):
        return np.array([self.predict(user_id, book_id) for book_id in book_ids])

# Makes personalized book enjoyment recommendations using user-book rating matrix factorization
class MFModel:
    def __init__(
        self,
        n_factors=50,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
        use_surprise=True,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.use_surprise = use_surprise
        self.backend = "unfit"

    def fit(self, train_df, books_df=None):
        train = train_df.copy()
        train["user_id"] = train["user_id"].map(_as_id)
        train["book_id"] = train["book_id"].map(_as_id)
        self.global_mean = float(train["rating"].mean()) if len(train) else 3.0

        # Save set of all books trained on
        self.all_book_ids = set(train["book_id"])
        if books_df is not None:
            self.all_book_ids.update(books_df["book_id"].map(_as_id))
        self.all_book_ids = sorted(self.all_book_ids)

        # If available, use SVD from Surprise library
        if self.use_surprise:
            try:
                from surprise import Dataset, Reader, SVD

                # Execute SVD on user-book ratings
                reader = Reader(rating_scale=(1, 5))
                data = Dataset.load_from_df(train[["user_id", "book_id", "rating"]], reader)
                trainset = data.build_full_trainset()
                self.model = SVD(
                    n_factors=self.n_factors,
                    n_epochs=self.n_epochs,
                    lr_all=self.lr_all,
                    reg_all=self.reg_all,
                    random_state=self.random_state,
                )
                self.model.fit(trainset)
                self.backend = "surprise"
                return self
            except Exception as exc:
                self.surprise_error = str(exc)

        # If Surprise not available, use SGD to implement matrix factorization
        self._fit_sgd(train)
        self.backend = "sgd"
        return self

    # Use the dot product of user and book vectors to make predictions
    # Use least squares to minimize the difference between predictions and true ratings
    def _fit_sgd(self, train):
        rng = np.random.default_rng(self.random_state)

        # Assign each user and book an index
        users = sorted(train["user_id"].unique())
        books = sorted(train["book_id"].unique())
        self.user_index = {user: idx for idx, user in enumerate(users)}
        self.book_index = {book: idx for idx, book in enumerate(books)}

        # Initialize biases and weights
        self.user_bias = np.zeros(len(users))
        self.book_bias = np.zeros(len(books))
        self.user_factors = rng.normal(0, 0.1, size=(len(users), self.n_factors))
        self.book_factors = rng.normal(0, 0.1, size=(len(books), self.n_factors))

        # Create (user, book, rating) samples using assigned indices
        samples = [
            (self.user_index[row.user_id], self.book_index[row.book_id], float(row.rating))
            for row in train.itertuples(index=False)
        ]

        for _ in range(self.n_epochs):
            rng.shuffle(samples)
            for u_idx, b_idx, rating in samples:
                # Make prediction and calculate loss
                pred = self._predict_sgd_indices(u_idx, b_idx)
                err = rating - pred

                # Save current relevant weights
                user_vec = self.user_factors[u_idx].copy()
                book_vec = self.book_factors[b_idx].copy()

                # Update relevant weights
                self.user_bias[u_idx] += self.lr_all * (err - self.reg_all * self.user_bias[u_idx])
                self.book_bias[b_idx] += self.lr_all * (err - self.reg_all * self.book_bias[b_idx])
                self.user_factors[u_idx] += self.lr_all * (err * book_vec - self.reg_all * user_vec)
                self.book_factors[b_idx] += self.lr_all * (err * user_vec - self.reg_all * book_vec)

    # Predict the rating a user will give a book
    def _predict_sgd_indices(self, u_idx, b_idx):
        # Adds dot product of user and book in latent feature space to global mean and biases
        return (
            self.global_mean
            + self.user_bias[u_idx]
            + self.book_bias[b_idx]
            + float(np.dot(self.user_factors[u_idx], self.book_factors[b_idx]))
        )

    # Return score for a given user and book
    def predict(self, user_id, book_id):
        user = _as_id(user_id)
        book = _as_id(book_id)

        # If we fit with Surprise, predict using Surprise model
        if self.backend == "surprise":
            return clamp_rating(self.model.predict(user, book).est)

        # If we fit with SGD, calculate the dot product of the vectors in latent feature space (along with global mean and biases)
        pred = self.global_mean
        u_idx = self.user_index.get(user)
        b_idx = self.book_index.get(book)
        if u_idx is not None:
            pred += self.user_bias[u_idx]
        if b_idx is not None:
            pred += self.book_bias[b_idx]
        if u_idx is not None and b_idx is not None:
            pred += float(np.dot(self.user_factors[u_idx], self.book_factors[b_idx]))
        return clamp_rating(pred)

   # Get predictions for a given user and a list of books
    def score_items(self, user_id, book_ids):
        return np.array([self.predict(user_id, book_id) for book_id in book_ids])

# Balances predictions of genre and matrix factorization models for greater accuracy
class HybridModel:
    def __init__(self, mf_model, genre_model, alpha=0.8):
        self.mf_model = mf_model
        self.genre_model = genre_model
        self.alpha = alpha
        self.all_book_ids = sorted(set(mf_model.all_book_ids) | set(genre_model.all_book_ids))

    # Return score for a given user and book
    def predict(self, user_id, book_id):
        # Predict with each model
        mf_score = self.mf_model.predict(user_id, book_id)
        genre_score = self.genre_model.predict(user_id, book_id)

        # Combine scores according to alpha parameter
        return clamp_rating(self.alpha * mf_score + (1 - self.alpha) * genre_score)

    # Get predictions for a given user and a list of books
    def score_items(self, user_id, book_ids):
        return np.array([self.predict(user_id, book_id) for book_id in book_ids])
