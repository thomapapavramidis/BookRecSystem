from collections import defaultdict

import numpy as np

try:
    from .data import parse_genres
except ImportError:  # Allows running python src/main.py
    from data import parse_genres


def clamp_rating(value):
    return float(np.clip(value, 1.0, 5.0))


def _as_id(value):
    return str(value).strip()


class PopularityModel:
    def __init__(self, smoothing=10):
        self.smoothing = smoothing

    def fit(self, train_df, books_df=None):
        train = train_df.copy()
        train["book_id"] = train["book_id"].map(_as_id)
        self.global_mean = float(train["rating"].mean()) if len(train) else 3.0

        stats = train.groupby("book_id")["rating"].agg(["mean", "count"])
        m = self.smoothing
        stats["score"] = (stats["count"] / (stats["count"] + m)) * stats["mean"]
        stats["score"] += (m / (stats["count"] + m)) * self.global_mean

        self.book_scores = stats["score"].to_dict()
        self.all_book_ids = set(train["book_id"])
        if books_df is not None:
            self.all_book_ids.update(books_df["book_id"].map(_as_id))
        self.all_book_ids = sorted(self.all_book_ids)
        return self

    def predict(self, user_id, book_id):
        return clamp_rating(self.book_scores.get(_as_id(book_id), self.global_mean))

    def score_items(self, user_id, book_ids):
        return np.array([self.predict(user_id, book_id) for book_id in book_ids])


class GenreModel:
    def fit(self, train_df, books_df=None):
        train = train_df.copy()
        train["user_id"] = train["user_id"].map(_as_id)
        train["book_id"] = train["book_id"].map(_as_id)
        self.global_mean = float(train["rating"].mean()) if len(train) else 3.0
        self.user_means = train.groupby("user_id")["rating"].mean().to_dict()

        self.book_genres = self._build_book_genres(books_df)
        self.has_genres = any(len(genres) > 0 for genres in self.book_genres.values())
        self.all_book_ids = set(train["book_id"])
        if books_df is not None:
            self.all_book_ids.update(books_df["book_id"].map(_as_id))
        self.all_book_ids = sorted(self.all_book_ids)

        sums = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(lambda: defaultdict(int))

        for row in train.itertuples(index=False):
            user = _as_id(row.user_id)
            book = _as_id(row.book_id)
            genres = self.book_genres.get(book, [])
            if not genres:
                continue

            deviation = float(row.rating) - self.user_means[user]
            for genre in genres:
                sums[user][genre] += deviation
                counts[user][genre] += 1

        self.user_genre_scores = {}
        for user, genre_sums in sums.items():
            self.user_genre_scores[user] = {
                genre: genre_sum / counts[user][genre]
                for genre, genre_sum in genre_sums.items()
            }

        return self

    def _build_book_genres(self, books_df):
        if books_df is None:
            return {}

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

    def predict(self, user_id, book_id):
        user = _as_id(user_id)
        book = _as_id(book_id)
        base = self.user_means.get(user, self.global_mean)

        user_scores = self.user_genre_scores.get(user, {})
        genres = self.book_genres.get(book, [])
        matching_scores = [user_scores[genre] for genre in genres if genre in user_scores]

        if matching_scores:
            base += float(np.mean(matching_scores))
        return clamp_rating(base)

    def score_items(self, user_id, book_ids):
        return np.array([self.predict(user_id, book_id) for book_id in book_ids])


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
        self.all_book_ids = set(train["book_id"])
        if books_df is not None:
            self.all_book_ids.update(books_df["book_id"].map(_as_id))
        self.all_book_ids = sorted(self.all_book_ids)

        if self.use_surprise:
            try:
                from surprise import Dataset, Reader, SVD

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

        self._fit_sgd(train)
        self.backend = "sgd"
        return self

    def _fit_sgd(self, train):
        rng = np.random.default_rng(self.random_state)
        users = sorted(train["user_id"].unique())
        books = sorted(train["book_id"].unique())
        self.user_index = {user: idx for idx, user in enumerate(users)}
        self.book_index = {book: idx for idx, book in enumerate(books)}

        self.user_bias = np.zeros(len(users))
        self.book_bias = np.zeros(len(books))
        self.user_factors = rng.normal(0, 0.1, size=(len(users), self.n_factors))
        self.book_factors = rng.normal(0, 0.1, size=(len(books), self.n_factors))

        samples = [
            (self.user_index[row.user_id], self.book_index[row.book_id], float(row.rating))
            for row in train.itertuples(index=False)
        ]

        for _ in range(self.n_epochs):
            rng.shuffle(samples)
            for u_idx, b_idx, rating in samples:
                pred = self._predict_sgd_indices(u_idx, b_idx)
                err = rating - pred

                user_vec = self.user_factors[u_idx].copy()
                book_vec = self.book_factors[b_idx].copy()

                self.user_bias[u_idx] += self.lr_all * (err - self.reg_all * self.user_bias[u_idx])
                self.book_bias[b_idx] += self.lr_all * (err - self.reg_all * self.book_bias[b_idx])
                self.user_factors[u_idx] += self.lr_all * (err * book_vec - self.reg_all * user_vec)
                self.book_factors[b_idx] += self.lr_all * (err * user_vec - self.reg_all * book_vec)

    def _predict_sgd_indices(self, u_idx, b_idx):
        return (
            self.global_mean
            + self.user_bias[u_idx]
            + self.book_bias[b_idx]
            + float(np.dot(self.user_factors[u_idx], self.book_factors[b_idx]))
        )

    def predict(self, user_id, book_id):
        user = _as_id(user_id)
        book = _as_id(book_id)

        if self.backend == "surprise":
            return clamp_rating(self.model.predict(user, book).est)

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

    def score_items(self, user_id, book_ids):
        return np.array([self.predict(user_id, book_id) for book_id in book_ids])


class HybridModel:
    def __init__(self, mf_model, genre_model, alpha=0.8):
        self.mf_model = mf_model
        self.genre_model = genre_model
        self.alpha = alpha
        self.all_book_ids = sorted(set(mf_model.all_book_ids) | set(genre_model.all_book_ids))

    def predict(self, user_id, book_id):
        mf_score = self.mf_model.predict(user_id, book_id)
        genre_score = self.genre_model.predict(user_id, book_id)
        return clamp_rating(self.alpha * mf_score + (1 - self.alpha) * genre_score)

    def score_items(self, user_id, book_ids):
        return np.array([self.predict(user_id, book_id) for book_id in book_ids])
