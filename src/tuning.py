# Method for tuning matrix factorization model hyperparameters

from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV

# Different hypeparameter settings
_PARAM_GRID = {
    "n_factors": [50, 100, 200],
    "n_epochs": [20, 40],
    "reg_all": [0.02, 0.05, 0.1],
    "lr_all": [0.005],
}

# Returns best root mean squared error parameters and score
def tune_svd(train_df, cv=3, n_jobs=4):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["user_id", "book_id", "rating"]], reader)
    gs = GridSearchCV(SVD, _PARAM_GRID, measures=["rmse"], cv=cv, n_jobs=n_jobs)
    gs.fit(data)
    return gs.best_params["rmse"], gs.best_score["rmse"]
