from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd

import sys
import logging

from utils import update_model

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)

logger.info("Loading data ...")
data = pd.read_csv("dataset/full_data.csv")

logger.info("Loading model ...")
model = Pipeline(
    [
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("core_model", GradientBoostingRegressor()),
    ]
)

logger.info("Dataset train test split ...")
X = data.drop(["worldwide_gross"], axis=1)
y = data["worldwide_gross"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42
)

logger.info("Setting hyperparams to tune ...")
param_tunning = {"core_model__n_estimators": range(20, 301, 20)}

grid_search = GridSearchCV(model, param_grid=param_tunning, scoring="r2", cv=5)

logger.info("Starting grid search ...")
grid_search.fit(X_train, y_train)

logger.info("Cross validation with best model")
final_result = cross_validate(
    grid_search.best_estimator_, X_train, y_train, return_train_score=True, cv=7
)

train_score = np.mean(final_result["train_score"])
test_score = np.mean(final_result["test_score"])
assert train_score > 0.7
assert test_score > 0.65

logger.info(f"Train score: {train_score}")
logger.info(f"Test score: {test_score}")

logger.info("Updating model ...")

update_model(grid_search.best_estimator_)
