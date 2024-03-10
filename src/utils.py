from joblib import dump
from sklearn.pipeline import Pipeline


def update_model(model: Pipeline) -> None:
    dump(model, "model/model.pkl")
