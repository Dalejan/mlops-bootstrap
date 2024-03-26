from joblib import load
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame


def get_model() -> Pipeline:
    model = load("model/model.pkl")
    return model


def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    transition_dict = {key: [value] for key, value in class_model.model_dump().items()}
    data_frame = DataFrame(transition_dict)
    return data_frame
