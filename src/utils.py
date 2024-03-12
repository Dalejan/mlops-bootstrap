from joblib import dump
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def update_model(model: Pipeline) -> None:
    dump(model, "model/model.pkl")


def save_metrics_report(
    train_score: float, test_score: float, validation_score: float, model: Pipeline
) -> None:
    with open("report.txt", "w") as report_file:
        report_file.write("# Model Pipeline Description")

        for key, value in model.named_steps.items():
            report_file.write(f"### {key}:{value.__repr__()}" + "\n")

        report_file.write(f"## Train Score: {train_score} \n")
        report_file.write(f"## Test Score: {test_score} \n")
        report_file.write(f"## Validation Score: {validation_score} \n")


def get_test_performance(y_real: pd.Series, y_pred: pd.Series) -> None:
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=y_real, y=y_pred, ax=ax)
    ax.set_xlabel("Y Pred")
    ax.set_ylabel("Y Real")
    ax.set_title("Model prediction R2")
    fig.savefig("r2_prediction.png")
