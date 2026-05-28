import numpy as np
import pandas as pd

from src.hash_model import load_model, predict_scores


def score_dataset(df: pd.DataFrame, model_path) -> tuple[np.ndarray, np.ndarray, dict]:
    model = load_model(model_path)
    scores = predict_scores(df, model)
    threshold = float(model.get("threshold", 0.5))
    predictions = (scores >= threshold).astype(int)
    return predictions, scores, model
