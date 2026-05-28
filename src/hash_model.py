import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def stable_hash(value: str, modulo: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % modulo


def load_model(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_model(model: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def build_design_matrix(df: pd.DataFrame, model: dict) -> np.ndarray:
    numeric_features = model["numeric_features"]
    categorical_features = model["categorical_features"]
    hash_size = int(model["hash_size"])

    numeric_part = np.empty((len(df), len(numeric_features)), dtype=np.float32)
    for idx, feature in enumerate(numeric_features):
        mean = float(model["numeric_means"].get(feature, 0.0))
        std = float(model["numeric_stds"].get(feature, 1.0)) or 1.0
        values = pd.to_numeric(df[feature], errors="coerce").fillna(mean).to_numpy()
        numeric_part[:, idx] = ((values - mean) / std).astype(np.float32)

    hashed_part = np.zeros((len(df), hash_size), dtype=np.float32)
    for feature in categorical_features:
        values = df[feature].fillna("unknown").astype(str).to_numpy()
        unique_values = pd.unique(values)
        index_map = {
            value: stable_hash(f"{feature}={value}", hash_size)
            for value in unique_values
        }
        hashed_indices = np.fromiter(
            (index_map[value] for value in values),
            dtype=np.int32,
            count=len(values),
        )
        hashed_part[np.arange(len(df)), hashed_indices] += 1.0

    return np.hstack([numeric_part, hashed_part])


def predict_scores(df: pd.DataFrame, model: dict) -> np.ndarray:
    design_matrix = build_design_matrix(df, model)
    weights = np.asarray(model["weights"], dtype=np.float32)
    bias = float(model["bias"])
    return sigmoid(design_matrix @ weights + bias)
