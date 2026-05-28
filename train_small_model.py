import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CAT_FEATURES, NUMERIC_FEATURES, TARGET_COL
from src.hash_model import build_design_matrix, predict_scores, save_model
from src.preprocess import preprocess_dataset


DEFAULT_TRAIN_PATH = Path("data/train.csv")


def iter_chunks(path: Path, chunksize: int):
    return pd.read_csv(path, chunksize=chunksize)


def collect_numeric_stats(path: Path, chunksize: int) -> tuple[dict, dict, int, int]:
    sums = np.zeros(len(NUMERIC_FEATURES), dtype=np.float64)
    sums_sq = np.zeros(len(NUMERIC_FEATURES), dtype=np.float64)
    counts = np.zeros(len(NUMERIC_FEATURES), dtype=np.float64)
    total_rows = 0
    positive_rows = 0

    for chunk in iter_chunks(path, chunksize):
        prepared = preprocess_dataset(chunk)
        numeric = prepared[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        finite = np.isfinite(values)

        sums += np.where(finite, values, 0.0).sum(axis=0)
        sums_sq += np.where(finite, values * values, 0.0).sum(axis=0)
        counts += finite.sum(axis=0)

        target = chunk[TARGET_COL].to_numpy(dtype=np.int8)
        total_rows += len(target)
        positive_rows += int(target.sum())

    means = sums / np.maximum(counts, 1)
    variances = sums_sq / np.maximum(counts, 1) - means * means
    stds = np.sqrt(np.maximum(variances, 1e-8))

    return (
        dict(zip(NUMERIC_FEATURES, means.tolist())),
        dict(zip(NUMERIC_FEATURES, stds.tolist())),
        total_rows,
        positive_rows,
    )


def model_shell(numeric_means: dict, numeric_stds: dict, hash_size: int) -> dict:
    feature_count = len(NUMERIC_FEATURES) + hash_size
    return {
        "model_type": "hashed_logistic_regression",
        "version": 1,
        "hash_size": hash_size,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CAT_FEATURES,
        "numeric_means": numeric_means,
        "numeric_stds": numeric_stds,
        "weights": [0.0] * feature_count,
        "bias": 0.0,
        "threshold": 0.5,
    }


def fit_model(
    train_path: Path,
    model: dict,
    chunksize: int,
    epochs: int,
    learning_rate: float,
    positive_weight: float,
    l2: float,
) -> dict:
    weights = np.zeros(len(model["weights"]), dtype=np.float32)
    bias = np.float32(0.0)

    for epoch in range(epochs):
        row_offset = 0
        for chunk_id, chunk in enumerate(iter_chunks(train_path, chunksize), start=1):
            row_ids = np.arange(row_offset, row_offset + len(chunk))
            row_offset += len(chunk)
            train_mask = row_ids % 5 != 0
            if not train_mask.any():
                continue

            prepared = preprocess_dataset(chunk.loc[train_mask].copy())
            design_matrix = build_design_matrix(prepared, model)
            target = chunk.loc[train_mask, TARGET_COL].to_numpy(dtype=np.float32)

            logits = design_matrix @ weights + bias
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            sample_weights = np.where(target > 0.5, positive_weight, 1.0).astype(np.float32)
            errors = (probabilities - target) * sample_weights

            grad_w = design_matrix.T @ errors / len(target) + l2 * weights
            grad_b = errors.mean()

            step = learning_rate / np.sqrt(epoch + 1)
            weights -= step * grad_w.astype(np.float32)
            bias -= np.float32(step * grad_b)

            if chunk_id % 5 == 0:
                print(
                    f"epoch={epoch + 1}/{epochs} chunk={chunk_id} "
                    f"positive_rate={target.mean():.5f}"
                )

    model["weights"] = np.round(weights.astype(float), 8).tolist()
    model["bias"] = float(np.round(bias, 8))
    return model


def collect_validation_scores(path: Path, model: dict, chunksize: int) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    targets = []
    row_offset = 0

    for chunk in iter_chunks(path, chunksize):
        row_ids = np.arange(row_offset, row_offset + len(chunk))
        row_offset += len(chunk)
        valid_mask = row_ids % 5 == 0
        if not valid_mask.any():
            continue

        prepared = preprocess_dataset(chunk.loc[valid_mask].copy())
        scores.append(predict_scores(prepared, model))
        targets.append(chunk.loc[valid_mask, TARGET_COL].to_numpy(dtype=np.int8))

    return np.concatenate(scores), np.concatenate(targets)


def best_f1_threshold(scores: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_target = target[order]

    positives = max(int(sorted_target.sum()), 1)
    tp = np.cumsum(sorted_target)
    fp = np.cumsum(1 - sorted_target)
    fn = positives - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    best_idx = int(np.argmax(f1))
    return float(sorted_scores[best_idx]), float(f1[best_idx])


def calculate_feature_importances(path: Path, model: dict, chunksize: int) -> dict:
    weights = np.abs(np.asarray(model["weights"], dtype=np.float32))
    numeric_weights = weights[: len(NUMERIC_FEATURES)]
    hash_weights = weights[len(NUMERIC_FEATURES) :]

    importances = {
        feature: float(weight)
        for feature, weight in zip(NUMERIC_FEATURES, numeric_weights)
    }

    row_offset = 0
    cat_sums = {feature: 0.0 for feature in CAT_FEATURES}
    cat_counts = {feature: 0 for feature in CAT_FEATURES}

    from src.hash_model import stable_hash

    for chunk in iter_chunks(path, chunksize):
        row_ids = np.arange(row_offset, row_offset + len(chunk))
        row_offset += len(chunk)
        valid_mask = row_ids % 5 == 0
        if not valid_mask.any():
            continue

        prepared = preprocess_dataset(chunk.loc[valid_mask].copy())
        for feature in CAT_FEATURES:
            values = prepared[feature].fillna("unknown").astype(str).to_numpy()
            hashed_indices = np.fromiter(
                (
                    stable_hash(f"{feature}={value}", model["hash_size"])
                    for value in values
                ),
                dtype=np.int32,
                count=len(values),
            )
            cat_sums[feature] += float(hash_weights[hashed_indices].sum())
            cat_counts[feature] += len(values)

    for feature in CAT_FEATURES:
        importances[feature] = cat_sums[feature] / max(cat_counts[feature], 1)

    total = sum(importances.values()) or 1.0
    return {
        feature: round(value / total, 8)
        for feature, value in sorted(importances.items(), key=lambda item: item[1], reverse=True)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train compact fraud model.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-model", type=Path, default=Path("models/fraud_hash_logreg.json"))
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--hash-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.0001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_path.exists():
        raise FileNotFoundError(
            f"Train file was not found: {args.train_path}. "
            "Pass --train-path with the path to train.csv if you need to retrain the model."
        )

    numeric_means, numeric_stds, total_rows, positive_rows = collect_numeric_stats(
        args.train_path,
        args.chunksize,
    )
    negative_rows = total_rows - positive_rows
    positive_weight = min(80.0, negative_rows / max(positive_rows, 1))

    print(json.dumps(
        {
            "rows": total_rows,
            "positive_rows": positive_rows,
            "positive_weight": positive_weight,
        },
        indent=2,
    ))

    model = model_shell(numeric_means, numeric_stds, args.hash_size)
    model = fit_model(
        train_path=args.train_path,
        model=model,
        chunksize=args.chunksize,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        positive_weight=positive_weight,
        l2=args.l2,
    )

    validation_scores, validation_target = collect_validation_scores(
        args.train_path,
        model,
        args.chunksize,
    )
    threshold, f1 = best_f1_threshold(validation_scores, validation_target)

    model["threshold"] = round(threshold, 8)
    model["metrics"] = {
        "validation_f1": round(f1, 8),
        "validation_rows": int(len(validation_target)),
        "validation_positive_rows": int(validation_target.sum()),
    }
    model["feature_importances"] = calculate_feature_importances(
        args.train_path,
        model,
        args.chunksize,
    )

    save_model(model, args.output_model)
    print(f"Saved model to {args.output_model}")
    print(json.dumps(model["metrics"], indent=2))


if __name__ == "__main__":
    main()
