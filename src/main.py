import argparse
from pathlib import Path

from src.config import DEFAULT_INPUT_PATH, DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR
from src.load_data import load_input
from src.predict import score_dataset
from src.preprocess import preprocess_dataset
from src.save_submission import save_submission


LEGACY_OUTPUTS = [
    "prediction_scores.csv",
    "feature_importances.json",
    "prediction_density.png",
]


def run(input_path: Path, output_dir: Path, model_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in LEGACY_OUTPUTS:
        legacy_path = output_dir / filename
        if legacy_path.exists():
            legacy_path.unlink()

    raw_data = load_input(input_path)
    prepared_data = preprocess_dataset(raw_data)
    predictions, _, _ = score_dataset(prepared_data, model_path)

    save_submission(predictions, output_dir / "sample_submission.csv")

    print(f"Rows scored: {len(predictions)}")
    print(f"Submission saved to: {output_dir / 'sample_submission.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fraud model inference.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output_dir, args.model)
