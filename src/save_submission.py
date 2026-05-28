from pathlib import Path

import numpy as np
import pandas as pd

from src.config import INDEX_COL, PREDICTION_COL


def save_submission(predictions: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {
            INDEX_COL: np.arange(len(predictions)),
            PREDICTION_COL: predictions.astype(int),
        }
    )
    submission.to_csv(output_path, index=False)
