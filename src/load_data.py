from pathlib import Path

import pandas as pd


def load_input(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {input_path}. "
            "Mount or copy test.csv into the input directory."
        )

    return pd.read_csv(input_path)
