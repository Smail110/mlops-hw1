import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_top_feature_importances(model: dict, output_path: str | Path, top_n: int = 5) -> None:
    importances = model.get("feature_importances", {})
    top_items = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:top_n]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(dict(top_items), file, ensure_ascii=False, indent=2)


def save_prediction_density(scores: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 900, 520
    margin_left, margin_right = 70, 30
    margin_top, margin_bottom = 35, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    hist, _ = np.histogram(scores, bins=40, range=(0.0, 1.0), density=True)
    max_density = max(float(hist.max()), 1e-9)

    axis_color = (45, 45, 45)
    bar_color = (42, 126, 170)
    grid_color = (225, 225, 225)

    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top

    for step in range(6):
        y = y0 - int(plot_height * step / 5)
        draw.line((x0, y, x1, y), fill=grid_color, width=1)

    draw.line((x0, y0, x1, y0), fill=axis_color, width=2)
    draw.line((x0, y0, x0, y1), fill=axis_color, width=2)

    bar_width = plot_width / len(hist)
    for idx, density in enumerate(hist):
        left = x0 + idx * bar_width + 1
        right = x0 + (idx + 1) * bar_width - 1
        top = y0 - (float(density) / max_density) * plot_height
        draw.rectangle((left, top, right, y0), fill=bar_color)

    draw.text((margin_left, 12), "Prediction score density", fill=axis_color)
    draw.text((x0, y0 + 25), "0.0", fill=axis_color)
    draw.text((x1 - 24, y0 + 25), "1.0", fill=axis_color)
    draw.text((width // 2 - 35, y0 + 42), "score", fill=axis_color)
    draw.text((10, margin_top), f"max density: {max_density:.2f}", fill=axis_color)

    image.save(output_path)
