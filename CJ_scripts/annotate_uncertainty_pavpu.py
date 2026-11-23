"""Add Table 5 PAvPU labels to the existing four-row EMSR466 comparison PNG.

Label positions match this figure's layout; no TIFFs or metrics are recomputed.
"""

import argparse
import csv
from pathlib import Path

from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PROJECT_ROOT / "artifacts/results/figures"
EVENT_ID = "EMSR466_AOI01_DEL_PRODUCT"
MODELS = ("EDL", "Deep Ensemble", "MC Dropout")
CRITERIA = ("Aleatoric + Epistemic", "Aleatoric", "Epistemic")
PANELS = [(0, 1, "EDL", "DST")] + [
    (row, col, model, criterion)
    for row, criterion in enumerate(CRITERIA, start=1)
    for col, model in enumerate(MODELS)
]

# Fractions of the original PNG dimensions, measured in its existing whitespace.
COLUMN_CENTERS = (0.170, 0.460, 0.750)
LABEL_TOPS = (0.224, 0.479, 0.734, 1.005)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path,
        default=FIGURE_ROOT / "uncertainty_compare" / f"{EVENT_ID}_uncertainty_compare.png",
        help="Source EMSR466 PNG with the existing four-row layout.",
    )
    parser.add_argument(
        "--table", type=Path,
        default=FIGURE_ROOT / "pavpu_tables/test_per_image_q90/table5_pavpu_event.csv",
        help="Table 5 CSV containing this event's PAvPU values.",
    )
    parser.add_argument("--output", type=Path, help="Destination PNG; defaults to *_pavpu_q90.png.")
    args = parser.parse_args()
    args.output = args.output or args.image.with_name(f"{args.image.stem}_pavpu_q90.png")
    if args.output.resolve() in {args.image.resolve(), args.table.resolve()}:
        parser.error("--output must differ from both input files.")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must have a .png extension.")
    return args


def read_scores(table_path):
    with table_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Model", "filename", "PAvPU (DST)"}
        required.update(f"PAvPU ({criterion})" for criterion in CRITERIA)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        rows = list(reader)

    if not rows or any(row["filename"] != EVENT_ID for row in rows):
        raise ValueError(f"Table 5 must contain only event {EVENT_ID}.")
    model_rows = {}
    for model in MODELS:
        matches = [row for row in rows if row["Model"] == model]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one Table 5 row for {model}.")
        model_rows[model] = matches[0]

    scores = {}
    for _, _, model, criterion in PANELS:
        value = float(model_rows[model][f"PAvPU ({criterion})"])
        if not 0 <= value <= 1:
            raise ValueError(f"Invalid PAvPU for {model} / {criterion}: {value}")
        scores[model, criterion] = value
    return scores


def main():
    args = parse_args()
    scores = read_scores(args.table)
    print(f"Image: {args.image}")
    print(f"Table: {args.table}")

    with Image.open(args.image) as source:
        width, height = source.size
        if abs(width / height - 4513 / 4155) > 0.005:
            raise ValueError("Expected the original EMSR466 four-row layout (4513:4155 aspect ratio).")
        font_size = max(1, round(width * 0.012))
        canvas = Image.new("RGBA", (width, height + 3 * font_size), "white")
        canvas.paste(source.convert("RGBA"), (0, 0))
        dpi = source.info.get("dpi", (300, 300))

    font_path = Path(get_data_path()) / "fonts/ttf/DejaVuSerif.ttf"
    font = ImageFont.truetype(str(font_path), font_size)
    draw = ImageDraw.Draw(canvas)
    for row, col, model, criterion in PANELS:
        label = f"PAvPU = {scores[model, criterion]:.4f}"
        position = (round(width * COLUMN_CENTERS[col]), round(height * LABEL_TOPS[row]))
        draw.text(position, label, font=font, fill="black", anchor="mt")
        print(f"  {model} / {criterion}: {label}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, dpi=dpi)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
