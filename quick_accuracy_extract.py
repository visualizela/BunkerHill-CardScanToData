from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ocr import OCR, EngineType
from src.region_processing import load_image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}


@dataclass(frozen=True)
class Args:
    run_dir: Path | None
    max_images_per_field: int
    fields: list[str]
    engine: str
    classifier: Path | None
    mode: str


def _parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=(
            "Quick accuracy extraction: OCR sliced crops and export CSV + JSON.\n\n"
            "Typical workflow:\n"
            "1) Run the UI (python -m src.main) and Save/Slice (Enter twice)\n"
            "2) Run this script to OCR a small sample of the slices\n"
        ),
        epilog=(
            "OpenAI (--engine openai): set OPENAI_API_KEY. Optional env: OPENAI_OCR_MODEL (default gpt-4o), "
            "OPENAI_VISION_DETAIL (default low)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help=(
            "Path to a specific sliced run folder (e.g. data/sliced_cards/02-09-2026). "
            "If omitted, the newest folder under data/sliced_cards is used."
        ),
    )
    parser.add_argument(
        "--max_images_per_field",
        type=int,
        default=10,
        help="Max images to OCR per field folder (keeps the run quick).",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        default=[],
        help=(
            "Optional allowlist of field folder names (box names). "
            "If provided, only these folders are OCR'd."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["easyocr", "paddleocr", "tesseract", "cloudvision", "trocr", "openai"],
        default="easyocr",
        help=(
            "OCR engine. Default is easyocr. trocr: local handwriting (Hugging Face). "
            "openai: GPT-4o vision (requires OPENAI_API_KEY; see epilog for optional env vars)."
        ),
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=None,
        help="Path to checkbox classifier pickle (e.g. static/checkbox_model_2.pkl). Used when a field has only subfolders (checkbox layout).",
    )
    parser.add_argument(
        "--mode",
        choices=["ocr", "checkbox", "auto"],
        default="auto",
        help="ocr: only OCR flat image folders. checkbox: only run classifier on subfolder layouts. auto: choose from folder layout (default).",
    )
    ns = parser.parse_args()
    return Args(
        run_dir=ns.run_dir,
        max_images_per_field=ns.max_images_per_field,
        fields=list(ns.fields or []),
        engine=ns.engine,
        classifier=ns.classifier,
        mode=ns.mode,
    )


def _pick_latest_run_dir(sliced_cards_root: Path) -> Path:
    run_dirs = [p for p in sliced_cards_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No sliced run folders found at: {sliced_cards_root}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _set_engine(ocr: OCR, engine: str) -> None:
    engine_map = {
        "easyocr": EngineType.EASYOCR,
        "paddleocr": EngineType.PADDLEOCR,
        "tesseract": EngineType.TESSERACT,
        "cloudvision": EngineType.CLOUDVISION,
        "trocr": EngineType.TROCR,
        "openai": EngineType.OPENAI_VISION,
    }
    if engine not in engine_map:
        raise ValueError(f"Unsupported engine: {engine}")
    ocr.set_engine(engine_map[engine])


def _is_checkbox_layout(field_dir: Path) -> bool:
    """True if field folder contains only subdirectories (no direct image files)."""
    children = list(field_dir.iterdir())
    if not children:
        return False
    direct_images = [p for p in children if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    subdirs = [p for p in children if p.is_dir()]
    return len(subdirs) > 0 and len(direct_images) == 0


def _process_field_checkbox(
    field_dir: Path,
    classifier_path: Path,
    classifier_cache: dict[str, object],
    max_per_field: int,
) -> list[dict[str, str]]:
    """Run checkbox classifier on option subfolders; return records with parsed_value = ';'.join(option names)."""
    cache_key = str(classifier_path.resolve())
    if cache_key not in classifier_cache:
        with open(classifier_path, "rb") as f:
            classifier_cache[cache_key] = pickle.load(f)
    classifier = classifier_cache[cache_key]

    subdirs = sorted([p for p in field_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    # Collect (card_image_name, option_name, crop_path) from all option subfolders
    all_crops: list[tuple[str, str, Path]] = []
    for opt_dir in subdirs:
        for p in sorted(opt_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                all_crops.append((p.name, opt_dir.name, p))

    # Unique card names in deterministic order; limit to first max_per_field
    unique_cards: list[str] = []
    seen: set[str] = set()
    for card_name, _opt, _path in all_crops:
        if card_name not in seen:
            seen.add(card_name)
            unique_cards.append(card_name)
    allowed_cards = set(unique_cards[:max_per_field])

    # For each allowed card, collect all (option, path) and run predict
    card_to_options: dict[str, list[tuple[str, Path]]] = {}
    for card_name, opt_name, path in all_crops:
        if card_name not in allowed_cards:
            continue
        if card_name not in card_to_options:
            card_to_options[card_name] = []
        card_to_options[card_name].append((opt_name, path))

    records: list[dict[str, str]] = []
    for card_name, options in sorted(card_to_options.items()):
        predicted_options: list[str] = []
        first_crop_path: str = ""
        for opt_name, path in options:
            if not first_crop_path:
                first_crop_path = str(path)
            try:
                feat = load_image(str(path))
                pred = classifier.predict([feat])[0]
                if pred == 1:
                    predicted_options.append(opt_name)
            except Exception as e:
                predicted_options.append(f"__ERROR__:{opt_name}:{e}")
        records.append(
            {
                "card_image": card_name,
                "field": field_dir.name,
                "ocr_text": "",
                "parsed_value": ";".join(predicted_options) if predicted_options else "",
                "crop_path": first_crop_path,
            }
        )
    return records


def main() -> None:
    # Avoid Windows charmap errors when EasyOCR (or other engines) print Unicode (e.g. progress bar).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    sliced_cards_root = project_root / "data" / "sliced_cards"

    run_dir = args.run_dir or _pick_latest_run_dir(sliced_cards_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    field_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
    if args.fields:
        allow = {f.strip() for f in args.fields if f.strip()}
        field_dirs = [p for p in field_dirs if p.name in allow]

    if not field_dirs:
        raise FileNotFoundError(
            f"No field folders found in {run_dir}. Expected subfolders like 'occupancy', 'business units', etc."
        )

    default_classifier = project_root / "static" / "checkbox_model_2.pkl"
    classifier_path = args.classifier if args.classifier is not None else default_classifier
    classifier_cache: dict[str, object] = {}

    ocr = OCR()
    _set_engine(ocr, args.engine)
    if args.engine == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set. Set the environment variable and retry "
            "(e.g. PowerShell: $env:OPENAI_API_KEY='sk-...').",
            file=sys.stderr,
        )
        raise SystemExit(1)

    records: list[dict[str, str]] = []

    for field_dir in sorted(field_dirs, key=lambda p: p.name.lower()):
        is_checkbox_layout = _is_checkbox_layout(field_dir)
        use_checkbox = (
            (args.mode == "checkbox" or args.mode == "auto")
            and is_checkbox_layout
        )
        if args.mode == "checkbox" and not is_checkbox_layout:
            continue  # Skip non-checkbox fields when mode is checkbox-only
        if args.mode == "ocr" and is_checkbox_layout:
            continue  # Skip checkbox-layout fields when mode is OCR-only

        if use_checkbox:
            if not classifier_path.exists():
                raise FileNotFoundError(
                    f"Checkbox classifier not found: {classifier_path}. Use --classifier or add static/checkbox_model_2.pkl"
                )
            records.extend(
                _process_field_checkbox(
                    field_dir,
                    classifier_path,
                    classifier_cache,
                    max(args.max_images_per_field, 0),
                )
            )
            continue

        # OCR path: direct image files in field folder
        images = [
            p
            for p in sorted(field_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ][: max(args.max_images_per_field, 0)]

        for img_path in images:
            try:
                text = ocr.run(img_path, field_context=field_dir.name).strip()
            except Exception as e:
                text = f"__ERROR__: {e}"

            records.append(
                {
                    "card_image": img_path.name,
                    "field": field_dir.name,
                    "ocr_text": text,
                    "parsed_value": text,
                    "crop_path": str(img_path),
                }
            )

    df = pd.DataFrame.from_records(records)

    out_dir = project_root / "data"
    out_csv = out_dir / f"accuracy_quick_{run_dir.name}.csv"
    out_json = out_dir / f"accuracy_quick_{run_dir.name}.json"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as f:
        df.to_json(f, orient="records", indent=2, force_ascii=False)

    print(f"Using sliced run: {run_dir}")
    print(f"Wrote CSV:  {out_csv}")
    print(f"Wrote JSON: {out_json}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

