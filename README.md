# BunkerHill-CardScanToData

Python tools for processing census cards: deskew/trim, interactive box selection, OCR, checkbox classification, and analysis.

## Documentation for day-to-day use

**Team members** should start with **[RUNBOOK_Team.md](RUNBOOK_Team.md)**. It explains how to install dependencies, run the card UI, find sliced outputs, optionally run the quick accuracy script, and (if needed) configure an **OpenAI API key** with copy-paste steps for Windows.

---

## Key features

1. **Card selection UI** — Draw and name regions on cards; export aligned crops and `boxes.json`. Entry point: `src/main.py`. Keyboard reference: `src/help_menu.txt`.
2. **Quick accuracy extraction** — `quick_accuracy_extract.py` samples cropped fields, runs OCR (default: **EasyOCR**, no API key), and writes CSV/JSON under `data/`. Optional **`--engine openai`** uses GPT-4o vision; see **RUNBOOK_Team.md** for obtaining and setting **`OPENAI_API_KEY`**.
3. **Notebooks** — Image prep, checkbox training, header/body text and checkboxes, analysis. Numbered pipeline: **01 → 05** in **`src/notebooks`**.
4. **Optional cloud OCR** — Google Cloud Vision and other engines are available in code for some workflows; they need separate cloud setup and credentials.

---

## Repository layout (short)

| Path | Role |
|------|------|
| `src/main.py` | CLI entry for the box-drawing UI |
| `src/card_selection_ui.py` | UI implementation |
| `src/ocr.py` | OCR backends (Tesseract, Paddle, EasyOCR, Cloud Vision, TrOCR, OpenAI vision) |
| `src/region_processing.py` | Checkbox/OCR batch helpers for regions |
| `quick_accuracy_extract.py` | Sample OCR export for accuracy checks |
| `src/notebooks/*.ipynb` | Processing and analysis pipeline |

---

## Running the UI

From the project root, with your virtual environment activated:

```text
python -m src.main
```

Optional arguments:

```text
python -m src.main --images_dir path\to\images --save_dir path\to\save
python -m src.main --region_mode
```

Use **`--images_dir`** when images are not in the default `sampleimgs` folder. See **RUNBOOK_Team.md** for defaults and troubleshooting.

---

## Quick accuracy script

After slices exist under `data/sliced_cards/<date>/`:

```text
python quick_accuracy_extract.py
python quick_accuracy_extract.py --run_dir data\sliced_cards\MM-DD-YYYY --engine openai
```

OpenAI requires **`OPENAI_API_KEY`**; see **RUNBOOK_Team.md** for full instructions.

---

## Full processing procedure (reference)

1. Clean source images: **`src/notebooks/01_Image_Deskew_and_Trim.ipynb`**.
2. Run the **card box selection UI** to define card-level regions; vertex search aligns crops across images (runtime depends on image count and box count).
3. Run the UI with **`--region_mode`** / `no_find_vertex` workflows on cropped subfolders to mark text fields and checkboxes (see `main.py argparse` and runbook).
4. If needed, train a checkbox model (**`src/notebooks/02_Train_Checkbox_Model.ipynb`**) from labeled crops.
5. **`src/notebooks/03_Header_Text_Detection.ipynb`** — start or extend the results table.
6. **`src/notebooks/04_Body_Detect_Text_And_Checkbox.ipynb`** — body fields and checkbox classification into the table.
7. **`src/notebooks/05_Analyze_Data.ipynb`** — summaries and visualizations.

---

## Dependencies

Install from the project root:

```text
pip install -r requirements.txt
```

Some features (ImageMagick/Wand, cloud APIs, GPU libraries) need extra system or cloud setup described in **RUNBOOK_Team.md**.
