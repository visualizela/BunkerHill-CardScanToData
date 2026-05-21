# BunkerHill-CardScanToData

Python tools for processing census cards: deskew/trim, interactive box selection, OCR, checkbox classification, and analysis.

## Setup and usage

**To install, configure, and run this project on Windows or Mac, follow [RUNBOOK.md](RUNBOOK.md).** It covers getting the code (git or ZIP), Python and ImageMagick setup, the optional OpenAI API key, switching to the `quick_accuracy_script` branch, running the box-drawing tool, and finding your sliced outputs.

---

## Key features

1. **Card selection UI** — Draw and name regions on cards; export aligned crops and `boxes.json`. Entry point: `src/main.py`. Keyboard reference: `src/help_menu.txt`.
2. **Quick accuracy extraction** (on the **`quick_accuracy_script`** branch) — `quick_accuracy_extract.py` samples cropped fields, runs OCR (default: **EasyOCR**, no API key), and writes CSV/JSON under `data/`. Optional **`--engine openai`** uses GPT-4o vision.
3. **Notebooks** — Image prep, checkbox training, header/body text and checkboxes, analysis. Numbered pipeline: **01 → 05** in **`src/notebooks`**.
4. **Optional cloud OCR** — Google Cloud Vision and other engines are available in code for some workflows; they need separate cloud setup and credentials.

---

## Repository layout

| Path | Role |
|------|------|
| `src/main.py` | CLI entry for the box-drawing UI |
| `src/card_selection_ui.py` | UI implementation |
| `src/ocr.py` | OCR backends (Tesseract, Paddle, EasyOCR, Cloud Vision, TrOCR, OpenAI vision) |
| `src/region_processing.py` | Checkbox/OCR batch helpers for regions |
| `quick_accuracy_extract.py` | Sample OCR export (on the `quick_accuracy_script` branch) |
| `src/notebooks/*.ipynb` | Processing and analysis pipeline |

---

## Full processing pipeline (notebooks)

For the complete workflow beyond box selection, run the Jupyter notebooks in order:

1. **`src/notebooks/01_Image_Deskew_and_Trim.ipynb`** — clean source images.
2. Run the **card box selection UI** (`python -m src.main`) to define card-level regions; vertex search aligns crops across images.
3. Run the UI with **`--region_mode`** on cropped subfolders to mark text fields and checkboxes.
4. **`src/notebooks/02_Train_Checkbox_Model.ipynb`** — train a checkbox model from labeled crops, if needed.
5. **`src/notebooks/03_Header_Text_Detection.ipynb`** — start or extend the results table.
6. **`src/notebooks/04_Body_Detect_Text_And_Checkbox.ipynb`** — body fields and checkbox classification into the table.
7. **`src/notebooks/05_Analyze_Data.ipynb`** — summaries and visualizations.

The notebook pipeline may require **Jupyter**, additional credentials (e.g. Google Cloud Vision), and more disk space than the basic UI workflow.

---

## Dependencies

Install from the project root inside an active virtual environment — see **[RUNBOOK.md](RUNBOOK.md)** for the full setup:

```text
pip install -r requirements.txt
```

Some features (ImageMagick/Wand, cloud APIs, GPU libraries) need extra system or cloud setup described in **[RUNBOOK.md](RUNBOOK.md)**.
