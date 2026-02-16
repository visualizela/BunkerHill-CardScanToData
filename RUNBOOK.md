# Runbook: BunkerHill Card Scan to Data

Simple steps to run the card-selection UI and where dependencies can bite.

---

## 1. Environment setup

### Python

- Use Python 3.x (3.8+ recommended).

### Virtual environment (recommended)

**Windows (PowerShell):**
```powershell
cd path\to\BunkerHill-CardScanToData
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (Terminal):**
```bash
cd path/to/BunkerHill-CardScanToData
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python dependencies

Same on both platforms (run with venv activated):

```bash
pip install -r requirements.txt
```

**From `requirements.txt`:** opencv-python-headless, pytesseract, numpy, scikit-learn, Pillow, tqdm, pandas, matplotlib, Wand, pydantic, paddleocr, paddlepaddle, google-cloud-vision (optional for full OCR).

---

## 2. Dependency issues and fixes

### ImageMagick (required for Wand)

**Symptom:**  
`ImportError: MagickWand shared library not found. You probably had not installed ImageMagick library.`

**Cause:** The `Wand` package is only a Python binding; the ImageMagick C library must be installed on the system.

**Fix:**

- **Windows:**  
  1. Download the Windows installer from [ImageMagick](https://imagemagick.org/script/download.php#windows).  
  2. Run the installer (e.g. 64-bit `ImageMagick-7.x.x-Q16-HDRI-x64-dll.exe`).  
  3. Restart the terminal. If the library is still not found, set `MAGICK_HOME` to the ImageMagick install folder (e.g. `C:\Program Files\ImageMagick-7.x.x-Q16-HDRI`).

- **macOS:**  
  Install with Homebrew: `brew install imagemagick`. Restart the terminal. If needed, set `MAGICK_HOME` to the install path (e.g. `/opt/homebrew` on Apple Silicon or `/usr/local` on Intel).

### Running without `--save_dir` (code fixes already in repo)

If you run `python -m src.main` with no arguments, the app uses default paths. The following issues were fixed in the codebase:

- **`main.py`:** Passing `None` as the data directory caused `TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType`. The call now uses `args.save_dir or DATA_DIR` so a valid path is always passed.
- **`card_selection_ui.py`:** When `save_dir` was `None`, `os.path.join(save_dir, "boxes.json")` crashed. The code now checks `save_dir is not None` before using it for loading existing boxes.
- **`card_selection_ui.py`:** Saving could raise `FileNotFoundError` because `_save_outline(base_dir)` was called before `base_dir` existed. The directory is now created with `os.makedirs(base_dir, exist_ok=True)` before the first save.

---

## 3. Where to put images

- **Default:** Put census card images in a folder named **`sampleimgs`** at the project root (same level as `src/`, `README.md`).
- **Custom:** Use any folder and pass it with `--images_dir` (see below). On Windows use backslashes or forward slashes; on macOS/Linux use forward slashes.

---

## 4. Run the card-selection UI

From the project root (with venv activated):

```bash
python -m src.main
```

On macOS/Linux you may need `python3 -m src.main` if `python` points to Python 2.

**Optional arguments:**

- `--images_dir path/to/images` — folder of census card images (default: `sampleimgs`). Use `path\to\images` on Windows, `path/to/images` on macOS/Linux.
- `--save_dir path/to/output` — where to write sliced output (default: under `data/sliced_cards/<date>`).
- `--region_mode` — run on subfolders of sliced boxes (for defining text/checkbox regions); use with `--images_dir` and `--save_dir`.

---

## 5. Using the UI

1. **Draw a box:** Click two corners (e.g. top-left, then bottom-right) around a field.
2. **Name the box:** Press **`t`**, type the name (e.g. `occupancy`, `name`), press **Enter**.
3. Repeat for each field. Use **←** / **→** to move between images.
4. **Save and slice:** Press **Enter** twice to save box definitions and slice all images into the output folder.

**Shortcuts:** **`s`** = save boxes only; **`q`** = quit (unsaved work is lost); **`h`** = help in terminal. Full controls are in `src/help_menu.txt`.

---

## 6. Where results are saved (default)

When you run **without** `--save_dir`:

- **Path:** `data/sliced_cards/<MM-DD-YYYY>/` (use `data\sliced_cards\<MM-DD-YYYY>\` on Windows if you type it).
- **Contents:**  
  - `boxes.json` — box definitions.  
  - One subfolder per box name (e.g. `business units`, `occupancy`), each containing one cropped image per original card.

**Examples:**  
- Windows: `BunkerHill-CardScanToData\data\sliced_cards\02-09-2026\business units\p15799coll8_82111.jpeg`  
- macOS/Linux: `BunkerHill-CardScanToData/data/sliced_cards/02-09-2026/business units/p15799coll8_82111.jpeg`

---

## 7. Optional: full pipeline

For the full flow (deskew, checkbox training, OCR, analysis), run the Jupyter notebooks in order:  
`01_Image_Deskew_and_Trim.ipynb` → `02_Train_Checkbox_Model.ipynb` → `03_Header_Text_Detection.ipynb` → `04_Body_Detect_Text_And_Checkbox.ipynb` → `05_Analyze_Data.ipynb`.  
Google Cloud Vision credentials are needed for full OCR; see the main README.
