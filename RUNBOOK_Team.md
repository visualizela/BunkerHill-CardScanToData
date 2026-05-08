# How to Run the Bunker Hill Card Scan Tool

A step-by-step guide for team members. It covers the main box-drawing tool, where outputs go, and optional steps for testing OCR accuracy (including OpenAI).

**Who this is for:** You can follow most of this guide by copying commands exactly. If anything fails, save a screenshot or copy the full error text and share it with your technical lead.

---

## What You Need Before Starting

- A **Windows** PC with the Bunker Hill project folder (the folder that contains **sampleimgs**, **src**, and **data**).
- **Python** installed (version **3.8 or newer**). To check: open PowerShell, type `python --version`, press Enter. If you are not sure Python is installed correctly, ask your technical lead.
- Your census card images, or a small test set (see “Where to Put Your Images”).

---

## Step 1: Open the Project in a Terminal

1. Open **PowerShell** (click Start, type **PowerShell**, open **Windows PowerShell**).
2. Go to the project folder. Type **cd**, a space, then the **full path** to your Bunker Hill folder in quotes, then press Enter.

   Example (your path will be different):

   ```text
   cd "C:\Users\YourName\Desktop\BunkerHill-CardScanToData"
   ```

3. Turn on the project’s **virtual environment** (a bundled copy of the right Python libraries):

   ```text
   .\.venv\Scripts\Activate.ps1
   ```

   After this works, you should see **(.venv)** at the beginning of the line.

   **If PowerShell blocks the script** (“cannot be loaded because running scripts is disabled”), ask your technical lead, or try once:

   ```text
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

   Then run the `Activate.ps1` line again.

**One-time setup (often done once per machine by a technical person):**

1. In the project folder, create the environment:

   ```text
   python -m venv .venv
   ```

2. Activate it (same as step 3 above).

3. Install dependencies:

   ```text
   pip install -r requirements.txt
   ```

4. If the tool later says **MagickWand** or **ImageMagick** is missing, someone with admin rights should install **ImageMagick** for Windows from the official ImageMagick website, then restart the computer.

---

## Step 2: Where to Put Your Images

- Put your census card images inside a folder named **sampleimgs** in the project folder (same level as the **src** folder).
- Supported file types include **.png**, **.jpg**, **.jpeg**, **.tiff**, **.bmp**, and **.gif**.
- If **sampleimgs** does not exist yet, running the tool once may create it. Add your images there, then run the tool again.

---

## Step 3: Start the Tool

1. Make sure you are in the project folder and **(.venv)** shows in the terminal.
2. Run:

   ```text
   python -m src.main
   ```

3. Press Enter. A window should open with a card image.
4. If you see a message about creating folders and no images yet, add images to **sampleimgs** and run **`python -m src.main`** again.

---

## Step 4: Using the Tool (Drawing Boxes and Saving)

1. **Draw a box** around one field on the card (for example name, address, or a checkbox area):
   - Click **once** for one corner.
   - Click **again** for the opposite corner.
2. **Name the box:** Press **t**, type a short name (for example **occupancy** or **year_built**), then press **Enter**.
3. Repeat for each field you care about.
4. Use the **Left** and **Right arrow** keys to move between card images and check that boxes still look correct.
5. When you are done:
   - Press **Enter** once. The program asks you to confirm.
   - Press **Enter** again. It saves and cuts out each region on every card. Wait until you see **Save successful!** in the terminal.

**Other useful keys:**

- **s** — Save box definitions only (no cutting yet).
- **q** — Quit (**unsaved work is lost** unless you saved first).
- **h** — Print more keyboard shortcuts in the terminal (same content as **src/help_menu.txt**).
- **m** — Move mode: click inside a box and drag to reposition it; press **m** again to turn off.
- **n** — Toggle box **name labels** on the main image.

---

## Step 5: Where to Find Your Results

After a successful save, outputs are under the project’s **data** folder.

- Open **data**, then **sliced_cards**, then the folder named with **today’s date** in **month-day-year** order (for example **05-07-2026**).
- Inside that folder you will find:
  - **boxes.json** — list of boxes and coordinates.
  - One **folder per box name**. Each folder contains one cropped image per census card for that field.

Example path (your drive and folder names may differ):

`BunkerHill-CardScanToData → data → sliced_cards → 05-07-2026 → year_built`

---

## Optional: Quick Accuracy Check (Sample OCR on Cropped Fields)

This is a **separate** command-line step after you have saved slices. It reads a **small sample** of cropped images, runs OCR, and writes a CSV and JSON file under **data**. You do **not** need OpenAI for this unless you choose the OpenAI engine (see below).

1. Open PowerShell, **cd** to the project folder, activate **.venv** (see Step 1).
2. Default run (uses the **newest** date folder under **data\sliced_cards**, sample up to 10 images per field, **EasyOCR** — no API key):

   ```text
   python quick_accuracy_extract.py
   ```

3. Outputs:

   - `data\accuracy_quick_<date-folder-name>.csv`
   - `data\accuracy_quick_<date-folder-name>.json`

**Useful options (ask your technical lead if you are unsure):**

- **Specific slice folder:**  
  `python quick_accuracy_extract.py --run_dir data\sliced_cards\05-07-2026`
- **Only some fields:**  
  `python quick_accuracy_extract.py --fields year_built condition`
- **More images per field:**  
  `python quick_accuracy_extract.py --max_images_per_field 25`

Other engines (may need extra setup or keys): `--engine tesseract`, `paddleocr`, `cloudvision`, `trocr`, `openai`. The default **easyocr** is usually enough to spot-check whether crops are readable.

---

## OpenAI API Key (Only If You Use `--engine openai`)

Use this when you want **GPT-4o** (or another vision model) to read handwriting from small crops. **You can ignore this entire section** if you only use **EasyOCR** or other engines.

### Part A — Get an API key

1. Open a web browser and go to **[https://platform.openai.com](https://platform.openai.com)**.
2. **Sign up** or **log in** with an OpenAI account.
3. Open the **API keys** page:  
   **[https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)**  
   (Or: click your **profile** → **Your profile** / **Settings**, then find **API keys**.)
4. Click **Create new secret key** (the exact button name may vary slightly).
5. Give it a simple label (for example **Bunker Hill OCR**), create the key, then **copy the key immediately** and paste it into a **private** place (Notepad is fine **temporarily**).  
   **Important:** OpenAI often shows the full secret **only once**. If you lose it, delete the old key and create a new one.
6. **Billing:** Using the API usually requires a **paid plan** or **credits**. On the same site, open **Settings → Billing** (or **Plans**) and add a payment method if prompted. Your technical lead can help estimate cost for small test runs.

**Security (read this):**

- Treat the key like a **password**. Do **not** put it in email, Slack, screenshots, or **GitHub**.
- Do **not** paste it into **boxes.json** or any file you might commit to git.
- If you think a key was exposed, go back to **API keys**, **revoke** that key, and create a new one.

### Part B — Where to set the key on your Windows PC

You must set an **environment variable** named **`OPENAI_API_KEY`** before running the script. Two common ways:

#### Option 1 — Only for this terminal session (good for first tests)

1. Open **PowerShell**.
2. Set the variable (replace the example with your real key; **keep the quotes**):

   ```text
   $env:OPENAI_API_KEY = "sk-your-actual-key-pasted-here"
   ```

3. **Same window:** `cd` to the project folder, run `.\.venv\Scripts\Activate.ps1`, then run:

   ```text
   python quick_accuracy_extract.py --engine openai
   ```

If you **close** PowerShell, the key is forgotten; repeat step 2 next time (or use Option 2).

#### Option 2 — Persistent for your user account (survives reboot)

1. Press **Windows key**, type **environment variables**, open **Edit environment variables for your account**.
2. Under **User variables**, click **New**.
3. **Variable name:** `OPENAI_API_KEY`  
   **Variable value:** paste your secret key (starts with `sk-`).
4. Click **OK** on all dialogs.
5. **Close and reopen** PowerShell (and your editor if it runs Python), then activate **.venv** and run:

   ```text
   python quick_accuracy_extract.py --engine openai
   ```

**Optional settings (technical):** Your lead can set **`OPENAI_OCR_MODEL`** (default model name used in code: **`gpt-4o`**) and **`OPENAI_VISION_DETAIL`** (often **`low`** or **`high`**) if you need different behavior.

### Part C — Check that it worked

- If the key is **missing**, the script prints an error telling you to set **`OPENAI_API_KEY`** and exits.
- If it **works**, you should see paths to new **CSV** and **JSON** files under **data**, similar to the default **quick_accuracy_extract** run.

---

## If Something Goes Wrong

- **“Image file not found”, “no images”, or error about the image folder**  
  Put images in **sampleimgs**, confirm you ran **`python -m src.main`** from the **project** folder (not from inside **src**).

- **MagickWand / ImageMagick**  
  Install ImageMagick on Windows (official installer), restart the PC, try again.

- **“Failed to save image” for one or two files**  
  A box may be invalid on some cards. Re-run the tool, adjust or enlarge that box, save again.

- **OpenAI errors**  
  Confirm **`OPENAI_API_KEY`** is set in the **same** PowerShell window you use to run the script (unless you used the persistent Windows method). Check billing and that the key was **revoked** then **recreated** if you lost the secret.

- **The window does not open or Python errors**  
  Confirm Python works (`python --version`), **.venv** is active, and **`pip install -r requirements.txt`** finished without errors. Copy the **full** error message for your technical lead.

---

## Going Further (Optional — Full Pipeline)

If your team uses the full pipeline (deskew, checkbox model, text extraction, analysis), that is done in **Jupyter** notebooks **in order**:  
**01 → 02 → 03 → 04 → 05** in the project’s **`src\notebooks`** folder.  
Your technical lead can explain when each step is needed.

For developer-oriented overview and repository layout, see **README.md** in the project root.
