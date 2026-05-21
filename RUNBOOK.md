# How to Run the Bunker Hill Card Scan Tool

Step-by-step setup and use on **Windows** or **Mac**: get the project, install prerequisites, optionally set an OpenAI API key, then run the box-drawing tool and find your results.

Copy commands exactly from the gray boxes. If something fails, save a screenshot or copy the **full** error text from the terminal.

**Where to start:**

| Situation | Start here |
|-----------|------------|
| **Don't have the project folder yet** | [Get the project](#step-0-get-the-project-on-your-computer) |
| **Windows** (folder already on PC) | [Windows setup](#windows-setup-one-time-then-each-session) |
| **Mac** (folder already on Mac) | [Mac setup](#mac-setup-one-time-then-each-session) |

[Running the tool](#running-the-tool) and [finding results](#where-to-find-your-results) are the same on both systems once setup is done.

---

## What You Need Before Starting

- **Windows 10/11** or **macOS**
- Internet for first-time downloads
- **Python 3.8 or newer** ([install steps](#install-python-if-you-do-not-have-it-yet))
- Census card images (or a small test set)
- About **30–60 minutes** the first time; later runs are faster

The box-drawing tool does **not** require an OpenAI key. You only need the key if you plan to use **`--engine openai`** in the optional OCR script ([OpenAI API key](#openai-api-key-optional)).

---

## Step 0: Get the Project on Your Computer

Repository: **[BunkerHill-CardScanToData](https://github.com/visualizela/BunkerHill-CardScanToData)**

When finished, you should have a folder **`BunkerHill-CardScanToData`** containing **`src`**, **`requirements.txt`**, and **`README.md`**.

### First time: download the project with `git clone`

1. **Install Git** if you do not have it:
   - **Windows:** [Git for Windows](https://git-scm.com/download/win) — run the installer with default options, then open a **new** PowerShell window.
   - **Mac:** Open **Terminal** and run `git --version`. If you see a version, you are ready. If not, install [Git for macOS](https://git-scm.com/download/mac), or accept macOS's prompt to install **Command Line Tools**.

2. **Clone** (download) the project. Paste the lines for your system into PowerShell or Terminal, pressing Enter after each line:

   **Windows (PowerShell):**

   ```text
   cd $env:USERPROFILE\Desktop
   git clone https://github.com/visualizela/BunkerHill-CardScanToData.git
   cd BunkerHill-CardScanToData
   ```

   **Mac (Terminal):**

   ```text
   cd ~/Desktop
   git clone https://github.com/visualizela/BunkerHill-CardScanToData.git
   cd BunkerHill-CardScanToData
   ```

3. Confirm the folder exists: type **`dir`** on Windows or **`ls`** on Mac. You should see **`src`** and **`requirements.txt`**.

4. Next: [Install Python](#install-python-if-you-do-not-have-it-yet) → [Windows setup](#windows-setup-one-time-then-each-session) or [Mac setup](#mac-setup-one-time-then-each-session).

### Already have the folder? Get the latest version

If you cloned the project before, update it before each work session:

**Windows (PowerShell):**

```text
cd "C:\Users\YourName\Desktop\BunkerHill-CardScanToData"
git pull
```

**Mac (Terminal):**

```text
cd ~/Desktop/BunkerHill-CardScanToData
git pull
```

If `git pull` says **"not a git repository,"** the folder was downloaded as a ZIP. Re-clone using the steps above into a different location.

### If `git clone` will not work — ZIP fallback

1. Open **[https://github.com/visualizela/BunkerHill-CardScanToData](https://github.com/visualizela/BunkerHill-CardScanToData)** in a browser.
2. Click the green **Code** button → **Download ZIP**.
3. Unzip the file and place the folder on your **Desktop** as **`BunkerHill-CardScanToData`**.

**Note:** ZIP copies cannot use **`git pull`** for updates. To update, download a fresh ZIP, or follow the clone steps above. ZIP copies also cannot switch branches, which the optional [quick accuracy script](#optional-quick-accuracy-check-sample-ocr) requires.

---

## Install Python (if you do not have it yet)

You need **Python 3.8 or newer**.

### Windows

1. Download from **[Python for Windows](https://www.python.org/downloads/windows/)** — choose **Python 3.11** or **3.12** if unsure.
2. Run the installer.
3. **Important:** On the first screen, check **"Add python.exe to PATH"**, then click **Install Now**.
4. Close and reopen **PowerShell**.
5. Verify:

   ```text
   python --version
   ```

   You should see `Python 3.12.x` (or similar). If not, reinstall and make sure **Add to PATH** is checked.

### Mac

1. Download from **[Python for macOS](https://www.python.org/downloads/macos/)** — choose **Python 3.11** or **3.12** if unsure.
2. Open the **`.pkg`** file and follow the prompts (Continue → Agree → Install).
3. Close and reopen **Terminal**.
4. Verify (use **`python3`** on Mac, not `python`):

   ```text
   python3 --version
   ```

   You should see `Python 3.12.x` (or similar).

---

## Windows setup (one-time, then each session)

### Open PowerShell

1. Click **Start**, type **PowerShell**, open **Windows PowerShell**.
2. Go to the project folder (change the path to match your computer):

   ```text
   cd "C:\Users\YourName\Desktop\BunkerHill-CardScanToData"
   ```

3. Confirm you are in the right place:

   ```text
   dir
   ```

   You should see **`requirements.txt`**.

### One-time setup on Windows

Do these steps **once per computer**.

**1. Create the virtual environment** (an isolated copy of Python libraries for this project):

```text
python -m venv .venv
```

**2. Activate the virtual environment:**

```text
.\.venv\Scripts\Activate.ps1
```

You should see **`(.venv)`** appear at the start of the line.

**If PowerShell blocks the script** ("running scripts is disabled"), run this one-time command. It allows PowerShell to run trusted scripts for your user account only:

```text
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then run the `Activate.ps1` line again.

**3. Install Python packages** (this can take **5–15 minutes**):

```text
pip install -r requirements.txt
```

When it finishes, the last lines should say **`Successfully installed ...`**. Warnings in yellow are usually OK; red **`ERROR`** lines mean something failed — copy the full text before continuing.

**4. Install ImageMagick** (required for saving sliced images):

- Download: **[ImageMagick for Windows](https://imagemagick.org/script/download.php#windows)** (use the 64-bit **Q16-HDRI** installer).
- Run it with default options.
- **Restart your computer** when the installer finishes.

After restarting, reopen PowerShell, `cd` back into the project folder, and re-activate **`.venv`** before the next steps.

### Every time you work on Windows

```text
cd "C:\Users\YourName\Desktop\BunkerHill-CardScanToData"
.\.venv\Scripts\Activate.ps1
```

Optional update before starting:

```text
git pull
```

Then continue to [Running the tool](#running-the-tool).

---

## Mac setup (one-time, then each session)

### Open Terminal

1. Press **Command (⌘) + Space**, type **Terminal**, press **Enter**.
2. A window opens with a prompt ending in `%` or `$`. That is **Terminal**.

### One-time setup on Mac

Do these steps **once per computer**.

**1. Go to the project folder:**

```text
cd ~/Desktop/BunkerHill-CardScanToData
ls
```

You should see **`requirements.txt`**, **`src`**, and **`README.md`**.

**2. Create the virtual environment:**

```text
python3 -m venv .venv
```

**3. Activate the virtual environment:**

```text
source .venv/bin/activate
```

You should see **`(.venv)`** appear at the start of the line.

**4. Install Python packages** (this can take **5–15 minutes**):

```text
python3 -m pip install -r requirements.txt
```

When it finishes, the last lines should say **`Successfully installed ...`**. Warnings in yellow are usually OK; red **`ERROR`** lines mean something failed — copy the full text before continuing.

**5. Install ImageMagick** (required for saving sliced images)

First check whether **Homebrew** is installed:

```text
brew --version
```

- If you see a version number, skip to the **`brew install`** line below.
- If you see **`brew: command not found`**, install Homebrew first: open **[https://brew.sh](https://brew.sh)**, copy the install command shown on that page, paste into Terminal, and follow the prompts (you may need your Mac password).

Then install ImageMagick:

```text
brew install imagemagick
```

This one-time install runs outside the virtual environment, so close Terminal, reopen it, and re-activate **`.venv`** before the next step:

```text
cd ~/Desktop/BunkerHill-CardScanToData
source .venv/bin/activate
```

If you later see **MagickWand** errors, set **`MAGICK_HOME`** to `/opt/homebrew` (Apple Silicon) or `/usr/local` (older Intel Macs).

**6. Confirm Python is working:**

```text
python3 --version
```

### Every time you work on Mac

```text
cd ~/Desktop/BunkerHill-CardScanToData
source .venv/bin/activate
```

Optional update before starting:

```text
git pull
```

Then continue to [Running the tool](#running-the-tool).

---

## OpenAI API key (optional)

**Skip this section** if you are only drawing boxes and slicing images, or if you will use **EasyOCR** (the default for `quick_accuracy_extract.py`).

The key is **not** needed for `python -m src.main` / `python3 -m src.main` — only for `quick_accuracy_extract.py --engine openai`.

### Get an API key

1. Open **[https://platform.openai.com](https://platform.openai.com)** and **sign up** or **log in**.
2. Go to **[API keys](https://platform.openai.com/api-keys)** and click **Create new secret key**.
3. **Copy the key immediately** — OpenAI usually shows it only **once**. If lost, revoke the old key and create a new one on the same page.
4. Open **[Billing](https://platform.openai.com/settings/organization/billing)** and add a payment method or credits if prompted.

**Security:** Treat the key like a password. Do not email it, paste it in chat, or commit it to GitHub or to files like **`boxes.json`**. Revoke exposed keys at **[API keys](https://platform.openai.com/api-keys)**.

### Set the key on Windows

The variable name is **`OPENAI_API_KEY`**.

**This PowerShell window only** (must repeat each new window):

```text
$env:OPENAI_API_KEY = "sk-your-actual-key-pasted-here"
```

**Saved permanently for your Windows user:**

1. Press the **Windows key**, type **environment variables**, open **Edit environment variables for your account**.
2. Under **User variables**, click **New**.
3. **Name:** `OPENAI_API_KEY` — **Value:** your `sk-...` key.
4. Click **OK** on all dialogs, then close and reopen PowerShell.

### Set the key on Mac

The variable name is **`OPENAI_API_KEY`**.

**This Terminal window only** (must repeat each new window):

```text
export OPENAI_API_KEY="sk-your-actual-key-pasted-here"
```

**Saved permanently for your Mac user** (zsh — default on modern macOS). This appends one line to your shell config and reloads it. Replace the example key first:

```text
echo 'export OPENAI_API_KEY="sk-your-actual-key-pasted-here"' >> ~/.zshrc
source ~/.zshrc
```

On older Macs that use bash, replace **`~/.zshrc`** with **`~/.bash_profile`** in both commands.

### Optional OpenAI settings

- **`OPENAI_OCR_MODEL`** — default **`gpt-4o`**
- **`OPENAI_VISION_DETAIL`** — **`low`** or **`high`**

### Verify the key

After you finish setup and have **`(.venv)`** active in the project folder, run:

**Windows:**

```text
python quick_accuracy_extract.py --engine openai
```

**Mac:**

```text
python3 quick_accuracy_extract.py --engine openai
```

- Missing key → error asking for **`OPENAI_API_KEY`**.
- Success → new CSV and JSON files under **`data`**.

(You also need to switch to the correct branch first — see the [quick accuracy script section](#optional-quick-accuracy-check-sample-ocr).)

---

## Running the tool

Make sure **`(.venv)`** is active and you are in the project folder.

### Add your images

1. Inside the project folder (next to **`src`**), create a folder called **`sampleimgs`** if it does not exist.
2. Copy your census card images into **`sampleimgs`**.
3. Supported types: **`.png`**, **`.jpg`**, **`.jpeg`**, **`.tiff`**, **`.bmp`**, **`.gif`**.

### Start the program

**Windows:**

```text
python -m src.main
```

**Mac:**

```text
python3 -m src.main
```

A window opens with a card image. If you see an "no images" message, add files to **`sampleimgs`** and run the same command again.

### Draw boxes and save

1. Click **two corners** to draw a box around a field.
2. Press **t**, type a short name (e.g. **`year_built`**), press **Enter**.
3. Repeat for each field.
4. Use **Left** / **Right** arrows to move between cards.
5. When finished:
   - Press **Enter** once (confirmation prompt).
   - Press **Enter** again to save and slice all cards.
   - Wait for **`Save successful!`** in the terminal.

| Key | Action |
|-----|--------|
| **s** | Save boxes only (no slice yet) |
| **q** | Quit — **unsaved work is lost** unless you saved first |
| **h** | Print keyboard help in the terminal |
| **m** | Move mode: drag a box; press **m** again to exit |
| **n** | Toggle box name labels on the image |

Full list: **`src/help_menu.txt`**.

### Where to find your results

Open the project's **`data`** folder:

**`data` → `sliced_cards` → `MM-DD-YYYY` → `<box name>`**

Example date folder: **`05-21-2026`**.

- **`boxes.json`** — box names and coordinates
- **One subfolder per box name** — one cropped image per card

You can open this in **File Explorer** (Windows) or **Finder** (Mac).

---

## Optional: Quick accuracy check (sample OCR)

The **`quick_accuracy_extract.py`** script lives on a separate Git branch called **`quick_accuracy_script`**, not on the default `main` branch. You must switch branches before you can run it.

### Switch to the `quick_accuracy_script` branch

In the project folder (PowerShell or Terminal), run:

```text
git fetch
git checkout quick_accuracy_script
git pull
```

Confirm you are on the right branch:

```text
git status
```

The first line should say **`On branch quick_accuracy_script`**.

To go back to the main branch later:

```text
git checkout main
```

**Note:** ZIP downloads cannot switch branches. If you used the ZIP fallback, use [`git clone`](#first-time-download-the-project-with-git-clone) instead.

### Run the script

Run **after** you have saved slices with the box tool. The script writes CSV and JSON files under **`data`**.

- **EasyOCR (default)** — no API key needed
- **OpenAI** — requires `--engine openai` and the [API key setup](#openai-api-key-optional)

**Windows:**

```text
python quick_accuracy_extract.py
python quick_accuracy_extract.py --engine openai
```

**Mac:**

```text
python3 quick_accuracy_extract.py
python3 quick_accuracy_extract.py --engine openai
```

Outputs land in **`data/accuracy_quick_<date-folder>.csv`** and **`.json`**.

**Useful options:**

Windows:

```text
python quick_accuracy_extract.py --run_dir data\sliced_cards\05-21-2026
python quick_accuracy_extract.py --fields year_built condition
```

Mac:

```text
python3 quick_accuracy_extract.py --run_dir data/sliced_cards/05-21-2026
python3 quick_accuracy_extract.py --fields year_built condition
```

---

## If something goes wrong

### Both platforms

| Problem | What to try |
|--------|-------------|
| No images / folder error | Put images in **`sampleimgs`**; run from the **project root** (the folder that contains **`src`**) |
| "Failed to save image" on a card | A box may be off the image — redraw or enlarge that box, save again |
| Window does not open | Confirm **`(.venv)`** is active and **`pip install -r requirements.txt`** finished without **ERROR** lines |
| OpenAI errors | Key set in the **same** terminal session (or saved permanently); check **[billing](https://platform.openai.com/settings/organization/billing)**; recreate key if you lost the secret |
| `quick_accuracy_extract.py` not found | Switch branches: `git checkout quick_accuracy_script` |

### Windows only

| Problem | What to try |
|--------|-------------|
| MagickWand / ImageMagick errors | [ImageMagick installer](https://imagemagick.org/script/download.php#windows), restart PC, reactivate **`.venv`** |
| `python` not found | Reinstall Python with **Add to PATH** checked: [python.org/windows](https://www.python.org/downloads/windows/) |
| Cannot activate `.venv` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then `Activate.ps1` again |

### Mac only

| Problem | What to try |
|--------|-------------|
| MagickWand / ImageMagick errors | Run `brew install imagemagick`; if it persists, set **`MAGICK_HOME`** to `/opt/homebrew` or `/usr/local` |
| `python` not found | Use **`python3`** and **`python3 -m pip`** everywhere |
| `brew: command not found` | Install Homebrew from **[brew.sh](https://brew.sh)** |
| Permission errors during `pip install` | Make sure **`(.venv)`** is active, then use `python3 -m pip install -r requirements.txt` |

---

## Quick reference

| Task | Windows | Mac |
|------|---------|-----|
| Go to project | `cd "C:\...\BunkerHill-CardScanToData"` | `cd ~/Desktop/BunkerHill-CardScanToData` |
| Activate venv | `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| Run box tool | `python -m src.main` | `python3 -m src.main` |
| Switch to OCR branch | `git checkout quick_accuracy_script` | `git checkout quick_accuracy_script` |
| Quick OCR (no key) | `python quick_accuracy_extract.py` | `python3 quick_accuracy_extract.py` |
| OpenAI OCR | set `OPENAI_API_KEY`, then `--engine openai` | same |
| Update project | `git pull` | `git pull` |

**Repository:** [https://github.com/visualizela/BunkerHill-CardScanToData](https://github.com/visualizela/BunkerHill-CardScanToData)
