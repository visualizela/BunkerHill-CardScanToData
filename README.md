# BunkerHill-CardScanToData

This project is a Python-based application that processes and analyzes images, specifically census cards. It includes functionalities for image preprocessing, text detection, and user interface for card selection.

## Key Features

1. Image Preprocessing: The project includes Jupyter notebooks (`Header Text Detection.ipynb`, `Image Deskew and Trim.ipynb`, `Train Checkbox Model.ipynb`) that perform various image preprocessing tasks such as deskewing, trimming, and text detection.

2. Text Detection: The project uses Google Cloud Vision API for document text detection. Refer to the function detect_document in `Header Text Detection.ipynb`.

3. Card Selection User Interface: The project provides a user interface for selecting sub-data fields on each census card. The controls for this interface are detailed in `src/help_menu.txt`.

## Codebase Structure

The codebase is organized into several Python scripts, Jupyter notebooks, and configuration files. The main scripts include:

- `src/main.py`: The main entry point of the application.
- `src/analyze_cards.py`: Contains code for analyzing the cards.
- `src/card_selection_ui.py`: Contains the user interface for card selection.

The Jupyter notebooks contain code for image preprocessing, text detection, and checkbox classification.


## Running the Project

To run the project, execute the main.py script with the appropriate command-line arguments. For example:

```
pythom -m src.main --images_dir path/to/images --save_dir path/to/save
```

### Note

Please ensure you have the necessary permissions and environment variables set up for using Google Cloud Vision API.

### Running Procedure
1. First, the source images need to be cleaned. Run source images through `notebooks/01_Image_Deskew_and_Trim.ipynb` to produce folders of processed images.
2. Then, use the card box selection UI to select key areas of the box. It uses image similarity to pinpoint the bounding box anchor points to ensure the cropped/sliced cards are perfectly aligned. The runtime may vary, but the expected time to run would be 30 mins per box per 1000 images. Considering there are around 30 boxes and 3000 images, it should take 45 hrs to completely run.
3. Then, run the UI with `--no_find_vertex` on the subfolders that contain the cropped boxes, outlining the text areas and checkboxes, and then naming them.
4. If there has not been a trained checkbox identification model, one must be trained. Use the data created in the last step, and manually copy around 20-30 boxes with checks and boxes without checks to two folders. Then adjust the `notebooks/02_Train_Checkbox_Model.ipynb` and train the model.
5. Then, initiate the CSV by starting the `notebooks/03_Header_Text_Detection.ipynb` notebook, this will put the image index and relevant info into a table.
6. Then, modify and run the `notebooks/04_Body_Detect_Text_And_Checkbox.ipynb` to classify the checkbox and detect the text, then save them to the CSV.
7. Finally, run `notebooks/05_Analyze_Data.ipynb` to generate insight and visualizations for the retrieved card data.
