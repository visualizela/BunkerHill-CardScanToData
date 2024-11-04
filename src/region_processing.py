import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import pandas as pd
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

from src.constants import PROJECT_ROOT_PATH, MAX_WORKERS
from src.ocr import OCR, EngineType


def load_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((25, 25))  # Resize image
    img_array = np.array(img).flatten()  # Flatten image
    return img_array



def predict_image(image_path, folder_name, classifier):
    prediction = classifier.predict([load_image(str(image_path))])
    return image_path.name, prediction[0], folder_name  # Return folder name as well

def process_region_segments_checkbox(base_path: Path, classifier_path, skip_folders: list[str] = [], default_fillna='No Prediction'):
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)
    
    column_name = base_path.name

    # Initialize an empty DataFrame
    df = pd.DataFrame()

    df[column_name] = None

    checkbox_subfolders = [f for f in base_path.iterdir() if f.is_dir() and f.name not in skip_folders]

    with ProcessPoolExecutor() as executor:
        futures = []
        for folder in checkbox_subfolders:
            images = [f for f in folder.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']]

            futures.extend([executor.submit(predict_image, image, folder.name, classifier) for image in images])
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            image_name, prediction, folder_name = future.result()  # Unpack folder_name
            if image_name not in df.index:
                df = pd.concat([df, pd.DataFrame(index=[image_name])], ignore_index=False)
            if prediction == 1:
                if pd.isnull(df.at[image_name, column_name]):
                    df.at[image_name, column_name] = folder_name
                else:
                    df.at[image_name, column_name] += ';' + folder_name

    df = df.fillna(default_fillna)

    return df

def ocr_task(image_path, ocr_processor, column_name):
    if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
        image_index = image_path.name
        ocr_text = ocr_processor.run(image_path).strip()
        return image_index, ocr_text, column_name
    return None, None, None
    
def process_region_segments_ocr(base_path: Path, ocr_engine: EngineType = EngineType.PADDLEOCR, skip_folders: list[str] = [], default_fillna='No OCR Text'):
    ocr_processor = OCR()
    ocr_processor.set_engine(ocr_engine)

    # Initialize an empty DataFrame
    df = pd.DataFrame()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for folder in base_path.iterdir():
            if folder.is_dir() and folder.name not in skip_folders:
                column_name = f"{base_path.name}.{folder.name}"
                df[column_name] = None  # Initialize the column for this subfolder

                images = folder.glob('*')
                futures.extend([executor.submit(ocr_task, image, ocr_processor, column_name) for image in images])

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing OCR"):
            image_index, ocr_text, column_name = future.result()
            if image_index and ocr_text:
                # If the DataFrame does not yet have a row for this image, add it
                if image_index not in df.index:
                    df = pd.concat([df, pd.DataFrame(index=[image_index])], ignore_index=False)
                
                # Set the OCR text for the current image in the appropriate column
                df.at[image_index, column_name] = ocr_text

    # Fill NaN values for rows where no OCR text was extracted
    df = df.fillna(default_fillna)

    return df


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    print(process_region_segments_checkbox(PROJECT_ROOT_PATH / "data/06_Segments/heating"))
    print(process_region_segments_checkbox(PROJECT_ROOT_PATH / "data/06_Segments/race"))
    print(process_region_segments_ocr(PROJECT_ROOT_PATH / "data/06_Segments/rent", EngineType.PADDLEOCR))
    print(process_region_segments_ocr(PROJECT_ROOT_PATH / "data/06_Segments/roomers", EngineType.PADDLEOCR))
