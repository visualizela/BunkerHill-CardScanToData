from pathlib import Path
from enum import Enum, auto
import logging
import os
import difflib
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from google.cloud import vision
import pytesseract
from PIL import Image
from paddleocr import PaddleOCR
import pandas as pd

from src.constants import DEBUG, PROJECT_ROOT_PATH


class EngineType(Enum):
    TESSERACT = auto()
    CLOUDVISION = auto()
    PADDLEOCR = auto()


class OCR:
    def __init__(self):
        self.engine_type = EngineType.TESSERACT
        self.engine_model = None

    def set_engine(self, engine_type: EngineType):
        self.engine_type = engine_type
        self.engine_model = None

    def run(self, image_path: Path) -> str:
        if self.engine_type == EngineType.TESSERACT:
            return self._tesseract_ocr(image_path)
        elif self.engine_type == EngineType.CLOUDVISION:
            return self._cloudvision_ocr(image_path)
        elif self.engine_type == EngineType.PADDLEOCR:
            return self._paddleocr_ocr(image_path)
        else:
            raise ValueError(f"Invalid engine type: {self.engine_type}")

    def _tesseract_ocr(self, path):
        custom_config = r'--oem 1 --psm 7' # https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage
        image = Image.open(path)
        text = pytesseract.image_to_string(image, config=custom_config)
        return text

    
    def _cloudvision_ocr(self, path):
        client = vision.ImageAnnotatorClient()
        with open(path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        image_context = vision.ImageContext(language_hints=["en"])
        response = client.document_text_detection(image=image, image_context=image_context)

        if DEBUG:
            self.__debug_print_response(response)
        if response.error.message:
            raise Exception(response.error.message)
        return response.full_text_annotation.text

    def _paddleocr_ocr(self, path):
        if self.engine_model is None:
            if DEBUG:
                logging.getLogger('ppocr').setLevel(logging.DEBUG)
            else:
                logging.getLogger('ppocr').setLevel(logging.WARN)
            self.engine_model = PaddleOCR(use_angle_cls=True, use_gpu=False, ocr_version='PP-OCRv4', lang='en')  # Initialize PaddleOCR. Modify 'lang' as needed.
        result = self.engine_model.ocr(str(path), cls=False, det=False)  # Convert Path object to string for compatibility
        if DEBUG:
            print(result)
        text = ""
        for line in result[0]:  # Assuming result[0] contains the OCR results
            text += line[0]
        return text

    def __debug_print_response(self, response):
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                print(f"\nBlock confidence: {block.confidence}\n")

                for paragraph in block.paragraphs:
                    print("Paragraph confidence: {}".format(paragraph.confidence))

                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        print(
                            "Word text: {} (confidence: {})".format(
                                word_text, word.confidence
                            )
                        )

                        for symbol in word.symbols:
                            print(
                                "\tSymbol: {} (confidence: {})".format(
                                    symbol.text, symbol.confidence
                                )
                            )
class ImageOCRProcessor:
    def __init__(self, ocr_engine: EngineType, base_path: Path, directories: list, street_options: list):
        self.ocr = OCR()
        self.ocr.set_engine(ocr_engine)
        self.base_path = base_path
        self.directories = directories
        self.street_options = street_options

    def best_match(self, text, options):
        return difflib.get_close_matches(text, options, n=1, cutoff=0.0)[0]

    def process_image(self, index):
        record = {'index': index}
        for dir in self.directories:
            image_path = os.path.join(self.base_path, dir, index)
            text = self.ocr.run(image_path).lower()
            if dir == "street":
                record['street_raw'] = text  # Store the raw OCR result for street
                text = self.best_match(text, self.street_options)
            record[dir] = text
        return record

    def process_images_sync(self):
        all_files = os.listdir(os.path.join(self.base_path, self.directories[0]))[:100] if DEBUG else os.listdir(os.path.join(self.base_path, self.directories[0]))
        data = []
        for index in tqdm(all_files, desc="Processing images synchronously"):
            if not index.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                continue
            data.append(self.process_image(index))
        return pd.DataFrame(data)

    def process_images_concurrently(self):
        all_files = os.listdir(os.path.join(self.base_path, self.directories[0]))[:100] if DEBUG else os.listdir(os.path.join(self.base_path, self.directories[0]))
        data = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(self.process_image, file): file for file in all_files}
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(all_files), desc="Processing images concurrently"):
                data.append(future.result())
        return pd.DataFrame(data)


