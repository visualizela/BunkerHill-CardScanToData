from pathlib import Path
from enum import Enum, auto
import base64
import logging
import mimetypes
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
    EASYOCR = auto()
    TROCR = auto()
    OPENAI_VISION = auto()


class OCR:
    def __init__(self):
        self.engine_type = EngineType.TESSERACT
        self.engine_model = None

    def set_engine(self, engine_type: EngineType):
        self.engine_type = engine_type
        self.engine_model = None

    def run(self, image_path: Path, field_context: str = "") -> str:
        if self.engine_type == EngineType.TESSERACT:
            return self._tesseract_ocr(image_path)
        elif self.engine_type == EngineType.CLOUDVISION:
            return self._cloudvision_ocr(image_path)
        elif self.engine_type == EngineType.PADDLEOCR:
            return self._paddleocr_ocr(image_path)
        elif self.engine_type == EngineType.EASYOCR:
            return self._easyocr_ocr(image_path)
        elif self.engine_type == EngineType.TROCR:
            return self._trocr_ocr(image_path)
        elif self.engine_type == EngineType.OPENAI_VISION:
            return self._openai_vision_ocr(image_path, field_context)
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
            # PaddleOCR's constructor arguments vary across versions.
            # Use a minimal, broadly compatible initialization (CPU by default).
            self.engine_model = PaddleOCR(lang="en", ocr_version="PP-OCRv4")
        result = self.engine_model.ocr(str(path), cls=False, det=False)  # Convert Path object to string for compatibility
        if DEBUG:
            print(result)
        if not result or not result[0]:
            return ""

        parts: list[str] = []
        for item in result[0]:
            # Common formats: (text, score) or [text, score]
            if isinstance(item, (list, tuple)) and item:
                parts.append(str(item[0]))
            else:
                parts.append(str(item))
        return "".join(parts)

    def _easyocr_ocr(self, path):
        """EasyOCR: pip-only, works on Windows without separate binaries."""
        import easyocr
        if self.engine_model is None:
            self.engine_model = easyocr.Reader(["en"], gpu=False, verbose=False)
        result = self.engine_model.readtext(str(path))
        # readtext returns list of (bbox, text, confidence)
        return " ".join(item[1] for item in result).strip() if result else ""

    def _trocr_ocr(self, path):
        """TrOCR: transformer-based handwriting recognition via Hugging Face."""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if self.engine_model is None:
            model_name = "microsoft/trocr-large-handwritten"
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.engine_model = (processor, model)

        processor, model = self.engine_model
        image = Image.open(path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _openai_vision_ocr(self, path: Path, field_context: str) -> str:
        """
        GPT-4o (or OPENAI_OCR_MODEL) vision: transcribe a single field crop.
        Requires OPENAI_API_KEY. Optional: OPENAI_OCR_MODEL (default gpt-4o),
        OPENAI_VISION_DETAIL (default low).
        """
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in the environment (or .env) before using --engine openai."
            )

        if self.engine_model is None:
            self.engine_model = OpenAI(api_key=api_key)

        client = self.engine_model
        model = os.environ.get("OPENAI_OCR_MODEL", "gpt-4o")
        detail = os.environ.get("OPENAI_VISION_DETAIL", "low")

        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"

        with open(path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        field_line = (
            f"The field name or path is: {field_context!r}."
            if field_context.strip()
            else "No field name was provided."
        )
        instructions = (
            "You transcribe handwritten text from historical U.S. census or dwelling-schedule card crops. "
            + field_line
            + " Return ONLY the transcribed text as plain text: no quotes, labels, markdown, or explanation. "
            "If the crop has no handwriting or is unreadable, return an empty string."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": detail},
                        },
                    ],
                }
            ],
            max_tokens=512,
        )
        choice = response.choices[0].message
        text = (choice.content or "").strip()
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
            text = self.ocr.run(Path(image_path), field_context=dir).lower()
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


