
import os
import pandas as pd
import difflib
from google.cloud import vision
import concurrent
from concurrent.futures import ThreadPoolExecutor
from google.api_core.exceptions import GoogleAPIError

def detect_document(path):
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["en"])
    try:
        response = client.document_text_detection(image=image, image_context=image_context)
    except GoogleAPIError as e:
        print(e)
        return ""

    if response.error.message:
        print(response.error.message)
        return ""
    return response.full_text_annotation.text

def best_match(text, options):
    return difflib.get_close_matches(text, options, n=1, cutoff=0.0)[0]

def process_image(index, directories, street_options, base_path):
    record = {'index': index}
    for dir in directories:
        image_path = os.path.join(base_path, dir, index)
        text = detect_document(image_path).lower()
        if dir == "street":
            record["street_raw"] = text
            text = best_match(text, street_options)
        record[dir] = text
    return record

def process_images_concurrently():
    base_path = "data/form_info_001"
    directories = ["street",] #  "street_no", "apartment_no", "structure_no"
    street_options = ["flower", "clay", "grand", "figueroa", "bunker hill ave", "cinnabar", "hope", "west"]

    all_files = sorted(os.listdir(os.path.join(base_path, directories[0])))
    print(all_files)
    data = []

    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_file = {executor.submit(process_image, file, directories, street_options, base_path): file for file in all_files}
        for future in concurrent.futures.as_completed(future_to_file):
            data.append(future.result())

    return pd.DataFrame(data)


df_new = process_images_concurrently()
df_new.set_index('index', inplace=True)
df_old = pd.read_csv("all_processed_text_data.csv", index_col='index')
df = pd.concat([df_old, df_new])
df.to_csv("all_processed_text_data.csv")