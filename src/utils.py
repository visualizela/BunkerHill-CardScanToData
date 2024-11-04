import os


def find_images(folder_path, extensions=['.jpg', '.png', '.jpeg', '.gif', '.bmp']):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths
