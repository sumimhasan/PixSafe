import os
import re
from PIL import Image

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def resize_images(source_folder, target_folder, size=(256, 256), start_index=0):
    os.makedirs(target_folder, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]

    files.sort(key=natural_sort_key)

    print(f"Found {len(files)} images in '{source_folder}'")

    for idx, filename in enumerate(files, start=start_index):
        src_path = os.path.join(source_folder, filename)
        try:
            img = Image.open(src_path).convert('RGB')
            img_resized = img.resize(size, Image.Resampling.LANCZOS)

            # Save all images as JPEG to avoid naming collisions
            save_name = f"img{idx}.jpg"
            save_path = os.path.join(target_folder, save_name)

            img_resized.save(save_path, 'JPEG')
            print(f"Resized and saved '{src_path}' → '{save_path}'")

        except Exception as e:
            print(f"Error processing '{src_path}': {e}")

if __name__ == "__main__":
    source_folder = ""  # your source folder
    target_folder = ""  # your target folder

    resize_images(source_folder, target_folder)
