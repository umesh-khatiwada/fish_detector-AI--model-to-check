import os
from PIL import Image
import numpy as np

def load_image_files(images_dir):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    result = []
    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        result.append((img_name, image_np))
    return result
