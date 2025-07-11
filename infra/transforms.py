import json
import cv2

def load_transforms(transforms_path):
    with open(transforms_path) as f:
        return json.load(f)

def apply_transforms(image, transforms_config):
    max_size = None
    for t in transforms_config.get("transform", {}).get("transforms", []):
        if t.get("__class_fullname__") == "LongestMaxSize":
            max_size = t.get("max_size", [])[0]
    if max_size:
        h, w = image.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w*scale), int(h*scale)))
    return image
