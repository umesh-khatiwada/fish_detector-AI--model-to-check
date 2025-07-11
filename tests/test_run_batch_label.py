import os
import sys
import json
import numpy as np
from PIL import Image

# Add project root to sys.path so 'app' and 'infra' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import types

def setup_dummy_modules():
    # Patch infra.model_inference with dummy functions
    dummy_module = types.ModuleType("infra.model_inference")
    dummy_module.run_inference = lambda model, image_np, device: {
        'pred_boxes': np.array([[10, 10, 50, 50]]),
        'pred_classes': np.array([1]),
        'scores': np.array([0.9])
    }
    dummy_module.load_model = lambda path, device: None
    sys.modules["infra.model_inference"] = dummy_module

def create_dummy_files(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img.save(images_dir / "test1.jpg")

    class_mapping_path = tmp_path / "class_mapping.json"
    with open(class_mapping_path, "w") as f:
        json.dump([{"model_idx": 1, "class_name": "fish"}], f)

    transforms_path = tmp_path / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump({"transform": {"transforms": []}}, f)

    # Save output.json in the current working directory
    output_path = os.path.abspath("output.json")
    return images_dir, class_mapping_path, transforms_path, output_path

def test_run_batch_label(tmp_path):
    setup_dummy_modules()
    from app.use_cases.run_batch_label import run_batch_label

    images_dir, class_mapping_path, transforms_path, output_path = create_dummy_files(tmp_path)

    run_batch_label(
        images_dir=str(images_dir),
        model_path="",  # Not used due to patch
        class_mapping_path=str(class_mapping_path),
        transforms_path=str(transforms_path),
        output_path=str(output_path)
    )

    # Check if output file exists before opening
    assert os.path.exists(output_path), f"Output file {output_path} was not created."

    with open(output_path) as f:
        coco = json.load(f)
    assert coco["images"][0]["file_name"] == "test1.jpg"
    assert coco["annotations"][0]["category_id"] == 1
    assert coco["categories"][0]["name"] == "fish"

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdirname:
        test_run_batch_label(Path(tmpdirname))
        print("Test passed.")
