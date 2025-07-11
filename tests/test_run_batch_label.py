import os
import tempfile
import json
import numpy as np
from PIL import Image

from app.use_cases.run_batch_label import run_batch_label

def test_run_batch_label(tmp_path):
    # Setup dummy images
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img.save(images_dir / "test1.jpg")

    # Dummy model (torch.jit.save a dummy model if needed), here we mock infra/model_inference.py
    # For simplicity, patch run_inference to return fixed boxes/classes/scores
    import sys
    sys.modules['infra.model_inference'].run_inference = lambda model, image_np, device: {
        'pred_boxes': np.array([[10, 10, 50, 50]]),
        'pred_classes': np.array([1]),
        'scores': np.array([0.9])
    }
    sys.modules['infra.model_inference'].load_model = lambda path, device: None

    # Dummy class mapping and transforms
    class_mapping_path = tmp_path / "class_mapping.json"
    with open(class_mapping_path, "w") as f:
        json.dump([{"model_idx": 1, "class_name": "fish"}], f)
    transforms_path = tmp_path / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump({"transform": {"transforms": []}}, f)
    output_path = tmp_path / "output.json"

    run_batch_label(
        images_dir=str(images_dir),
        model_path="dummy.pt",
        class_mapping_path=str(class_mapping_path),
        transforms_path=str(transforms_path),
        output_path=str(output_path)
    )

    with open(output_path) as f:
        coco = json.load(f)
    assert coco["images"][0]["file_name"] == "test1.jpg"
    assert coco["annotations"][0]["category_id"] == 1
    assert coco["categories"][0]["name"] == "fish"
