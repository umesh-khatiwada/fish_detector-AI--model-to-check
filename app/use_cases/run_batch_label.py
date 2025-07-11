import os
from tqdm import tqdm
from infra.image_loader import load_image_files
from infra.model_inference import load_model, run_inference
from infra.transforms import load_transforms, apply_transforms
from infra.class_mapping import load_class_mapping
from infra.coco_format import coco_results
import threading

def process_image(model, device, transforms_config, img_name, image_np, predictions, lock):
    image_np = apply_transforms(image_np, transforms_config)
    h, w = image_np.shape[:2]
    y = run_inference(model, image_np, device)
    pred = {
        "boxes": y['pred_boxes'],
        "labels": y['pred_classes'],
        "scores": y['scores'],
        "height": h,
        "width": w
    }
    with lock:
        predictions.append((img_name, pred))

def run_batch_label(images_dir, model_path, class_mapping_path, transforms_path, output_path):
    import json
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_mapping = load_class_mapping(class_mapping_path)
    transforms_config = load_transforms(transforms_path)
    model = load_model(model_path, device)

    image_files = load_image_files(images_dir)
    predictions = []
    lock = threading.Lock()
    threads = []

    for img_name, image_np in image_files:
        t = threading.Thread(
            target=process_image,
            args=(model, device, transforms_config, img_name, image_np, predictions, lock)
        )
        threads.append(t)
        t.start()

    for t in tqdm(threads):
        t.join()

    # Sort predictions to match image_files order
    predictions_sorted = []
    img_name_to_pred = {img_name: pred for img_name, pred in predictions}
    for img_name, _ in image_files:
        predictions_sorted.append(img_name_to_pred[img_name])

    coco = coco_results([img for img, _ in image_files], predictions_sorted, class_mapping)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"Saved predictions in COCO format to {output_path}")
