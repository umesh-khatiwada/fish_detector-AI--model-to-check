def coco_results(images, predictions, class_mapping):
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    ann_id = 1
    for idx, (img_name, pred) in enumerate(zip(images, predictions)):
        img_id = idx + 1
        h, w = pred['height'], pred['width']
        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "height": h,
            "width": w
        })
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            x1, y1, x2, y2 = map(float, box)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [x1, y1, x2-x1, y2-y1],
                "score": float(score),
                "area": float((x2-x1)*(y2-y1)),
                "iscrowd": 0
            })
            ann_id += 1
    for idx, name in class_mapping.items():
        coco["categories"].append({"id": int(idx), "name": name})
    return coco
