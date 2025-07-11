# Fish Detector - Object Detection Inference

This project provides a Python script to perform object detection using a trained TorchScript model.

## Setup

1. **Clone this repository** (if not already done).

2. **Create a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the inference script with your model, class mapping, and image:

```bash
python object_detector_inference.py model.pt class_mapping.json /path/to/image.jpg
```

- `model.pt`: Path to your exported TorchScript model.
- `class_mapping.json`: JSON file mapping model indices to class names.
- `/path/to/image.jpg`: Path to the image you want to run detection on.

## Output

The script will display the input image with predicted bounding boxes and class labels drawn on detected objects.

## Files

- `object_detector_inference.py`: Main inference script.
- `requirements.txt`: Python dependencies.
- `class_mapping.json`: Example class mapping file (you must provide this).
- `model.pt`: Your trained TorchScript model (you must provide this).

## Notes

- Make sure your `class_mapping.json` is in the format:
  ```json
  [
    {"model_idx": 0, "class_name": "fish"},
    {"model_idx": 1, "class_name": "crab"}
    // ...
  ]
  ```
- The script supports both Faster RCNN and FBNetv3 TorchScript models as described in the code.

# Fish Detector - Batch Labeling Pipeline

## Multi-threaded Batch Inference

The batch labeling pipeline uses **multi-threading** to speed up inference on large image batches. Each image is processed in a separate thread, allowing for parallel execution and improved throughput, especially when I/O or preprocessing is a bottleneck.

**How it works:**
- Each image is loaded and preprocessed in its own thread.
- Inference is run in parallel for all images.
- Results are collected and merged in the original order for COCO output.

This approach is especially effective for CPU-bound or I/O-bound workloads, and is safe for PyTorch inference as long as the model is only used for reading (inference) and not training.

## Quickstart

1. **Build the Docker image:**
   ```bash
   docker build -t fish-detector .
   ```

2. **Run the batch labeling pipeline:**
   ```bash
   docker run --rm -v $(pwd):/app fish-detector \
     python batch_label.py --images_dir images --output predictions_coco.json
   ```

   - Place your images in the `images/` directory.
   - The output will be saved as `predictions_coco.json` in COCO format.

## Pipeline Steps

- Deploys a Docker container for reproducible inference.
- Applies image augmentations as specified in `transforms.json`.
- **Runs inference using multi-threading for faster batch processing.**
- Outputs predictions in COCO format.
- Container is removed after completion.

## Files

- `batch_label.py`: Batch labeling script.
- `Dockerfile`: For containerized execution.
- `requirements.txt`: Python dependencies.
- `transforms.json`, `class_mapping.json`, `model.pt`: Model and config files (provide your own).

## Troubleshooting

### OpenCV ImportError: `libGL.so.1: cannot open shared object file: No such file or directory`

If you see an error like:

```bash
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

You need to install the missing library. On Ubuntu, you can do this with:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
```

For other systems, please install the equivalent package using your package manager.

### OpenCV ImportError: `libgthread-2.0.so.0: cannot open shared object file: No such file or directory`

If you see an error like:

```bash
ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
```

You need to install the missing library. In your Dockerfile, add:

```dockerfile
RUN apt-get update && apt-get install -y libglib2.0-0 && rm -rf /var/lib/apt/lists/*
```

This should be added **before** installing Python dependencies.

For local Ubuntu systems, you can run:

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-0
```
