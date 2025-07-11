import argparse
from app.use_cases.run_batch_label import run_batch_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--model', default='model.pt')
    parser.add_argument('--class_mapping', default='class_mapping.json')
    parser.add_argument('--transforms', default='transforms.json')
    parser.add_argument('--output', default='predictions_coco.json')
    args = parser.parse_args()

    run_batch_label(
        images_dir=args.images_dir,
        model_path=args.model,
        class_mapping_path=args.class_mapping,
        transforms_path=args.transforms,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
