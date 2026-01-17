import json
import os
from tqdm import tqdm
import argparse

def find_img(img_name, coco_dirs):
    coco_prefixes = [
        "COCO_train2014_",
        "COCO_val2014_",
        "COCO_test2014_",
        "COCO_test2015_", # Extended for potential other datasets
    ]

    for coco_dir in coco_dirs:
        for prefix in coco_prefixes:
            # Try specific prefix
            path = os.path.join(coco_dir, prefix + img_name)
            if os.path.exists(path):
                return path
            
            # Try direct join just in case
            path = os.path.join(coco_dir, img_name)
            if os.path.exists(path):
                return path
                
    return None

def main():
    parser = argparse.ArgumentParser(description="Prepare LLaVA training data")
    parser.add_argument("--input_json", type=str, default="llava_instruct_150k.json", help="Input JSON file")
    parser.add_argument("--output_jsonl", type=str, default="llava_train.jsonl", help="Output JSONL file")
    parser.add_argument("--coco_path", type=str, default="coco2014", help="Path to COCO dataset root")
    args = parser.parse_args()

    coco_dirs = [
        os.path.join(args.coco_path, "images", "train2014"),
        os.path.join(args.coco_path, "images", "val2014"),
        os.path.join(args.coco_path, "images", "test2014"),
    ]

    print(f"Checking for images in: {coco_dirs}")

    if not os.path.exists(args.input_json):
        print(f"Error: Input file {args.input_json} not found.")
        return

    print(f"Loading {args.input_json}...")
    with open(args.input_json, "r") as f:
        llava_data = json.load(f)

    valid = 0
    missing = 0

    print(f"Processing {len(llava_data)} samples...")
    with open(args.output_jsonl, "w") as out:
        for sample in tqdm(llava_data):
            image_name = sample.get("image", "")
            if not image_name:
                # Text-only sample?
                continue
                
            image_path = find_img(image_name, coco_dirs)

            if image_path is None:
                missing += 1
                continue

            record = {
                "id": sample.get("id"),
                "image": image_path,
                "conversations": sample["conversations"]
            }

            out.write(json.dumps(record) + "\n")
            valid += 1

    print(f"Valid samples: {valid}")
    print(f"Missing images: {missing}")
    print(f"Saved to: {args.output_jsonl}")

if __name__ == "__main__":
    main()
