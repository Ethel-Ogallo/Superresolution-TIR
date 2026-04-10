import os
import json
import argparse


def derive_campaign_id(entry: dict, patch_id: str) -> str:
    """Derive campaign ID (e.g., PDR/DZM/BRC) from metadata fields."""
    hr_image_id = entry.get("hr_image_id")
    if isinstance(hr_image_id, str) and hr_image_id:
        return hr_image_id.split("_")[0]
    return patch_id.split("_")[0]

def create_metadata(
    processed_dir: str,
    hr_patch_dir: str,
    lr_patch_dir: str,
    output_path: str
):
    """
    Merge per-scene JSONs in `processed_dir` into one global JSON metadata,
    adding absolute HR/LR patch paths.
    """
    all_metadata = []

    # Scan processed_dir for all JSON files
    metadata_dir = os.path.join(processed_dir, "metadata")

    for fname in sorted(os.listdir(metadata_dir)):
        if not fname.endswith(".json"):
            continue
        scene_path = os.path.join(metadata_dir, fname)
        with open(scene_path, "r") as f:
            scene_meta = json.load(f)

        for entry in scene_meta:
            patch_id = entry.get("patch_id", entry["patch_name"])

            # Absolute paths
            hr_file = patch_id if patch_id.endswith(".tif") else patch_id + ".tif"
            lr_file = patch_id if patch_id.endswith(".tif") else patch_id + ".tif"

            hr_path = os.path.abspath(os.path.join(hr_patch_dir, hr_file))
            lr_path = os.path.abspath(os.path.join(lr_patch_dir, lr_file))

            # Skip missing files
            if not os.path.exists(hr_path):
                print(f"[WARN] Missing HR patch: {hr_path}")
                continue
            if not os.path.exists(lr_path):
                print(f"[WARN] Missing LR patch: {lr_path}")
                continue

            # Add patch_id and paths
            entry["patch_id"] = patch_id
            entry["campaign_id"] = derive_campaign_id(entry, patch_id)
            entry["hr_path"] = hr_path
            entry["lr_path"] = lr_path

            all_metadata.append(entry)

    # Save global metadata JSON
    with open(output_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n[INFO] Merged {len(all_metadata)} patches → {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build global patch metadata JSON for all scenes."
    )
    parser.add_argument("--processed_dir", required=True,
                        help="Directory containing per-scene JSONs.")
    parser.add_argument("--hr_patch_dir", required=True,
                        help="Directory containing HR patches.")
    parser.add_argument("--lr_patch_dir", required=True,
                        help="Directory containing LR patches.")
    parser.add_argument("--output_json", default="global_patch_metadata.json",
                        help="Path to save the global metadata JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    create_metadata(
        processed_dir=args.processed_dir,
        hr_patch_dir=args.hr_patch_dir,
        lr_patch_dir=args.lr_patch_dir,
        output_path=args.output_json
    )


if __name__ == "__main__":
    main()

# usage
# python scripts/preprocess/create_metadata.py \
#   --processed_dir data/processed \
#   --hr_patch_dir data/processed/tir_patches/HR \
#   --lr_patch_dir data/processed/tir_patches/LR \
#   --output_json data/full_metadata.json