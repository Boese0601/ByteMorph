import os
import shutil
from tqdm import tqdm

def copy_matching_json(src_json_root, tgt_json_root):
    for subdir, _, files in os.walk(tgt_json_root):
        for fname in tqdm(files):
            if not fname.endswith(".png"):
                continue
            rel_dir = os.path.relpath(subdir, tgt_json_root)
            json_name = fname.replace(".png", ".json")

            src_json_path = os.path.join(src_json_root, rel_dir, json_name)
            tgt_json_path = os.path.join(tgt_json_root,rel_dir)


            if os.path.exists(src_json_path):
                shutil.copy2(src_json_path, tgt_json_path)
            else:
                print(f"⚠️ Missing source JSON: {src_json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_json_root", required=True, help="Source JSON root directory")
    parser.add_argument("--tgt_json_root", required=True, help="Target PNG directory to match structure")
    args = parser.parse_args()

    copy_matching_json(args.src_json_root, args.tgt_json_root)
