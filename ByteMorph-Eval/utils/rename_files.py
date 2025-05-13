import os


root_dir = "/mnt/bn/dichang-bytenas/dichang-seed/xedit/test_log/xedit-bench-test/wantrain_lora35000"

for subdir, _, files in os.walk(root_dir):
    for fname in files:
        if fname.endswith(".png") and "_edit" in fname:
            old_path = os.path.join(subdir, fname)
            new_fname = fname.replace("_edit", "")
            new_path = os.path.join(subdir, new_fname)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} → {new_path}")
