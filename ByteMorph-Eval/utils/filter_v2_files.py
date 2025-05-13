import os
from tqdm import tqdm
src_src_root = "/mnt/bn/dichang-bytenas/dichang-seed/xedit/test_log/xedit-bench-test" 

ref_root = "/mnt/bn/dichang-bytenas/dichang-seed/xedit/data/evaluation_dataset/X-Edit-Bench_test/output_bench_v2"

for model_name in tqdm(sorted(os.listdir(src_src_root))):
    print("Process: ", model_name)
    src_root = os.path.join(src_src_root,model_name)
    for subfolder in os.listdir(src_root):
        src_subfolder = os.path.join(src_root, subfolder)
        ref_subfolder = os.path.join(ref_root, subfolder)

        if not os.path.isdir(src_subfolder) or not os.path.isdir(ref_subfolder):
            continue

        for filename in os.listdir(src_subfolder):
            src_file = os.path.join(src_subfolder, filename)
            ref_file = os.path.join(ref_subfolder, filename)

            if not os.path.exists(ref_file):
                print(f"Removing: {src_file}")
                os.remove(src_file)
