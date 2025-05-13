import os
from PIL import Image
import sys
root_dir = sys.argv[1]
# root_dir = "/mnt/bn/dichang-bytenas/dichang-seed/xedit/test_log/xedit-bench-test/instructmove_xedit"

for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    gen_path = os.path.join(subfolder_path, "gen_images")
    ref_path = os.path.join(subfolder_path, "ref_images")
    
    if not (os.path.isdir(gen_path) and os.path.isdir(ref_path)):
        continue

    gen_files = [f for f in os.listdir(gen_path) if f.endswith('.png')]

    for file_name in gen_files:
        ref_file = os.path.join(ref_path, file_name)
        gen_file = os.path.join(gen_path, file_name)

        if not os.path.exists(ref_file):
            continue

        ref_img = Image.open(ref_file).resize((512, 512))
        gen_img = Image.open(gen_file).resize((512, 512))

        combined_img = Image.new('RGB', (1024, 512))
        combined_img.paste(ref_img, (0, 0))
        combined_img.paste(gen_img, (512, 0))

        combined_img.save(os.path.join(subfolder_path, file_name))
        print(f"Combined and saved: {os.path.join(subfolder_path, file_name)}")
