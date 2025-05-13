"""
from v2 -> v3: add CLIP Score and CLIP I.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import time
import torch
import clip
import json
import os
import io

from tqdm import tqdm
from PIL import Image
from scipy import spatial
from torch import nn
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from pyiqa.utils.img_util import is_image_file

from optical_flow import calculate_optical_flow, backward_warp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Clip-I: call the clip model and self implement it

########################### Basic Func ################################

def imread(img_source, rgb=False, target_size=None):
    """Read image
    Args:
        img_source (str, bytes, or PIL.Image): image filepath string, image contents as a bytearray or a PIL Image instance
        rgb: convert input to RGB if true
        target_size: resize image to target size if not None
    """
    if type(img_source) == bytes:
        img = Image.open(io.BytesIO(img_source))
    elif type(img_source) == str:
        assert is_image_file(img_source), f'{img_source} is not a valid image file.'
        img = Image.open(img_source)
    elif type(img_source) == Image.Image:
        img = img_source
    else:
        raise Exception("Unsupported source type")
    if rgb:
        img = img.convert('RGB')
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)
    return img

########################### Evaluation ################################

def eval_warp_err(image_pairs, score_type="max"):
    criterion = nn.L1Loss()
    eval_score = 0
    for img_pair in tqdm(image_pairs):
        gt_img = Image.open(img_pair[1]).convert('RGB')
        gt_img = gt_img.resize((512, 512))
        gt_img = transforms.ToTensor()(gt_img).unsqueeze(0)
        err_list = []
        for gen_img_path in img_pair[0]:
            gen_img = Image.open(gen_img_path).convert('RGB')
            # resize to gt size
            gen_img = gen_img.resize((512, 512))
            # convert to tensor
            gen_img = transforms.ToTensor()(gen_img).unsqueeze(0)

            flow = calculate_optical_flow(gt_img, gen_img)
            warped_img, _ = backward_warp(gen_img, flow)
            warped_img = warped_img.squeeze(0)
            # calculate distance
            per_score = criterion(warped_img, gt_img).detach().cpu().numpy().item()
            err_list.append(per_score)
        if score_type == 'max':
            err_score = min(err_list)
        else:
            err_score = sum(err_list) / len(err_list)
        eval_score = eval_score + err_score

    return eval_score / len(image_pairs)


def eval_distance(image_pairs, metric='l1'):
    """
    Using pytorch to evaluate l1 or l2 distance
    """
    if metric == 'l1':
        criterion = nn.L1Loss()
    elif metric == 'l2':
        criterion = nn.MSELoss()
    eval_score = 0
    for img_pair in tqdm(image_pairs):
        gen_img = Image.open(img_pair[0]).convert('RGB')
        gt_img = Image.open(img_pair[1]).convert('RGB')
        # resize to gt size
        gen_img = gen_img.resize(gt_img.size)
        # convert to tensor
        gen_img = transforms.ToTensor()(gen_img)
        gt_img = transforms.ToTensor()(gt_img)
        # calculate distance
        per_score = criterion(gen_img, gt_img).detach().cpu().numpy().item()
        eval_score += per_score

    return eval_score / len(image_pairs)


def eval_clip_i(image_pairs, model, transform, metric='clip_i', score_type='mean'):
    """
    Calculate CLIP-I score, the cosine similarity between the generated image and the ground truth image
    """
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            if metric == 'clip_i':
                image_features = model.encode_image(image_input).detach().cpu().float()
            elif metric == 'dino':
                image_features = model(image_input).detach().cpu().float()
        return image_features
    # model, transform = clip.load("ViT-B/32", device)
    eval_score = 0
    for img_pair in tqdm(image_pairs):
        similarity_list = []
        generated_features_list = []
        gt_features = encode(Image.open(img_pair[1]).convert('RGB'), model, transform)
        for gen_img_path in img_pair[0]:
            generated_features = encode(Image.open(gen_img_path).convert('RGB'), model, transform)
            generated_features_list.append(generated_features)
        
            similarity = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                                    gt_features.view(gt_features.shape[1]))
            if similarity > 1 or similarity < -1:
                raise ValueError(" strange similarity value")
            similarity_list.append(similarity)
        if score_type == 'max':
            similarity = max(similarity_list)
        else:
            similarity = sum(similarity_list) / len(similarity_list)
        eval_score = eval_score + similarity
        
    return eval_score / len(image_pairs)


def eval_clip_score(image_pairs, clip_metric, caption_dict):
    """
    Calculate CLIP score, the cosine similarity between the image and caption
    return gen_clip_score, gt_clip_score
    """
    trans = transforms.Compose([
        transforms.Resize(256),  # scale to 256x256
        transforms.CenterCrop(224),  # crop to 224x224
        transforms.ToTensor(),  # convert to pytorch tensor
    ])

    def clip_score(image_path, caption):
        image = Image.open(image_path).convert('RGB')
        image_tensor = trans(image).to(device)
        return clip_metric(image_tensor, caption).detach().cpu().float()
    
    gen_clip_score = 0
    gt_clip_score = 0
    
    for img_pair in tqdm(image_pairs):
        gen_img_path = img_pair[0]
        gt_img_path = img_pair[1]
        gt_img_name = gt_img_path.split('/')[-1]
        gt_caption = caption_dict[gt_img_name]
        gen_clip_score += clip_score(gen_img_path, gt_caption)
        gt_clip_score += clip_score(gt_img_path, gt_caption)

    
    return gen_clip_score / len(image_pairs), gt_clip_score / len(image_pairs)


def eval_clip_t(image_pairs, model, transform, score_type="mean"):
    """
    Calculate CLIP-T score, the cosine similarity between the image and the text CLIP embedding
    """
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).detach().cpu().float()
        return image_features
    # model, transform = clip.load("ViT-B/32", device)
    gen_clip_t = 0
    gt_clip_t = 0
    clip_dir = 0
    
    for img_pair in tqdm(image_pairs):
        gen_img_path = img_pair[0]
        gt_img_path = img_pair[1]

        gen_img_caption = img_pair[2]
        gt_caption = img_pair[3]

        # get text CLIP embedding
        gt_text_features = clip.tokenize(gt_caption).to(device)
        gen_text_features = clip.tokenize(gen_img_caption).to(device)
        with torch.no_grad():
            gt_text_features = model.encode_text(gt_text_features).detach().cpu().float()
            gen_text_features = model.encode_text(gen_text_features).detach().cpu().float()

        gt_features = encode(Image.open(gt_img_path).convert('RGB'), model, transform)

        gen_clip_t_list = []
        clip_dir_list = []
        for img_path in gen_img_path:
            generated_features = encode(Image.open(img_path).convert('RGB'), model, transform)
            gen_clip_t_list.append(1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                                    gen_text_features.view(gen_text_features.shape[1])))
        
            clip_dir_list.append(1 - spatial.distance.cosine((generated_features - gt_features).view(generated_features.shape[1]),
                                                    (gen_text_features - gt_text_features).view(gt_text_features.shape[1])))

        if score_type == 'max':
            gen_clip_t_item = max(gen_clip_t_list)
            clip_dir_item = max(clip_dir_list)
        else:
            gen_clip_t_item = sum(gen_clip_t_list) / len(gen_clip_t_list)
            clip_dir_item = sum(clip_dir_list) / len(clip_dir_list)
        
        gen_clip_t += gen_clip_t_item
        clip_dir += clip_dir_item
        gt_clip_t += 1 - spatial.distance.cosine(gt_features.view(gt_features.shape[1]),
                                                    gt_text_features.view(gt_text_features.shape[1]))

    return gen_clip_t / len(image_pairs), gt_clip_t / len(image_pairs), clip_dir / len(image_pairs)


########################### Data Loading ################################
def traverse_images(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[-1]
            if ext in [".jpg", ".jpeg", ".png"]:
                image_paths.append(os.path.join(root, file))
    return image_paths


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help=('Number of processes to use for data loading. '
                            'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='Device to use. Like cuda, cuda or cpu')
    parser.add_argument('--generated_path',
                        type=str,
                        help='Paths of generated images (folders)')
    parser.add_argument('--gt_path',
                        type=str,
                        help='Paths to the gt images (folders)')
    parser.add_argument('--caption_path',
                        type=str,
                        default="global_description.json",
                        help='the file path to store the global captions for text-image similarity calculation')
    parser.add_argument('--metric',
                        type=str,
                        default='clip-i,dino,clip-t',
                        # default='clip-i,clipscore',
                        # default='clip-t',
                        help='the metric to calculate (l1, l2, clip-i, dino, clip-t)')
    parser.add_argument('--save_path',
                        type=str,
                        default='results',
                        help='Path to save the results')

    args = parser.parse_args()
    args.metric = args.metric.split(',')
    # args.metric = ["warp_err"]

    for arg in vars(args):
        print(arg, getattr(args, arg))

    benchmark_path = "/home/mingdengc/sensei-fs/projects/NEdit-Benchmark"
    generated_path = "/home/mingdengc/sensei-fs/projects/instruct-pix2pix/NEdit-Benchmark-Ours-011-IG1.5-G7.5-50steps"
    meta_info_path = "/home/mingdengc/sensei-fs/projects/NEdit-Benchmark/instruction-captioned.json"
    with open(meta_info_path, "r") as f:
        meta_info = json.load(f)
    print(f"Total editings: {len(meta_info)}")

    args.save_path = generated_path + "_metrics.json"

    image_paths = traverse_images(benchmark_path)
    print(f"Total images: {len(image_paths)}")
    image_names = [os.path.basename(p) for p in image_paths]
    image_dict = {os.path.basename(p): p for p in image_paths}

    edited_image_paths = traverse_images(generated_path)
    if len(edited_image_paths) == 0:
        print(f"Generated images not found.")
        exit()

    image_pairs = []
    for i, inst in enumerate(meta_info):
        if image_dict.get(inst["img_name"], None) is None:
            print(f"Image {inst['img_name']} not found.")
            continue
        img_path = image_dict[inst["img_name"]]

        instruction = inst["instruction"]
        src_caption = inst.get("src_caption")
        tgt_caption = inst.get("tgt_caption")

        # search for the generated images
        rel_path_wo_ext = os.path.relpath(img_path, benchmark_path).split(".")[0]
        edited_img_path_prefix = os.path.join(generated_path, rel_path_wo_ext+f"-{instruction}")
        print("Edited image path prefix: ", edited_img_path_prefix)

        edited_img_path_filtered = [p for p in edited_image_paths if edited_img_path_prefix in p]
        if len(edited_img_path_filtered) == 0:
            print(f"Generated image {edited_img_path_prefix} not found.")
            continue
        image_pairs.append(
            [
                edited_img_path_filtered,
                img_path,
                tgt_caption,
                src_caption,
            ]
        )
    print(f"Total image pairs: {len(image_pairs)}")
    print("Image pairs: ", image_pairs[0])
    evaluated_metrics_dict = {}

    # Image qualtiy metrics
    if 'clip-i' in args.metric:
        model, transform = clip.load("ViT-B/32", device)
        # print("CLIP-I model loaded: ", model)
        clip_i_eval_score = eval_clip_i(image_pairs, model, transform)
        evaluated_metrics_dict['clip-i'] = clip_i_eval_score
        print(f"Final turn CLIP-I: {clip_i_eval_score}")
    if 'dino' in args.metric:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        model.eval()
        model.to(device)
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dino_i_eval_score = eval_clip_i(image_pairs, model, transform, metric='dino')
        evaluated_metrics_dict['dino'] = dino_i_eval_score
        print(f"DINO-I Score: {dino_i_eval_score}")

    if 'clip-t' in args.metric:
        model, transform = clip.load("ViT-B/32", device)
        clip_t_eval_score, clip_t_oracle_score, clip_dir_score = eval_clip_t(image_pairs, model, transform, score_type="max")
        print(f"Final turn CLIP-T: {clip_t_eval_score}")
        print(f"Final turn CLIP-T Oracle: {clip_t_oracle_score}")
        print(f"Final turn CLIP-T Direction: {clip_dir_score}")
        evaluated_metrics_dict['clip-t'] = clip_t_eval_score
        evaluated_metrics_dict['clip-t_oracle'] = clip_t_oracle_score
        evaluated_metrics_dict['clip-t_dir'] = clip_dir_score

    if 'warp_err' in args.metric:
        warp_err_score = eval_warp_err(image_pairs)
        evaluated_metrics_dict['warp_err'] = warp_err_score
        print(f"Final turn Warp Error: {warp_err_score}")

    print(evaluated_metrics_dict)

    with open(args.save_path, 'w') as f:
        json.dump(evaluated_metrics_dict, f, indent=4)
