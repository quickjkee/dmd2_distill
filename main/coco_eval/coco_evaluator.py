# Part of this code is modified from GigaGAN: https://github.com/mingukkang/GigaGAN
# The MIT License (MIT)
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np 
import shutil
from transformers import AutoProcessor, AutoModel
import torch 
import time 
import os 

resizer_collection = {"nearest": InterpolationMode.NEAREST,
                      "box": InterpolationMode.BOX,
                      "bilinear": InterpolationMode.BILINEAR,
                      "hamming": InterpolationMode.HAMMING,
                      "bicubic": InterpolationMode.BICUBIC,
                      "lanczos": InterpolationMode.LANCZOS}


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """ 
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


@torch.no_grad()
def compute_fid(fake_arr, gt_dir, device,
    resize_size=None, feature_extractor="inception", 
    patch_fid=False):
    from main.coco_eval.cleanfid import fid
    center_crop_trsf = CenterCropLongEdge()
    def resize_and_center_crop(image_np):
        image_pil = Image.fromarray(image_np) 
        if patch_fid:
            # if image_pil.size[0] != 1024 and image_pil.size[1] != 1024:
            #     image_pil = image_pil.resize([1024, 1024])

            # directly crop to the 299 x 299 patch expected by the inception network
            if image_pil.size[0] >= 299 and image_pil.size[1] >= 299:
                image_pil = transforms.functional.center_crop(image_pil, 299)
            # else:
            #     raise ValueError("Image is too small to crop to 299 x 299")
        else:
            image_pil = center_crop_trsf(image_pil)

            if resize_size is not None:
                image_pil = image_pil.resize((resize_size, resize_size),
                                            Image.LANCZOS)
        return np.array(image_pil)

    if feature_extractor == "inception":
        model_name = "inception_v3"
    elif feature_extractor == "clip":
        model_name = "clip_vit_b_32"
    else:
        raise ValueError(
            "Unrecognized feature extractor [%s]" % feature_extractor)
    # fid, fake_feats, real_feats = fid.compute_fid(
    fid = fid.compute_fid(
        None,
        gt_dir,
        model_name=model_name,
        custom_image_tranform=resize_and_center_crop,
        use_dataparallel=False,
        device=device,
        pred_arr=fake_arr
    )
    # return fid, fake_feats, real_feats 
    return fid 

def evaluate_model(args, device, all_images, patch_fid=False):
    fid = compute_fid(
        fake_arr=all_images,
        gt_dir=args.ref_dir,
        device=device,
        resize_size=256,
        feature_extractor="inception",
        patch_fid=patch_fid
    )

    return fid 


def tensor2pil(image: torch.Tensor):
    ''' output image : tensor to PIL
    '''
    if isinstance(image, list) or image.ndim == 4:
        return [tensor2pil(im) for im in image]

    assert image.ndim == 3
    output_image = Image.fromarray(((image + 1.0) * 127.5).clamp(
        0.0, 255.0).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy())
    return output_image

class CLIPScoreDataset(Dataset):
    def __init__(self, images, captions, transform, preprocessor) -> None:
        super().__init__()
        self.images = images 
        self.captions = captions 
        self.transform = transform
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        image_pil = self.transform(image)
        image_pil = self.preprocessor(image_pil)
        caption = self.captions[index]
        return image_pil, caption


@torch.no_grad()
def calc_pick_and_clip_scores(model, image_inputs, text_inputs, batch_size=50):
    assert len(image_inputs) == len(text_inputs)

    scores = torch.zeros(len(text_inputs))
    for i in range(0, len(text_inputs), batch_size):
        image_batch = image_inputs[i:i + batch_size]
        text_batch = text_inputs[i:i + batch_size]
        # embed
        with torch.cuda.amp.autocast():
            image_embs = model.get_image_features(image_batch)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.cuda.amp.autocast():
            text_embs = model.get_text_features(text_batch)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores[i:i + batch_size] = (text_embs * image_embs).sum(-1)  # model.logit_scale.exp() *
    return scores.cpu()


@torch.no_grad()
def compute_clip_score(
    images, prompts, args, device="cuda", how_many=30000):
    print("Computing CLIP score")
    clip_preprocessor = AutoProcessor.from_pretrained(args.clip_model_name_or_path)
    clip_model = AutoModel.from_pretrained(args.clip_model_name_or_path).eval().to(device)

    image_inputs = clip_preprocessor(
        images=images,
        return_tensors="pt",
    )['pixel_values'].to(device)

    text_inputs = clip_preprocessor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )['input_ids'].to(device)

    print(len(image_inputs), len(text_inputs))

    clip_score = calc_pick_and_clip_scores(clip_model, image_inputs, text_inputs).mean()

    return clip_score

@torch.no_grad()
def compute_image_reward(
    images, captions, device
):
    import ImageReward as RM
    from tqdm import tqdm 
    model = RM.load("ImageReward-v1.0", device=device)
    rewards = [] 
    for image, prompt in tqdm(zip(images, captions)):
        reward = model.score(prompt, Image.fromarray(image))
        rewards.append(reward)
    return np.mean(np.array(rewards))

@torch.no_grad()
def compute_diversity_score(
    lpips_loss_func, images, device
):
    # resize all image to 512 and convert to tensor 
    images = [Image.fromarray(image) for image in images]
    images = [image.resize((512, 512), Image.LANCZOS) for image in images]
    images = np.stack([np.array(image) for image in images], axis=0)
    images = torch.tensor(images).to(device).float() / 255.0
    images = images.permute(0, 3, 1, 2) 

    num_images = images.shape[0] 
    loss_list = []

    for i in range(num_images):
        for j in range(i+1, num_images):
            image1 = images[i].unsqueeze(0)
            image2 = images[j].unsqueeze(0)
            loss = lpips_loss_func(image1, image2)

            loss_list.append(loss.item())
    return np.mean(loss_list)
