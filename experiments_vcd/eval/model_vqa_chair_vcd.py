import argparse
from argparse import Namespace
import torch
import os, sys
import json
from tqdm import tqdm
import shortuuid
import debugpy
import random

import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from .vcd_utils import add_diffusion_noise
from .vcd_utils_phi import forward, generate, forward_phiforcausallm

from ascd_utils_v3.ascd_utils_v3 import *
from ascd_utils_v3.ascd_models_v3 import AttnSteerConfig
from ascd_utils_v3.contrastive_sample import _sample, _greedy_search, _beam_search

from tinyllava.utils_tinyllava import load_tinyllava

import transformers

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    if "tinyllava" in args.model_path:
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor = load_tinyllava(model_path,
                                                            attn_implementation="eager")
    else:
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,
                                                                                args.model_base,
                                                                                model_name,
                                                                                attn_implementation="eager")

    # change sample function
    transformers.generation.utils.GenerationMixin._sample = _sample
    transformers.generation.utils.GenerationMixin._greedy_search = _greedy_search
    transformers.generation.utils.GenerationMixin._beam_search = _beam_search
    transformers.generation.utils.GenerationMixin.cd_config = args.vcd_config

    # change for tinyllava-phi2
    if "tinyllava" in str(type(model)):
        model.__class__.forward = forward
        model.__class__.generate = generate
        model.language_model.__class__.forward = forward_phiforcausallm

    data_raw = json.load(open(os.path.join(os.path.expanduser(args.annotation_folder), 'captions_val2014.json'), "r"))

    random.seed(args.sample_seed)
    data = random.sample(data_raw['images'], args.sample_num)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    model.to(device='cuda')

    if args.greedy_decoding:
        num_beams = 1
        do_sampling = False
        top_p = None
        temperature = 0.0
    elif args.nucleus_sampling:
        num_beams = 1
        do_sampling = True
        top_p=args.top_p
        assert args.top_p < 1.0, "You are using nucleus sampling, but the top-p is not smaller than 1.0!"
        temperature = args.temperature
    elif args.beam_search:
        num_beams = args.num_beams
        assert args.num_beams > 1, "You are using beam search for decoding, but the number of beam is set to 1."
        do_sampling = False
        top_p = None
        temperature = 0.0
    else:
        raise ValueError("Select one decoding method from greedy_decoding, nucleus_sampling and beam_search!")
        

    for i, line in enumerate(tqdm(data)):
        file_name = line["file_name"]
        id = line["id"]

        image = Image.open(os.path.join(args.image_folder, file_name))
        image_sizes = [image.size]
        if image.layers == 1:
            image = Image.merge("RGB", (image, image, image))
        elif image.mode == "CMYK":
            image = image.convert("RGB")

        qs = "Please describe this image in detail."

        if hasattr(model.config, "mm_use_im_start_end") and model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor_cd = add_diffusion_noise(image_tensor, args.vcd_config.noise_stemp)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=do_sampling,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                images_cd=image_tensor_cd.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True))
            
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        ans_file.write(json.dumps({
                                "image_id": id,
                                "caption": outputs
                                }) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--annotation-folder", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_num", type=int, default=500)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--greedy_decoding", action='store_true', default=False)
    parser.add_argument("--nucleus_sampling", action='store_true', default=False)
    parser.add_argument("--beam_search", action='store_true', default=False)

    parser.add_argument("--vcd_config", type=str, default='experiments_vcd/assets/vcd_config.yaml')

    args = parser.parse_args()

    import yaml

    assert args.vcd_config and os.path.exists(args.vcd_config), f"The path {args.vcd_config} for vcd_config doesnot exist!"
    with open(args.vcd_config, "r") as f:
        loaded_params = yaml.safe_load(f)
    args.vcd_config = Namespace(**loaded_params)

    eval_model(args)
