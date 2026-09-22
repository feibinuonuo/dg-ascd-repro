import argparse
from argparse import Namespace
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image
import math

from ascd_utils_v3.ascd_utils_v3 import *
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

    # debugpy.listen(("0.0.0.0", 5678))
    # print("waitng for debugger attach ...")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()
    # print("debugger is attached!")

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if "tinyllava" in args.model_path:
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor = load_tinyllava(model_path,
                                                            attn_implementation="eager")
        model2modify = model.language_model
    else:
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,
                                                                                args.model_base,
                                                                                model_name,
                                                                                attn_implementation="eager")
        model2modify = model.model

    replace_denoise_attn(model2modify,
                         contrastive_attn_type=args.contrastive_attn_type,
                         contrastive_layer_ids=args.contrastive_layer_ids,
                         yaml_configs=(args.direct_steer_config, args.contrastive_config))
    
    # change sample function
    transformers.generation.utils.GenerationMixin._sample = _sample
    transformers.generation.utils.GenerationMixin._greedy_search = _greedy_search
    transformers.generation.utils.GenerationMixin._beam_search = _beam_search
    transformers.generation.utils.GenerationMixin.cd_config = args.contrastive_decoding_config

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if getattr(model, "hf_device_map", None):
        print(f"Model already dispatched with device_map: {model.hf_device_map}")
    else:
        model.to(device='cuda')

    # os.makedirs(args.save_data_dir, exist_ok=True)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

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
        
    device = "cuda:0"
    for line in tqdm(questions):
        torch.cuda.reset_peak_memory_stats(device)
        idx = line["question_id"]
        qs = line["text"]
        image_file = line["image"]

        if hasattr(model.config, "mm_use_im_start_end") and model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        # image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_sizes = image.size

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(device='cuda', non_blocking=True)

        # compute sys, image token length
        _, pos_in_batch = torch.where(input_ids==-200)
        sys_len = pos_in_batch[0].item()
        if "tinyllava" in str(type(model)):
            img_len = (model.vision_tower.config.image_size // model.vision_tower.config.patch_size)**2
            if model.config.vision_feature_select_strategy == "patch":
                img_len -= 1
        else:
            img_len = (model.model.vision_tower.config.image_size // model.model.vision_tower.config.patch_size)**2
        
        set_self_denoise_attn_attr(model2modify, args.contrastive_attn_type, {"sys_len": sys_len,
                                                                              "img_len": img_len})
        
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
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": line["text"],
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--greedy_decoding", action='store_true', default=False)
    parser.add_argument("--nucleus_sampling", action='store_true', default=False)
    parser.add_argument("--beam_search", action='store_true', default=False)

    parser.add_argument("--contrastive_attn_type", type=str, default="hall_attn_v1")
    parser.add_argument("--contrastive_layer_ids", type=str, default="all")

    parser.add_argument("--direct_steer_config", type=str, default='experiments_v3/assets/direct_steer_config.yaml')
    parser.add_argument("--contrastive_config", type=str, default='experiments_v3/assets/contrastive_config.yaml')
    parser.add_argument("--contrastive_decoding_config", type=str, default='experiments_v3/assets/contrastive_decoding.yaml')

    args = parser.parse_args()

    import yaml

    assert args.direct_steer_config and os.path.exists(args.direct_steer_config)
    with open(args.direct_steer_config, "r") as f:
        loaded_params = yaml.safe_load(f)
    args.direct_steer_config = Namespace(**loaded_params)

    if args.contrastive_config and os.path.exists(args.contrastive_config):
        with open(args.contrastive_config, "r") as f:
            loaded_params = yaml.safe_load(f)
        args.contrastive_config = Namespace(**loaded_params)
    else:
        args.contrastive_config = None
        print("The file for 2nd attn_steer_config is not loaded!")

    if args.contrastive_decoding_config and os.path.exists(args.contrastive_decoding_config):
        with open(args.contrastive_decoding_config, "r") as f:
            loaded_params = yaml.safe_load(f)
        args.contrastive_decoding_config = Namespace(**loaded_params)
    else:
        args.contrastive_decoding_config = None
        print("The config file for contrastive decoding is not loaded!")


    eval_model(args)
