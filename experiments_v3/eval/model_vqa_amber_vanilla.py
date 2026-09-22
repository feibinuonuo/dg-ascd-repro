"""Official-format AMBER generative adapter for frozen greedy Vanilla LLaVA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from experiments_v3.eval.model_vqa_amber_detector import load_questions
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


if os.environ.get("ASCD_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    print("ASCD_DISABLE_CUDNN=1: disabled cuDNN for the Vanilla visual forward.")


def evaluate(args) -> None:
    if not args.greedy_decoding:
        raise ValueError("Frozen AMBER Vanilla evaluation requires --greedy-decoding")
    questions, _, _ = load_questions(
        args.query_file, args.manifest_file, args.num_chunks, args.chunk_idx, args.question_limit
    )
    answers_path = Path(os.path.expanduser(args.answers_file))
    if answers_path.exists():
        raise FileExistsError(f"Refusing to overwrite AMBER answers: {answers_path}")
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name, attn_implementation="eager"
    )
    if getattr(model, "hf_device_map", None):
        print(f"Model already dispatched with device_map: {model.hf_device_map}")
    else:
        model.to(device="cuda")
    responses = []
    for question in tqdm(questions):
        image_id = int(question["id"])
        query = str(question["query"])
        user_text = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query
            if getattr(model.config, "mm_use_im_start_end", False)
            else DEFAULT_IMAGE_TOKEN + "\n" + query
        )
        conversation = conv_templates[args.conv_mode].copy()
        conversation.append_message(conversation.roles[0], user_text)
        conversation.append_message(conversation.roles[1], None)
        image = Image.open(os.path.join(args.image_folder, str(question["image"]))).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        input_ids = tokenizer_image_token(
            conversation.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device="cuda", non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device="cuda", non_blocking=True),
                image_sizes=image.size,
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        responses.append({"id": image_id, "response": tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()})
    with answers_path.open("x", encoding="utf-8") as handle:
        json.dump(responses, handle, indent=2)
        handle.write("\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model-path", required=True)
    result.add_argument("--model-base", default=None)
    result.add_argument("--image-folder", required=True)
    result.add_argument("--query-file", required=True)
    result.add_argument("--manifest-file", default="")
    result.add_argument("--answers-file", required=True)
    result.add_argument("--conv-mode", default="llava_v1")
    result.add_argument("--num-chunks", type=int, default=1)
    result.add_argument("--chunk-idx", type=int, default=0)
    result.add_argument("--question-limit", type=int, default=0)
    result.add_argument("--max-new-tokens", type=int, default=512)
    result.add_argument("--greedy-decoding", action="store_true")
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
