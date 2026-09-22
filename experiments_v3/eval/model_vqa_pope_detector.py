"""POPE adapter for the already-frozen Detector-Grounded ASCD policy.

Without --detector-grounded-audit-file this follows the existing
experiments_v3.eval.model_vqa_loader Fixed-ASCD generation path.  The optional
detector branch is deliberately isolated from that historical entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from argparse import Namespace
from pathlib import Path

import shortuuid
import torch
import transformers
from PIL import Image
from tqdm import tqdm

from ascd_detector_grounded import Owlv2ObjectRuntime, sha256 as detector_sha256
from ascd_utils_v3.ascd_utils_v3 import replace_denoise_attn, set_self_denoise_attn_attr
from ascd_utils_v3.contrastive_sample import _beam_search, _greedy_search, _sample
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

if os.environ.get("ASCD_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    print("ASCD_DISABLE_CUDNN=1: disabled cuDNN for the OWLv2 visual forward.")


def split_list(items, chunks):
    chunk_size = math.ceil(len(items) / chunks)
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


def get_chunk(items, chunks, index):
    return split_list(items, chunks)[index]


def image_id_from_name(image_name: str) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(image_name).name)
    if match is None:
        raise ValueError(f"Cannot parse COCO image id from {image_name!r}")
    return int(match.group(1))


def validate_detector_policy(policy_path: str, model_path: str) -> dict:
    with open(os.path.expanduser(policy_path), "r", encoding="utf-8") as handle:
        policy = json.load(handle)
    if (
        policy.get("schema_version") != 1
        or policy.get("method") != "Detector-Grounded ASCD"
        or policy.get("status") != "frozen"
        or policy.get("parent") != "fixed_ascd_pos0625"
        or policy.get("runtime_detector") != "google/owlv2-base-patch16-ensemble"
    ):
        raise ValueError("POPE adapter requires the frozen Detector-Grounded ASCD policy")
    if int(policy.get("top_k", 0)) < 1 or "threshold" not in policy:
        raise ValueError("Detector policy is missing threshold or top_k")
    checkpoint = Path(os.path.expanduser(model_path))
    expected_hashes = policy.get("calibration_model_file_sha256")
    if not checkpoint.is_dir() or not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError("Detector checkpoint or policy hashes are incomplete")
    for file_name, expected_hash in expected_hashes.items():
        candidate = checkpoint / str(file_name)
        if not candidate.is_file() or detector_sha256(candidate) != str(expected_hash):
            raise ValueError(f"OWLv2 checkpoint hash mismatch: {candidate}")
    return policy


def detector_record(config, *, image_id: int, question_id: int, max_new_tokens: int, model) -> dict:
    generated = list(config.detector_generated_token_ids)
    eos_value = model.generation_config.eos_token_id
    eos_ids = {int(eos_value)} if isinstance(eos_value, int) else {int(value) for value in eos_value}
    terminated_by_eos = bool(generated and generated[-1] in eos_ids)
    return {
        "image_id": int(image_id),
        "question_id": int(question_id),
        "parent": "fixed_ascd_pos0625",
        "threshold": float(config.detector_threshold),
        "top_k": int(config.detector_top_k),
        "generated_tokens": len(generated),
        "terminated_by_eos": terminated_by_eos,
        "hit_max_new_tokens": len(generated) >= int(max_new_tokens) and not terminated_by_eos,
        "decoding_steps": int(config.detector_decoding_steps),
        "object_candidates_considered": int(config.detector_object_candidates),
        "masked_candidates": int(config.detector_masked_candidates),
        "selection_changes": int(config.detector_selection_changes),
        "no_finite_protections": int(config.detector_no_finite_protections),
        "support_score_min": float(config.detector_support_score_min),
        "support_score_max": float(config.detector_support_score_max),
        "image_support_cache_hit": bool(config.detector_image_support_cache_hit),
        "events": config.detector_events,
    }


def configure_detector(args, tokenizer) -> dict | None:
    if not args.detector_grounded_audit_file:
        return None
    if not args.greedy_decoding:
        raise ValueError("Detector-Grounded ASCD POPE evaluation requires --greedy-decoding")
    audit_path = Path(os.path.expanduser(args.detector_grounded_audit_file))
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite detector audit: {audit_path}")
    policy = validate_detector_policy(args.detector_grounded_policy_file, args.detector_grounded_model_path)
    config = args.contrastive_decoding_config
    if not bool(getattr(config, "if_cd", False)):
        raise ValueError("Detector-Grounded ASCD requires Fixed ASCD, not no-CD")
    config.detector_grounded_enabled = True
    config.detector_threshold = float(policy["threshold"])
    config.detector_top_k = int(policy["top_k"])
    config.detector_policy = policy
    config.detector_tokenizer = tokenizer
    config.detector_records = []
    return policy


def evaluate(args) -> None:
    if args.question_limit < 0:
        raise ValueError("--question-limit must be nonnegative")
    if args.greedy_decoding:
        do_sample, temperature, top_p, num_beams = False, 0.0, None, 1
    else:
        raise ValueError("This frozen POPE adapter supports only --greedy-decoding")

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name, attn_implementation="eager"
    )
    model2modify = model.model
    replace_denoise_attn(
        model2modify,
        contrastive_attn_type=args.contrastive_attn_type,
        contrastive_layer_ids=args.contrastive_layer_ids,
        yaml_configs=(args.direct_steer_config, args.contrastive_config),
    )
    transformers.generation.utils.GenerationMixin._sample = _sample
    transformers.generation.utils.GenerationMixin._greedy_search = _greedy_search
    transformers.generation.utils.GenerationMixin._beam_search = _beam_search
    transformers.generation.utils.GenerationMixin.cd_config = args.contrastive_decoding_config

    policy = configure_detector(args, tokenizer)
    questions = [json.loads(line) for line in open(os.path.expanduser(args.question_file), "r", encoding="utf-8")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.question_limit:
        questions = questions[:args.question_limit]
    if not questions:
        raise ValueError("No POPE questions selected")

    answers_path = Path(os.path.expanduser(args.answers_file))
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    if answers_path.exists():
        raise FileExistsError(f"Refusing to overwrite answer file: {answers_path}")
    if getattr(model, "hf_device_map", None):
        print(f"Model already dispatched with device_map: {model.hf_device_map}")
    else:
        model.to(device="cuda")

    config = args.contrastive_decoding_config
    image_support_cache: dict[str, dict[str, float]] = {}
    if policy is not None:
        config.detector_runtime = Owlv2ObjectRuntime(args.detector_grounded_model_path, device="cuda")
        print(
            "Detector-Grounded ASCD POPE adapter: loaded frozen OWLv2; "
            f"threshold={config.detector_threshold:.8f}; top_k={config.detector_top_k}"
        )

    with answers_path.open("x", encoding="utf-8") as answer_file:
        for question in tqdm(questions):
            question_id = int(question["question_id"])
            image_file = str(question["image"])
            image_id = image_id_from_name(image_file)
            question_text = str(question["text"])
            user_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question_text if getattr(model.config, "mm_use_im_start_end", False) else DEFAULT_IMAGE_TOKEN + "\n" + question_text
            conversation = conv_templates[args.conv_mode].copy()
            conversation.append_message(conversation.roles[0], user_text)
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()
            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image_size = image.size
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device="cuda", non_blocking=True)
            _, positions = torch.where(input_ids == -200)
            sys_len = positions[0].item()
            img_len = (model.model.vision_tower.config.image_size // model.model.vision_tower.config.patch_size) ** 2
            set_self_denoise_attn_attr(model2modify, args.contrastive_attn_type, {"sys_len": sys_len, "img_len": img_len})

            if policy is not None:
                cache_hit = image_file in image_support_cache
                if cache_hit:
                    support_scores = image_support_cache[image_file]
                    config.detector_runtime.scores = dict(support_scores)
                else:
                    support_scores = config.detector_runtime.set_image(image)
                    image_support_cache[image_file] = dict(support_scores)
                config.detector_image_id = image_id
                config.detector_generated_token_ids = []
                config.detector_events = []
                config.detector_decoding_steps = 0
                config.detector_object_candidates = 0
                config.detector_masked_candidates = 0
                config.detector_selection_changes = 0
                config.detector_no_finite_protections = 0
                config.detector_support_score_min = float(min(support_scores.values()))
                config.detector_support_score_max = float(max(support_scores.values()))
                config.detector_image_support_cache_hit = cache_hit

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device="cuda", non_blocking=True),
                    image_sizes=image_size,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            answer_file.write(json.dumps({
                "question_id": question_id,
                "prompt": question_text,
                "text": output_text,
                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
                "metadata": {"pope_detector_adapter": True},
            }) + "\n")
            if policy is not None:
                config.detector_records.append(detector_record(
                    config,
                    image_id=image_id,
                    question_id=question_id,
                    max_new_tokens=args.max_new_tokens,
                    model=model,
                ))

    if policy is not None:
        records = config.detector_records
        if len(records) != len(questions):
            raise RuntimeError(f"Detector audit mismatch: records={len(records)} questions={len(questions)}")
        audit_path = Path(os.path.expanduser(args.detector_grounded_audit_file))
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "method": "Detector-Grounded ASCD",
                "task": "POPE",
                "detector_model_path": os.path.abspath(args.detector_grounded_model_path),
                "policy": policy,
                "image_support_cache_entries": len(image_support_cache),
                "records": records,
            }, handle, indent=2)
            handle.write("\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model-path", required=True)
    result.add_argument("--model-base", default=None)
    result.add_argument("--image-folder", required=True)
    result.add_argument("--question-file", required=True)
    result.add_argument("--answers-file", required=True)
    result.add_argument("--conv-mode", default="llava_v1")
    result.add_argument("--num-chunks", type=int, default=1)
    result.add_argument("--chunk-idx", type=int, default=0)
    result.add_argument("--question-limit", type=int, default=0)
    result.add_argument("--max-new-tokens", type=int, default=64)
    result.add_argument("--greedy-decoding", action="store_true")
    result.add_argument("--contrastive-attn-type", default="hall_attn_v1")
    result.add_argument("--contrastive-layer-ids", default="all")
    result.add_argument("--direct-steer-config", required=True)
    result.add_argument("--contrastive-config", required=True)
    result.add_argument("--contrastive-decoding-config", required=True)
    result.add_argument("--detector-grounded-audit-file", default="")
    result.add_argument("--detector-grounded-policy-file", default="")
    result.add_argument(
        "--detector-grounded-model-path",
        default="/root/projects/ASCD-main/models/owlv2-base-patch16-ensemble",
    )
    return result


def load_configs(args) -> None:
    import yaml

    with open(args.direct_steer_config, "r", encoding="utf-8") as handle:
        args.direct_steer_config = Namespace(**yaml.safe_load(handle))
    with open(args.contrastive_config, "r", encoding="utf-8") as handle:
        args.contrastive_config = Namespace(**yaml.safe_load(handle))
    with open(args.contrastive_decoding_config, "r", encoding="utf-8") as handle:
        args.contrastive_decoding_config = Namespace(**yaml.safe_load(handle))


def self_test() -> None:
    assert image_id_from_name("COCO_val2014_000000310196.jpg") == 310196
    assert image_id_from_name("plain_42.png") == 42
    try:
        image_id_from_name("no-id.jpg")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid image name must fail")


if __name__ == "__main__":
    parsed = parser().parse_args()
    load_configs(parsed)
    evaluate(parsed)
