"""Default-off AMBER generative adapter for the frozen Detector-Grounded ASCD.

The absence of ``--detector-grounded-audit-file`` is intentionally the plain
Fixed-ASCD path.  The detector branch validates the previously frozen policy
and writes one audit record per AMBER image.  This file never selects or tunes
any detector/ASCD parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from argparse import Namespace
from pathlib import Path

import torch
import transformers
from PIL import Image
from tqdm import tqdm

from ascd_detector_grounded import Owlv2ObjectRuntime
from ascd_utils_v3.ascd_utils_v3 import replace_denoise_attn, set_self_denoise_attn_attr
from ascd_utils_v3.contrastive_sample import _beam_search, _greedy_search, _sample
from experiments_v3.eval.model_vqa_pope_detector import detector_record, validate_detector_policy
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


if os.environ.get("ASCD_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    print("ASCD_DISABLE_CUDNN=1: disabled cuDNN for the OWLv2 visual forward.")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_list(items, chunks):
    chunk_size = math.ceil(len(items) / chunks)
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


def get_chunk(items, chunks, index):
    if chunks < 1 or not 0 <= index < chunks:
        raise ValueError("chunk index must be in [0, num_chunks)")
    return split_list(items, chunks)[index]


def validate_queries(rows: list[dict]) -> dict[int, dict]:
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or set(("id", "image", "query")) - set(row):
            raise ValueError("AMBER query rows must have id, image, and query")
        image_id = int(row["id"])
        expected_image = f"AMBER_{image_id}.jpg"
        if image_id < 1 or row["image"] != expected_image or not str(row["query"]).strip():
            raise ValueError(f"Invalid AMBER generative query row: {row}")
        if image_id in by_id:
            raise ValueError(f"Duplicate AMBER image id: {image_id}")
        by_id[image_id] = {"id": image_id, "image": expected_image, "query": str(row["query"])}
    if not by_id:
        raise ValueError("No AMBER generative queries")
    return by_id


def load_questions(query_file: str, manifest_file: str, chunks: int, chunk_idx: int, limit: int) -> tuple[list[dict], str, str | None]:
    query_path = Path(os.path.expanduser(query_file))
    if limit < 0:
        raise ValueError("--question-limit must be nonnegative")
    queries = validate_queries(json.loads(query_path.read_text(encoding="utf-8")))
    manifest_hash = None
    if manifest_file:
        manifest_path = Path(os.path.expanduser(manifest_file))
        manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_rows, list):
            raise ValueError("AMBER manifest must be a JSON list")
        selected_ids = []
        for row in manifest_rows:
            image_id = int(row["id"] if isinstance(row, dict) else row)
            if image_id not in queries or image_id in selected_ids:
                raise ValueError(f"Invalid or duplicate AMBER manifest id: {image_id}")
            selected_ids.append(image_id)
        manifest_hash = sha256(manifest_path)
    else:
        selected_ids = sorted(queries)
    questions = [queries[image_id] for image_id in selected_ids]
    questions = get_chunk(questions, chunks, chunk_idx)
    if limit:
        questions = questions[:limit]
    if not questions:
        raise ValueError("No AMBER questions selected")
    return questions, sha256(query_path), manifest_hash


def configure_detector(args, tokenizer) -> dict | None:
    if not args.detector_grounded_audit_file:
        return None
    if not args.greedy_decoding:
        raise ValueError("Detector-Grounded AMBER evaluation requires --greedy-decoding")
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
    if not args.greedy_decoding:
        raise ValueError("The frozen AMBER adapter supports only --greedy-decoding")
    questions, query_hash, manifest_hash = load_questions(
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
    if getattr(model, "hf_device_map", None):
        print(f"Model already dispatched with device_map: {model.hf_device_map}")
    else:
        model.to(device="cuda")

    config = args.contrastive_decoding_config
    image_support_cache: dict[str, dict[str, float]] = {}
    if policy is not None:
        config.detector_runtime = Owlv2ObjectRuntime(args.detector_grounded_model_path, device="cuda")
        print(
            "Detector-Grounded AMBER adapter: loaded frozen OWLv2; "
            f"threshold={config.detector_threshold:.8f}; top_k={config.detector_top_k}"
        )

    response_rows = []
    for question in tqdm(questions):
        image_id = int(question["id"])
        image_file = str(question["image"])
        question_text = str(question["query"])
        user_text = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question_text
            if getattr(model.config, "mm_use_im_start_end", False)
            else DEFAULT_IMAGE_TOKEN + "\n" + question_text
        )
        conversation = conv_templates[args.conv_mode].copy()
        conversation.append_message(conversation.roles[0], user_text)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_size = [image.size]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(
            device="cuda", non_blocking=True
        )
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
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        response_rows.append({"id": image_id, "response": output_text})
        if policy is not None:
            config.detector_records.append(detector_record(
                config,
                image_id=image_id,
                question_id=image_id,
                max_new_tokens=args.max_new_tokens,
                model=model,
            ))

    with answers_path.open("x", encoding="utf-8") as handle:
        json.dump(response_rows, handle, indent=2)
        handle.write("\n")
    if policy is not None:
        if len(config.detector_records) != len(questions):
            raise RuntimeError("Detector audit record count does not match AMBER questions")
        audit_path = Path(os.path.expanduser(args.detector_grounded_audit_file))
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "method": "Detector-Grounded ASCD",
                "task": "AMBER-generative",
                "query_file_sha256": query_hash,
                "manifest_file_sha256": manifest_hash,
                "detector_model_path": os.path.abspath(args.detector_grounded_model_path),
                "policy": policy,
                "image_support_cache_entries": len(image_support_cache),
                "records": config.detector_records,
            }, handle, indent=2)
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
    rows = [
        {"id": 1, "image": "AMBER_1.jpg", "query": "Describe this image."},
        {"id": 2, "image": "AMBER_2.jpg", "query": "Describe this image."},
    ]
    assert list(validate_queries(rows)) == [1, 2]
    try:
        validate_queries([{ "id": 1, "image": "wrong.jpg", "query": "x" }])
    except ValueError:
        pass
    else:
        raise AssertionError("invalid AMBER image name must fail")


if __name__ == "__main__":
    parsed = parser().parse_args()
    load_configs(parsed)
    evaluate(parsed)
