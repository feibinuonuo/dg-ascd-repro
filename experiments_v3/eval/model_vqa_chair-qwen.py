import argparse
from argparse import Namespace
import torch
import os
import json
from tqdm import tqdm
import random
from pathlib import Path
from PIL import Image

import math

from ascd_utils_v3_qwen.ascd_utils_v3 import *
from ascd_utils_v3_qwen.contrastive_sample import _sample

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from ascd_detector_grounded import Owlv2ObjectRuntime, sha256 as detector_sha256
import transformers


def _refuse_overwrite(path):
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        raise FileExistsError(f"Refusing to overwrite nonempty output: {path}")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    if args.context_entropy_audit_file and not args.greedy_decoding:
        raise ValueError("--context-entropy-audit-file requires greedy decoding")
    if args.detector_grounded_audit_file and not args.greedy_decoding:
        raise ValueError("--detector-grounded-audit-file requires greedy decoding")
    if args.detector_grounded_audit_file and args.context_entropy_audit_file:
        raise ValueError("Detector-Grounded ASCD cannot be combined with context entropy")
    if args.detector_grounded_audit_file and args.disable_ascd:
        raise ValueError("Detector-Grounded ASCD requires fixed ASCD, not --disable-ascd")

    if args.soft_grounded_audit_file and not args.greedy_decoding:
        raise ValueError("--soft-grounded-audit-file requires greedy decoding")
    if args.soft_grounded_audit_file and (
        args.detector_grounded_audit_file or args.context_entropy_audit_file
    ):
        raise ValueError("Soft-Grounded ASCD cannot be combined with hard detector or context entropy")
    if args.soft_grounded_audit_file and args.disable_ascd:
        raise ValueError("Soft-Grounded ASCD requires fixed ASCD, not --disable-ascd")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto", attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model2modify = model.model

    if not args.disable_ascd:
        replace_denoise_attn(model2modify,
                             contrastive_attn_type=args.contrastive_attn_type,
                             contrastive_layer_ids=args.contrastive_layer_ids,
                             yaml_configs=(args.direct_steer_config, args.contrastive_config),
                             context_entropy_enabled=bool(args.context_entropy_audit_file))
        transformers.generation.utils.GenerationMixin._sample = _sample
        transformers.generation.utils.GenerationMixin.cd_config = args.contrastive_decoding_config
        if args.alpha_stats_file:
            args.contrastive_decoding_config.record_alpha_values = True
            args.contrastive_decoding_config.recorded_alpha_values = []
        if args.margin_stats_file:
            args.contrastive_decoding_config.record_margin_values = True
            args.contrastive_decoding_config.recorded_margin_values = []
        if args.context_entropy_audit_file:
            args.contrastive_decoding_config.context_entropy_enabled = True
            args.contrastive_decoding_config.context_entropy_top_k = 3
            args.contrastive_decoding_config.context_entropy_beta = 10.0
            args.contrastive_decoding_config.context_entropy_records = []
    elif args.alpha_stats_file or args.margin_stats_file:
        raise ValueError("alpha/margin stats require ASCD to be enabled")
    if args.detector_grounded_audit_file:
        detector_audit_path = os.path.expanduser(args.detector_grounded_audit_file)
        if os.path.exists(detector_audit_path):
            raise FileExistsError(f"Refusing to overwrite detector audit: {detector_audit_path}")
        policy_path = os.path.expanduser(args.detector_grounded_policy_file)
        with open(policy_path, "r", encoding="utf-8") as handle:
            detector_policy = json.load(handle)
        if (
            detector_policy.get("schema_version") != 1
            or detector_policy.get("method") != "Detector-Grounded ASCD"
            or detector_policy.get("status") != "frozen_zero_shot_transfer"
        ):
            raise ValueError("Qwen Detector-Grounded ASCD requires its frozen transfer policy")
        if detector_policy.get("target_parent") != "qwen25vl_cc594898_local_top64_fixed":
            raise ValueError("Detector transfer policy has the wrong Qwen ASCD parent")
        if detector_policy.get("runtime_detector") != "google/owlv2-base-patch16-ensemble":
            raise ValueError("Detector transfer policy has the wrong detector")
        source_policy = Path(detector_policy.get("source_policy_path", ""))
        source_hash = str(detector_policy.get("source_policy_sha256", ""))
        if not source_policy.is_file() or detector_sha256(source_policy) != source_hash:
            raise ValueError("Detector transfer policy source hash mismatch")
        config = args.contrastive_decoding_config
        if not bool(getattr(config, "if_cd", False)):
            raise ValueError("Detector-Grounded ASCD requires fixed ASCD, not no-CD")
        if float(getattr(config, "cd_alpha", float("nan"))) != 1.0 or bool(getattr(config, "adaptive_alpha", False)):
            raise ValueError("Detector-Grounded ASCD requires the frozen Qwen Fixed-ASCD decoder")
        if int(detector_policy.get("top_k", 0)) < 1 or "threshold" not in detector_policy:
            raise ValueError("Detector transfer policy is missing threshold or top_k")
        model_path = Path(os.path.expanduser(args.detector_grounded_model_path))
        if not model_path.is_dir():
            raise FileNotFoundError(f"Missing local OWLv2 checkpoint: {model_path}")
        expected_hashes = detector_policy.get("calibration_model_file_sha256")
        if not isinstance(expected_hashes, dict) or not expected_hashes:
            raise ValueError("Detector transfer policy is missing model hashes")
        for file_name, expected_hash in expected_hashes.items():
            candidate = model_path / str(file_name)
            if not candidate.is_file() or detector_sha256(candidate) != str(expected_hash):
                raise ValueError(f"OWLv2 checkpoint hash mismatch: {candidate}")
        config.detector_grounded_enabled = True
        config.detector_threshold = float(detector_policy["threshold"])
        config.detector_top_k = int(detector_policy["top_k"])
        config.detector_policy = detector_policy
        config.detector_tokenizer = processor.tokenizer
        config.detector_records = []

    if args.soft_grounded_audit_file:
        soft_audit_path = os.path.expanduser(args.soft_grounded_audit_file)
        if os.path.exists(soft_audit_path):
            raise FileExistsError(f"Refusing to overwrite soft audit: {soft_audit_path}")
        policy_path = os.path.expanduser(args.soft_grounded_policy_file)
        with open(policy_path, "r", encoding="utf-8") as handle:
            soft_policy = json.load(handle)
        if (
            soft_policy.get("schema_version") != 1
            or soft_policy.get("method") != "Calibrated Soft-Grounded ASCD"
            or soft_policy.get("status") != "frozen"
            or soft_policy.get("runtime_detector") != "google/owlv2-base-patch16-ensemble"
        ):
            raise ValueError("Qwen Soft-Grounded ASCD requires the frozen visual-support policy")
        config = args.contrastive_decoding_config
        if not bool(getattr(config, "if_cd", False)):
            raise ValueError("Soft-Grounded ASCD requires fixed ASCD, not no-CD")
        if float(getattr(config, "cd_alpha", float("nan"))) != 1.0 or bool(getattr(config, "adaptive_alpha", False)):
            raise ValueError("Soft-Grounded ASCD requires the frozen Qwen Fixed-ASCD decoder")
        if int(soft_policy.get("top_k", 0)) < 1:
            raise ValueError("Soft policy has invalid top_k")
        model_path = Path(os.path.expanduser(args.soft_grounded_model_path))
        expected_hashes = soft_policy.get("calibration_model_file_sha256")
        if not model_path.is_dir() or not isinstance(expected_hashes, dict) or not expected_hashes:
            raise ValueError("Soft policy or OWLv2 checkpoint is incomplete")
        for file_name, expected_hash in expected_hashes.items():
            candidate = model_path / str(file_name)
            if not candidate.is_file() or detector_sha256(candidate) != str(expected_hash):
                raise ValueError(f"OWLv2 checkpoint hash mismatch: {candidate}")
        config.soft_grounded_enabled = True
        config.soft_grounded_policy = soft_policy
        config.soft_grounded_tokenizer = processor.tokenizer
        config.soft_grounded_records = []
    if args.image_manifest:
        manifest = json.load(open(args.image_manifest, "r"))
        if args.manifest_start_index < 0:
            raise ValueError("manifest_start_index must be nonnegative")
        if args.manifest_start_index + args.sample_num > len(manifest["images"]):
            raise ValueError("sample_num exceeds frozen image manifest")
        data = manifest["images"][args.manifest_start_index: args.manifest_start_index + args.sample_num]
    else:
        data_raw = json.load(open(os.path.join(os.path.expanduser(args.annotation_folder), 'captions_val2014.json'), "r"))
        random.seed(args.sample_seed)
        data = random.sample(data_raw['images'], args.sample_num)
    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    _refuse_overwrite(answers_file)
    _refuse_overwrite(args.alpha_stats_file)
    _refuse_overwrite(args.margin_stats_file)
    _refuse_overwrite(args.context_entropy_audit_file)
    _refuse_overwrite(args.soft_grounded_audit_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    model.to(device='cuda')
    if args.detector_grounded_audit_file:
        config = args.contrastive_decoding_config
        config.detector_runtime = Owlv2ObjectRuntime(
            args.detector_grounded_model_path, device="cuda"
        )
        print("Qwen Detector-Grounded ASCD: loaded frozen OWLv2 zero-shot transfer policy")
    if args.soft_grounded_audit_file:
        config = args.contrastive_decoding_config
        config.soft_grounded_runtime = Owlv2ObjectRuntime(
            args.soft_grounded_model_path, device="cuda"
        )
        print("Qwen Soft-Grounded ASCD: loaded frozen visual-support policy")

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
        if args.margin_stats_file:
            args.contrastive_decoding_config.margin_image_id = int(id)
            args.contrastive_decoding_config.margin_step = 0
        if args.detector_grounded_audit_file or args.soft_grounded_audit_file:
            image_path = os.path.join(args.image_folder, file_name)
            with Image.open(image_path) as opened_image:
                detector_image = opened_image.convert("RGB")
            config = args.contrastive_decoding_config
            if args.detector_grounded_audit_file:
                support_scores = config.detector_runtime.set_image(detector_image)
                config.detector_image_id = int(id)
                config.detector_generated_token_ids = []
                config.detector_events = []
                config.detector_decoding_steps = 0
                config.detector_object_candidates = 0
                config.detector_masked_candidates = 0
                config.detector_selection_changes = 0
                config.detector_no_finite_protections = 0
                config.detector_support_score_min = float(min(support_scores.values()))
                config.detector_support_score_max = float(max(support_scores.values()))
            else:
                support_scores = config.soft_grounded_runtime.set_image(detector_image)
                config.soft_grounded_image_id = int(id)
                config.soft_grounded_generated_token_ids = []
                config.soft_grounded_events = []
                config.soft_grounded_decoding_steps = 0
                config.soft_grounded_object_candidates = 0
                config.soft_grounded_penalized_candidates = 0
                config.soft_grounded_selection_changes = 0
                config.soft_grounded_no_finite_protections = 0
                config.soft_grounded_support_score_min = float(min(support_scores.values()))
                config.soft_grounded_support_score_max = float(max(support_scores.values()))

        qs = "Please describe this image in detail."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{os.path.join(args.image_folder, file_name)}",
                    },
                    {"type": "text", "text": f"{qs}"},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        mask = inputs['input_ids'] == 151655
        counts = mask.sum(dim=1)
        first_positions = torch.argmax(mask.int(), dim=1)
        sys_len = first_positions[0].item()
        img_len = counts[0].item()
        
        set_self_denoise_attn_attr(model2modify, args.contrastive_attn_type, {"sys_len": sys_len,
                                                                              "img_len": img_len})
        if args.context_entropy_audit_file:
            args.contrastive_decoding_config.context_entropy_image_start = int(sys_len)
            args.contrastive_decoding_config.context_entropy_image_length = int(img_len)
            args.contrastive_decoding_config.context_entropy_instruction_end = int(inputs.input_ids.shape[-1])
            args.contrastive_decoding_config.context_entropy_image_id = int(id)
            args.contrastive_decoding_config.context_entropy_step = 0
        
        with torch.inference_mode():
            generated_ids = model.generate(**inputs,
                                            use_cache=False,
                                            max_new_tokens=args.max_new_tokens,
                                            num_beams=num_beams,
                                            do_sample=do_sampling,
                                            top_p=top_p,
                                            temperature=temperature,)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        if args.detector_grounded_audit_file:
            config = args.contrastive_decoding_config
            generated = list(config.detector_generated_token_ids)
            eos_value = model.generation_config.eos_token_id
            eos_ids = {int(eos_value)} if isinstance(eos_value, int) else {int(value) for value in eos_value}
            terminated_by_eos = bool(generated and generated[-1] in eos_ids)
            config.detector_records.append({
                "image_id": int(id),
                "parent": "qwen25vl_cc594898_local_top64_fixed",
                "threshold": float(config.detector_threshold),
                "top_k": int(config.detector_top_k),
                "generated_tokens": len(generated),
                "terminated_by_eos": terminated_by_eos,
                "hit_max_new_tokens": len(generated) >= int(args.max_new_tokens) and not terminated_by_eos,
                "decoding_steps": int(config.detector_decoding_steps),
                "object_candidates_considered": int(config.detector_object_candidates),
                "masked_candidates": int(config.detector_masked_candidates),
                "selection_changes": int(config.detector_selection_changes),
                "no_finite_protections": int(config.detector_no_finite_protections),
                "support_score_min": float(config.detector_support_score_min),
                "support_score_max": float(config.detector_support_score_max),
                "events": config.detector_events,
            })

        if args.soft_grounded_audit_file:
            config = args.contrastive_decoding_config
            generated = list(config.soft_grounded_generated_token_ids)
            eos_value = model.generation_config.eos_token_id
            eos_ids = {int(eos_value)} if isinstance(eos_value, int) else {int(value) for value in eos_value}
            terminated_by_eos = bool(generated and generated[-1] in eos_ids)
            config.soft_grounded_records.append({
                "image_id": int(id),
                "parent": "qwen25vl_cc594898_local_top64_fixed",
                "top_k": int(config.soft_grounded_policy["top_k"]),
                "generated_tokens": len(generated),
                "terminated_by_eos": terminated_by_eos,
                "hit_max_new_tokens": len(generated) >= int(args.max_new_tokens) and not terminated_by_eos,
                "decoding_steps": int(config.soft_grounded_decoding_steps),
                "object_candidates_considered": int(config.soft_grounded_object_candidates),
                "soft_penalized_candidates": int(config.soft_grounded_penalized_candidates),
                "hard_masked_candidates": 0,
                "selection_changes": int(config.soft_grounded_selection_changes),
                "no_finite_protections": int(config.soft_grounded_no_finite_protections),
                "support_score_min": float(config.soft_grounded_support_score_min),
                "support_score_max": float(config.soft_grounded_support_score_max),
                "events": config.soft_grounded_events,
            })
        ans_file.write(json.dumps({
                                "image_id": id,
                                "caption": output_text[0]
                                }) + "\n")
        ans_file.flush()

    ans_file.close()

    if args.alpha_stats_file:
        values = args.contrastive_decoding_config.recorded_alpha_values
        if not values:
            raise RuntimeError("No adaptive alpha values were recorded")
        import statistics
        payload = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "values": values,
        }
        os.makedirs(os.path.dirname(args.alpha_stats_file), exist_ok=True)
        with open(args.alpha_stats_file, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    if args.margin_stats_file:
        values = args.contrastive_decoding_config.recorded_margin_values
        if not values:
            raise RuntimeError("No adaptive margin values were recorded")
        os.makedirs(os.path.dirname(args.margin_stats_file), exist_ok=True)
        with open(args.margin_stats_file, "x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "records": values}, handle, indent=2)
            handle.write("\n")
    if args.context_entropy_audit_file:
        records = args.contrastive_decoding_config.context_entropy_records
        if not records:
            raise RuntimeError("No context entropy records were produced")
        os.makedirs(os.path.dirname(args.context_entropy_audit_file), exist_ok=True)
        with open(args.context_entropy_audit_file, "x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "records": records}, handle, indent=2)
            handle.write("\n")
    if args.detector_grounded_audit_file:
        records = args.contrastive_decoding_config.detector_records
        if len(records) != len(data):
            raise RuntimeError(f"Detector-Grounded ASCD audit mismatch: records={len(records)} images={len(data)}")
        detector_path = os.path.expanduser(args.detector_grounded_audit_file)
        os.makedirs(os.path.dirname(detector_path) or ".", exist_ok=True)
        with open(detector_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "method": "Detector-Grounded ASCD",
                "target_backbone": "Qwen2.5-VL-7B-Instruct",
                "detector_model_path": os.path.abspath(args.detector_grounded_model_path),
                "policy": args.contrastive_decoding_config.detector_policy,
                "records": records,
            }, handle, indent=2)

    if args.soft_grounded_audit_file:
        records = args.contrastive_decoding_config.soft_grounded_records
        if len(records) != len(data):
            raise RuntimeError(f"Soft-Grounded ASCD audit mismatch: records={len(records)} images={len(data)}")
        soft_path = os.path.expanduser(args.soft_grounded_audit_file)
        os.makedirs(os.path.dirname(soft_path) or ".", exist_ok=True)
        with open(soft_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "method": "Calibrated Soft-Grounded ASCD",
                "target_backbone": "Qwen2.5-VL-7B-Instruct",
                "detector_model_path": os.path.abspath(args.soft_grounded_model_path),
                "policy": args.contrastive_decoding_config.soft_grounded_policy,
                "records": records,
            }, handle, indent=2)
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
    parser.add_argument("--image-manifest", type=str, default="")
    parser.add_argument("--manifest-start-index", type=int, default=0)
    parser.add_argument("--detector-grounded-audit-file", type=str, default="")
    parser.add_argument("--detector-grounded-policy-file", type=str, default="")
    parser.add_argument("--detector-grounded-model-path", type=str, default="/root/projects/ASCD-main/models/owlv2-base-patch16-ensemble")
    parser.add_argument("--soft-grounded-audit-file", type=str, default="")
    parser.add_argument("--soft-grounded-policy-file", type=str, default="")
    parser.add_argument("--soft-grounded-model-path", type=str, default="/root/projects/ASCD-main/models/owlv2-base-patch16-ensemble")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--greedy_decoding", action='store_true', default=False)
    parser.add_argument("--nucleus_sampling", action='store_true', default=False)
    parser.add_argument("--beam_search", action='store_true', default=False)

    parser.add_argument("--contrastive_attn_type", type=str, default="hall_attn_v1")

    parser.add_argument("--contrastive_layer_ids", type=str, default="all")
    parser.add_argument("--disable-ascd", action="store_true", default=False)
    parser.add_argument("--alpha-stats-file", type=str, default="")
    parser.add_argument("--margin-stats-file", type=str, default="")
    parser.add_argument("--context-entropy-audit-file", type=str, default="")
    
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
