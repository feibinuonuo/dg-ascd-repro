import argparse
from argparse import Namespace
import torch
import os
import json
from tqdm import tqdm
import random
from pathlib import Path

import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

from ascd_utils_v3.ascd_utils_v3 import *
from ascd_utils_v3.contrastive_sample import _sample, _greedy_search, _beam_search
from ascd_allpath import install_allpath_on_wrappers, write_configuration
from ascd_mfcd import configuration as mfcd_configuration
from ascd_mfcd import gaussian_high_pass_filter, gaussian_low_pass_filter
from ascd_inter import configuration as inter_configuration
from ascd_fuzzycd import configuration as fuzzycd_configuration, sharpen_images
from ascd_crops import configuration as crops_configuration
from ascd_cei import configuration as cei_configuration
from ascd_dive import configuration as dive_configuration
from ascd_selfaug import augment_image, configuration as selfaug_configuration
from ascd_vista import configuration as vista_configuration
from ascd_clearsight import configuration as clearsight_configuration, SOURCE_COMMIT as CLEARSIGHT_SOURCE_COMMIT
from ascd_verifier_constrained import CLIPVisualVerifier
from ascd_detector_grounded import Owlv2ObjectRuntime, sha256 as detector_sha256
from ascd_alias_guard import Owlv2AliasRuntime

from tinyllava.utils_tinyllava import load_tinyllava

import transformers

if os.environ.get("ASCD_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    print("ASCD_DISABLE_CUDNN=1: disabled cuDNN for a safer CLIP vision forward.")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image_ids(path):
    """Load image ids from a JSON list, CHAIR details JSON, or JSONL."""
    expanded_path = os.path.expanduser(path)
    if expanded_path.endswith(".jsonl"):
        with open(expanded_path, "r", encoding="utf-8") as handle:
            payload = [json.loads(line) for line in handle if line.strip()]
    else:
        with open(expanded_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

    if isinstance(payload, dict):
        if "sentences" in payload:
            payload = payload["sentences"]
        elif "images" in payload:
            payload = payload["images"]
        else:
            raise ValueError(
                f"Cannot find 'sentences' or 'images' in image-id file: {path}"
            )

    image_ids = set()
    for item in payload:
        if isinstance(item, dict):
            if "image_id" in item:
                image_ids.add(int(item["image_id"]))
            elif "id" in item:
                image_ids.add(int(item["id"]))
            else:
                raise ValueError(f"Image-id record lacks image_id/id in {path}: {item}")
        else:
            image_ids.add(int(item))
    return image_ids

def load_forced_token_plan(path):
    """Load and validate one forced-token intervention per image."""
    expanded_path = os.path.expanduser(path)
    records = {}
    with open(expanded_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {expanded_path}:{line_number}"
                ) from exc
            required = {
                "image_id",
                "step",
                "expected_prefix_token_ids",
                "expected_selected_token_id",
                "forced_token_id",
            }
            missing = required - set(record)
            if missing:
                raise ValueError(
                    f"Forced-token plan row {line_number} is missing {sorted(missing)}."
                )
            image_id = int(record["image_id"])
            if image_id in records:
                raise ValueError(f"Duplicate forced-token plan for image_id={image_id}.")
            step = int(record["step"])
            prefix = [int(token_id) for token_id in record["expected_prefix_token_ids"]]
            if step < 0 or len(prefix) != step:
                raise ValueError(
                    f"Invalid prefix length for image_id={image_id}: "
                    f"step={step} prefix={len(prefix)}."
                )
            record = dict(record)
            record["image_id"] = image_id
            record["step"] = step
            record["expected_prefix_token_ids"] = prefix
            record["expected_selected_token_id"] = int(
                record["expected_selected_token_id"]
            )
            record["forced_token_id"] = int(record["forced_token_id"])
            if min(
                record["expected_selected_token_id"], record["forced_token_id"]
            ) < 0:
                raise ValueError(f"Negative token id for image_id={image_id}.")
            records[image_id] = record
    if not records:
        raise ValueError(f"No forced-token plans found in {expanded_path}.")
    return records


def eval_model(args):
    active_released_methods = sum(
        mode != "off" for mode in (
            args.deco_mode, args.only_mode, args.mole_mode, args.sumgd_mode,
            args.allpath_mode,
            args.mfcd_mode,
            args.inter_mode,
            args.fuzzycd_mode,
            args.crops_mode,
            args.cei_mode,
            args.dive_mode,
            args.vhr_mode,
            args.selfaug_mode,
            args.vista_mode,
            args.clearsight_mode,
        )
    )
    if active_released_methods > 1:
        raise ValueError("Released-method frozen runs must use exactly one added method")
    if args.verifier_audit_file and active_released_methods:
        raise ValueError(
            "Verifier-Constrained ASCD is a standalone ASCD policy and cannot "
            "be combined with a released added method."
        )
    if args.verifier_audit_file and not args.greedy_decoding:
        raise ValueError("--verifier-audit-file requires greedy decoding")
    if args.verifier_observe_only and not args.verifier_audit_file:
        raise ValueError("--verifier-observe-only requires --verifier-audit-file")
    if args.verifier_audit_file and not args.verifier_calibration_file:
        raise ValueError("Verifier-Constrained ASCD requires a frozen calibration file")
    if args.detector_grounded_audit_file and active_released_methods:
        raise ValueError(
            "Detector-Grounded ASCD is a standalone ASCD policy and cannot "
            "be combined with a released added method."
        )
    if args.detector_grounded_audit_file and args.verifier_audit_file:
        raise ValueError("Detector-Grounded ASCD cannot be combined with Verifier-Constrained ASCD")
    if args.detector_grounded_audit_file and not args.greedy_decoding:
        raise ValueError("--detector-grounded-audit-file requires greedy decoding")
    if args.detector_grounded_audit_file and not args.detector_grounded_policy_file:
        raise ValueError("Detector-Grounded ASCD requires a frozen policy file")
    if args.detector_grounded_audit_file and (
        args.context_entropy_audit_file or args.forced_token_plan
    ):
        raise ValueError(
            "Detector-Grounded ASCD cannot be combined with context-entropy or forced-token interventions"
        )
    if args.soft_grounded_audit_file:
        if active_released_methods or args.verifier_audit_file or args.detector_grounded_audit_file or args.detector_comparative_audit_file:
            raise ValueError("Soft-Grounded ASCD is a standalone policy and cannot be combined with another intervention")
        if not args.greedy_decoding or not args.soft_grounded_policy_file:
            raise ValueError("Soft-Grounded ASCD requires greedy decoding and a frozen policy")
        if args.context_entropy_audit_file or args.forced_token_plan:
            raise ValueError("Soft-Grounded ASCD cannot combine context-entropy or forced-token interventions")
    if args.alias_guard_observe_audit_file:
        if (active_released_methods or args.verifier_audit_file or args.detector_grounded_audit_file
                or args.soft_grounded_audit_file or args.detector_comparative_audit_file):
            raise ValueError("Alias observation is a standalone Fixed-ASCD calibration mode")
        if not args.greedy_decoding or not args.alias_guard_hard_policy_file:
            raise ValueError("Alias observation requires greedy decoding and the frozen hard-detector policy")
        if args.context_entropy_audit_file or args.forced_token_plan:
            raise ValueError("Alias observation cannot combine another token intervention")
    if args.detector_comparative_audit_file and active_released_methods:
        raise ValueError(
            "Comparative Evidence Reversion ASCD is a standalone ASCD policy and cannot "
            "be combined with a released added method."
        )
    if args.detector_comparative_audit_file and (
        args.verifier_audit_file or args.detector_grounded_audit_file
    ):
        raise ValueError(
            "Comparative Evidence Reversion cannot be combined with verifier or detector-mask ASCD"
        )
    if args.detector_comparative_audit_file and not args.greedy_decoding:
        raise ValueError("--detector-comparative-audit-file requires greedy decoding")
    if args.detector_comparative_observe_only and not args.detector_comparative_audit_file:
        raise ValueError("--detector-comparative-observe-only requires its audit file")
    if args.detector_comparative_observe_only and args.detector_comparative_policy_file:
        raise ValueError("Observe-only comparative run must not load a frozen policy")
    if args.detector_comparative_audit_file and not args.detector_comparative_observe_only and not args.detector_comparative_policy_file:
        raise ValueError("Constrained comparative run requires a frozen policy file")
    if args.detector_comparative_audit_file and (
        args.context_entropy_audit_file or args.forced_token_plan
    ):
        raise ValueError(
            "Comparative Evidence Reversion cannot be combined with context-entropy or forced-token interventions"
        )
    if args.deco_audit_file and not args.greedy_decoding:
        raise ValueError("--deco-audit-file requires greedy decoding")
    if bool(args.deco_audit_file) != (args.deco_mode != "off"):
        raise ValueError("DeCo mode and --deco-audit-file must be enabled together")
    if args.only_audit_file and not args.greedy_decoding:
        raise ValueError("--only-audit-file requires greedy decoding")
    if bool(args.only_audit_file) != (args.only_mode != "off"):
        raise ValueError("ONLY mode and --only-audit-file must be enabled together")
    if args.mole_audit_file and not args.greedy_decoding:
        raise ValueError("--mole-audit-file requires greedy decoding")
    if bool(args.mole_audit_file) != (args.mole_mode != "off"):
        raise ValueError("MoLE mode and --mole-audit-file must be enabled together")
    if args.sumgd_audit_file and not args.greedy_decoding:
        raise ValueError("--sumgd-audit-file requires greedy decoding")
    if bool(args.sumgd_audit_file) != (args.sumgd_mode != "off"):
        raise ValueError("SumGD mode and --sumgd-audit-file must be enabled together")
    if args.allpath_audit_file and not args.greedy_decoding:
        raise ValueError("--allpath-audit-file requires greedy decoding")
    if bool(args.allpath_audit_file) != (args.allpath_mode != "off"):
        raise ValueError("AllPath mode and --allpath-audit-file must be enabled together")
    if args.mfcd_audit_file and not args.greedy_decoding:
        raise ValueError("--mfcd-audit-file requires greedy decoding")
    if bool(args.mfcd_audit_file) != (args.mfcd_mode != "off"):
        raise ValueError("MFCD mode and --mfcd-audit-file must be enabled together")
    if args.mfcd_mode != "off" and "tinyllava" in args.model_path.lower():
        raise ValueError("MFCD currently supports LLaVA-1.5 only")
    if args.inter_audit_file and not args.greedy_decoding:
        raise ValueError("--inter-audit-file requires greedy decoding")
    if bool(args.inter_audit_file) != (args.inter_mode != "off"):
        raise ValueError("INTER mode and --inter-audit-file must be enabled together")
    if args.inter_mode != "off" and "tinyllava" in args.model_path.lower():
        raise ValueError("INTER adaptation currently supports LLaVA-1.5 only")
    if args.fuzzycd_audit_file and not args.greedy_decoding:
        raise ValueError("--fuzzycd-audit-file requires greedy decoding")
    if bool(args.fuzzycd_audit_file) != (args.fuzzycd_mode != "off"):
        raise ValueError("FuzzyCD mode and --fuzzycd-audit-file must be enabled together")
    if args.fuzzycd_mode != "off" and "tinyllava" in args.model_path.lower():
        raise ValueError("FuzzyCD adaptation currently supports LLaVA-1.5 only")
    if args.fuzzycd_audit_file and not args.fuzzycd_calibration_file:
        raise ValueError("FuzzyCD requires --fuzzycd-calibration-file")
    if args.crops_audit_file and not args.greedy_decoding:
        raise ValueError("--crops-audit-file requires greedy decoding")
    if bool(args.crops_audit_file) != (args.crops_mode != "off"):
        raise ValueError("CRoPS mode and --crops-audit-file must be enabled together")
    if args.crops_mode != "off" and "tinyllava" in args.model_path.lower():
        raise ValueError("CRoPS adaptation currently supports LLaVA-1.5 only")
    if args.cei_audit_file and not args.greedy_decoding:
        raise ValueError("--cei-audit-file requires greedy decoding")
    if bool(args.cei_audit_file) != (args.cei_mode != "off"):
        raise ValueError("CEI mode and --cei-audit-file must be enabled together")
    if args.cei_mode != "off" and "tinyllava" in args.model_path.lower():
        raise ValueError("Static CEI adaptation currently supports LLaVA-1.5 only")
    if args.dive_audit_file and not args.greedy_decoding:
        raise ValueError("--dive-audit-file requires greedy decoding")
    if bool(args.dive_audit_file) != (args.dive_mode != "off"):
        raise ValueError("DiVE mode and --dive-audit-file must be enabled together")
    if args.dive_mode != "off" and "tinyllava" in args.model_path.lower():
        raise ValueError("DiVE adaptation currently supports LLaVA-1.5 only")
    if args.vhr_audit_file and not args.greedy_decoding:
        raise ValueError("--vhr-audit-file requires greedy decoding")
    if bool(args.vhr_audit_file) != (args.vhr_mode != "off"):
        raise ValueError("VHR mode and --vhr-audit-file must be enabled together")
    if args.vhr_mode != "off" and "tinyllava" in args.model_path.lower():
        raise ValueError("VHR currently supports LLaVA-1.5 only")
    if args.selfaug_audit_file and not args.greedy_decoding:
        raise ValueError("--selfaug-audit-file requires greedy decoding")
    if bool(args.selfaug_audit_file) != (args.selfaug_mode != "off"):
        raise ValueError("Self-Aug mode and --selfaug-audit-file must be enabled together")
    if args.selfaug_mode != "off" and "tinyllava" in args.model_path.lower():
        raise ValueError("Self-Aug currently supports LLaVA-1.5 only")
    if args.selfaug_mode != "off" and not args.selfaug_sas_file:
        raise ValueError("Self-Aug requires a frozen --selfaug-sas-file")
    if args.vista_audit_file and not args.greedy_decoding:
        raise ValueError("--vista-audit-file requires greedy decoding")
    if bool(args.vista_audit_file) != (args.vista_mode != "off"):
        raise ValueError("VISTA mode and --vista-audit-file must be enabled together")
    if args.vista_mode != "off" and "tinyllava" in args.model_path.lower():
        raise ValueError("VISTA currently supports LLaVA-1.5 only")
    if args.clearsight_audit_file and not args.greedy_decoding:
        raise ValueError("--clearsight-audit-file requires greedy decoding")
    if bool(args.clearsight_audit_file) != (args.clearsight_mode != "off"):
        raise ValueError("ClearSight mode and --clearsight-audit-file must be enabled together")
    if args.clearsight_mode != "off" and "tinyllava" in args.model_path.lower():
        raise ValueError("ClearSight VAF currently supports LLaVA-1.5 only")
    if args.context_entropy_audit_file and not args.greedy_decoding:
        raise ValueError("--context-entropy-audit-file requires greedy decoding")
    if args.diagnostics_file and not args.greedy_decoding:
        raise ValueError("--diagnostics-file currently supports --greedy_decoding only.")
    if args.diagnostics_file and not getattr(args.contrastive_decoding_config, "if_cd", False):
        raise ValueError("--diagnostics-file requires contrastive decoding with if_cd=True.")
    if args.margin_stats_file and not args.greedy_decoding:
        raise ValueError("--margin-stats-file currently supports greedy decoding only.")
    if args.margin_stats_file and not getattr(args.contrastive_decoding_config, "adaptive_alpha", False):
        raise ValueError("--margin-stats-file requires adaptive_alpha=True.")
    if args.neutral_directional_diagnostics and not args.diagnostics_file:
        raise ValueError("--neutral-directional-diagnostics requires --diagnostics-file.")
    if args.neutral_directional_diagnostics and "tinyllava" in args.model_path.lower():
        raise ValueError("Neutral directional diagnostics currently support LLaVA-1.5 only.")

    if args.forced_token_plan and not args.greedy_decoding:
        raise ValueError("--forced-token-plan currently supports --greedy_decoding only.")
    if args.forced_token_plan and not args.diagnostics_file:
        raise ValueError("--forced-token-plan requires --diagnostics-file.")
    if args.forced_token_plan and not args.forced_token_audit_file:
        raise ValueError("--forced-token-plan requires --forced-token-audit-file.")
    if args.forced_token_audit_file and not args.forced_token_plan:
        raise ValueError("--forced-token-audit-file requires --forced-token-plan.")
    if args.forced_token_plan and args.neutral_directional_diagnostics:
        raise ValueError("Forced-token audit and neutral diagnostics cannot run together.")
    if args.forced_token_plan and "tinyllava" in args.model_path.lower():
        raise ValueError("Forced-token audit currently supports LLaVA-1.5 only.")
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
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
                         yaml_configs=(args.direct_steer_config, args.contrastive_config),
                         neutral_diagnostics=args.neutral_directional_diagnostics,
                         context_entropy_enabled=bool(args.context_entropy_audit_file))
    install_allpath_on_wrappers(model2modify, args.allpath_mode != "off")
    
    # change sample function
    transformers.generation.utils.GenerationMixin._sample = _sample
    transformers.generation.utils.GenerationMixin._greedy_search = _greedy_search
    transformers.generation.utils.GenerationMixin._beam_search = _beam_search
    transformers.generation.utils.GenerationMixin.cd_config = args.contrastive_decoding_config

    data_raw = json.load(open(os.path.join(os.path.expanduser(args.annotation_folder), 'captions_val2014.json'), "r"))

    if args.image_manifest:
        manifest = json.load(open(os.path.expanduser(args.image_manifest), "r"))
        if args.manifest_start_index < 0:
            raise ValueError("manifest_start_index must be nonnegative")
        sampling_population = manifest["images"][args.manifest_start_index:]
    else:
        sampling_population = data_raw['images']
    if args.exclude_image_ids_file:
        excluded_image_ids = load_image_ids(args.exclude_image_ids_file)
        sampling_population = [
            image for image in sampling_population
            if int(image["id"]) not in excluded_image_ids
        ]
        print(
            f"Excluded {len(excluded_image_ids)} calibration image ids; "
            f"remaining population={len(sampling_population)}"
        )
    if args.sample_num > len(sampling_population):
        raise ValueError(
            f"sample_num={args.sample_num} exceeds available images="
            f"{len(sampling_population)} after exclusions."
        )

    if args.image_manifest:
        data = sampling_population[:args.sample_num]
    else:
        random.seed(args.sample_seed)
        data = random.sample(sampling_population, args.sample_num)
    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    if os.path.exists(answers_file) and os.path.getsize(answers_file) > 0:
        raise FileExistsError(
            f"Refusing to overwrite nonempty answer file: {answers_file}"
        )
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    if args.deco_audit_file:
        deco_path = os.path.expanduser(args.deco_audit_file)
        if os.path.exists(deco_path):
            raise FileExistsError(f"Refusing to overwrite DeCo audit: {deco_path}")
        config = args.contrastive_decoding_config
        config.deco_enabled = True
        config.deco_alpha = 0.6
        config.deco_threshold_top_p = 0.9
        config.deco_threshold_top_k = 20
        config.deco_early_exit_layers = list(range(20, 29))
        config.deco_records = []
    if args.only_audit_file:
        only_path = os.path.expanduser(args.only_audit_file)
        if os.path.exists(only_path):
            raise FileExistsError(f"Refusing to overwrite ONLY audit: {only_path}")
        config = args.contrastive_decoding_config
        config.only_enabled = True
        config.only_layer_index = 0
        config.only_positive_alpha = 3.0
        config.only_negative_alpha = 1.0
        config.only_beta = 0.1
        config.only_tvd_threshold = 0.25
        config.only_records = []
    if args.mole_audit_file:
        mole_path = os.path.expanduser(args.mole_audit_file)
        if os.path.exists(mole_path):
            raise FileExistsError(f"Refusing to overwrite MoLE audit: {mole_path}")
        config = args.contrastive_decoding_config
        if (args.mole_mode == "official") == bool(getattr(config, "if_cd", False)):
            raise ValueError("MoLE official requires no-CD; hybrid requires fixed ASCD")
        config.mole_enabled = True
        config.mole_top_n = 5
        config.mole_final_layers = 3
        config.mole_tmp = 50.0
        config.mole_w_exp = 0.2
        config.mole_records = []
    if args.sumgd_audit_file:
        sumgd_path = os.path.expanduser(args.sumgd_audit_file)
        if os.path.exists(sumgd_path):
            raise FileExistsError(f"Refusing to overwrite SumGD audit: {sumgd_path}")
        config = args.contrastive_decoding_config
        if (args.sumgd_mode == "official") == bool(getattr(config, "if_cd", False)):
            raise ValueError("SumGD official requires no-CD; hybrid requires fixed ASCD")
        config.sumgd_enabled = True
        config.sumgd_tokenizer = tokenizer
        config.sumgd_max_new_tokens = int(args.max_new_tokens)
        config.sumgd_max_summary_tokens = 128
        config.sumgd_records = []
    if args.allpath_audit_file:
        allpath_path = os.path.expanduser(args.allpath_audit_file)
        if os.path.exists(allpath_path):
            raise FileExistsError(f"Refusing to overwrite AllPath audit: {allpath_path}")
        config = args.contrastive_decoding_config
        if (args.allpath_mode == "official") == bool(getattr(config, "if_cd", False)):
            raise ValueError("AllPath official requires no-CD; hybrid requires fixed ASCD")
    if args.mfcd_audit_file:
        mfcd_path = os.path.expanduser(args.mfcd_audit_file)
        if os.path.exists(mfcd_path):
            raise FileExistsError(f"Refusing to overwrite MFCD audit: {mfcd_path}")
        config = args.contrastive_decoding_config
        if (args.mfcd_mode == "official") == bool(getattr(config, "if_cd", False)):
            raise ValueError("MFCD official requires no-CD; hybrid requires fixed ASCD")
        config.mfcd_enabled = True
        config.mfcd_high_alpha = 1.0
        config.mfcd_low_alpha = 1.0
        config.mfcd_beta = 0.3
        config.mfcd_max_new_tokens = int(args.max_new_tokens)
        config.mfcd_records = []
    if args.inter_audit_file:
        inter_path = os.path.expanduser(args.inter_audit_file)
        if os.path.exists(inter_path):
            raise FileExistsError(f"Refusing to overwrite INTER audit: {inter_path}")
        config = args.contrastive_decoding_config
        if (args.inter_mode == "official") == bool(getattr(config, "if_cd", False)):
            raise ValueError("INTER official-adaptation requires no-CD; hybrid requires fixed ASCD")
        config.inter_enabled = True
        config.inter_variance_threshold = 1.0
        config.inter_beta = 0.1
        config.inter_max_new_tokens = int(args.max_new_tokens)
        config.inter_records = []
    if args.fuzzycd_audit_file:
        fuzzycd_path = os.path.expanduser(args.fuzzycd_audit_file)
        if os.path.exists(fuzzycd_path):
            raise FileExistsError(f"Refusing to overwrite FuzzyCD audit: {fuzzycd_path}")
        config = args.contrastive_decoding_config
        if (args.fuzzycd_mode == "official") == bool(getattr(config, "if_cd", False)):
            raise ValueError("FuzzyCD official-adaptation requires no-CD; hybrid requires fixed ASCD")
        with open(os.path.expanduser(args.fuzzycd_calibration_file), "r", encoding="utf-8") as handle:
            calibration_payload = json.load(handle)
        if bool(calibration_payload.get("parent_if_cd")) != bool(getattr(config, "if_cd", False)):
            raise ValueError("FuzzyCD calibration parent does not match decoding parent")
        config.fuzzycd_enabled = True
        config.fuzzycd_calibration = {
            key: float(calibration_payload[key]) for key in ("mean", "std", "min", "max")
        }
        config.fuzzycd_beta = 0.1
        config.fuzzycd_max_new_tokens = int(args.max_new_tokens)
        config.fuzzycd_records = []
    if args.crops_audit_file:
        crops_path = os.path.expanduser(args.crops_audit_file)
        if os.path.exists(crops_path):
            raise FileExistsError(f"Refusing to overwrite CRoPS audit: {crops_path}")
        config = args.contrastive_decoding_config
        if (args.crops_mode == "official") == bool(getattr(config, "if_cd", False)):
            raise ValueError("CRoPS official requires no-CD; hybrid requires fixed ASCD")
        config.crops_enabled = True
        config.crops_lambda_lang_prior = 0.01
        config.crops_alpha_stat_bias = 1.0
        config.crops_beta_cutoff = 0.1
        config.crops_max_plausibility = 0.95
        config.crops_aggregate_layer = 2
        config.crops_visual_keep_fraction = 0.25
        config.crops_max_new_tokens = int(args.max_new_tokens)
        config.crops_records = []
    if args.cei_audit_file:
        cei_path = os.path.expanduser(args.cei_audit_file)
        if os.path.exists(cei_path):
            raise FileExistsError(f"Refusing to overwrite CEI audit: {cei_path}")
        config = args.contrastive_decoding_config
        if (args.cei_mode == "official") == bool(getattr(config, "if_cd", False)):
            raise ValueError("CEI official requires no-CD; hybrid requires fixed ASCD")
        config.cei_enabled = True
        config.cei_context_layer = -1
        config.cei_context_position = -1
        config.cei_injection_layer = 10
        config.cei_alpha = 0.1
        config.cei_max_new_tokens = int(args.max_new_tokens)
        config.cei_records = []
    if args.dive_audit_file:
        dive_path = os.path.expanduser(args.dive_audit_file)
        if os.path.exists(dive_path):
            raise FileExistsError(f"Refusing to overwrite DiVE audit: {dive_path}")
        config = args.contrastive_decoding_config
        if (args.dive_mode == "official") == bool(getattr(config, "if_cd", False)):
            raise ValueError("DiVE official adaptation requires no-CD; hybrid requires fixed ASCD")
        config.dive_enabled = True
        config.dive_exclusion_ratio = 0.05
        config.dive_gamma = 0.5
        config.dive_threshold = 0.85
        config.dive_epsilon = 1e-6
        config.dive_max_new_tokens = int(args.max_new_tokens)
        config.dive_records = []
    if args.vhr_audit_file:
        vhr_path = os.path.expanduser(args.vhr_audit_file)
        if os.path.exists(vhr_path):
            raise FileExistsError(f"Refusing to overwrite VHR audit: {vhr_path}")
        config = args.contrastive_decoding_config
        if bool(getattr(config, "if_cd", False)):
            raise ValueError("Official VHR-only requires no-CD")
        config.vhr_enabled = True
        config.vhr_augmentation_ratio = 2.0
        config.vhr_last_layers = 14
        config.vhr_include_layer_one = True
        config.vhr_outlier_filter = True
        config.vhr_records = []
    if args.selfaug_audit_file:
        selfaug_path = os.path.expanduser(args.selfaug_audit_file)
        if os.path.exists(selfaug_path):
            raise FileExistsError(f"Refusing to overwrite Self-Aug audit: {selfaug_path}")
        config = args.contrastive_decoding_config
        if bool(getattr(config, "if_cd", False)):
            raise ValueError("Official Self-Aug-only requires no-CD")
        with open(os.path.expanduser(args.selfaug_sas_file), "r", encoding="utf-8") as handle:
            sas_payload = json.load(handle)
        frozen = selfaug_configuration()
        config.selfaug_enabled = True
        config.selfaug_applied_aug = str(sas_payload.get("applied_aug", ""))
        config.selfaug_alpha = frozen["alpha"]
        config.selfaug_tau = frozen["tau"]
        config.selfaug_crop_ratio = frozen["crop_ratio"]
        config.selfaug_mask_ratio = frozen["mask_ratio"]
        config.selfaug_noise_step = frozen["noise_step"]
        config.selfaug_max_new_tokens = int(args.max_new_tokens)
        config.selfaug_sas_payload = sas_payload
        config.selfaug_records = []
        random.seed(args.sample_seed)
        np.random.seed(args.sample_seed)
        torch.manual_seed(args.sample_seed)
        torch.cuda.manual_seed_all(args.sample_seed)
    if args.vista_audit_file:
        vista_path = os.path.expanduser(args.vista_audit_file)
        if os.path.exists(vista_path):
            raise FileExistsError(f"Refusing to overwrite VISTA audit: {vista_path}")
        config = args.contrastive_decoding_config
        if bool(getattr(config, "if_cd", False)):
            raise ValueError("Official VISTA-only requires no-CD")
        frozen = vista_configuration()
        config.vista_enabled = True
        config.vista_vsv_lambda = frozen["vsv_lambda"]
        config.vista_sla_start_layer = frozen["sla_start_layer"]
        config.vista_sla_end_layer = frozen["sla_end_layer"]
        config.vista_sla_alpha = frozen["sla_alpha"]
        config.vista_max_new_tokens = int(args.max_new_tokens)
        config.vista_records = []
    if args.clearsight_audit_file:
        clearsight_path = os.path.expanduser(args.clearsight_audit_file)
        if os.path.exists(clearsight_path):
            raise FileExistsError(f"Refusing to overwrite ClearSight audit: {clearsight_path}")
        config = args.contrastive_decoding_config
        if bool(getattr(config, "if_cd", False)):
            raise ValueError("Official ClearSight VAF requires no-CD")
        config.clearsight_enabled = True
        config.clearsight_records = []
    if args.verifier_audit_file:
        verifier_path = os.path.expanduser(args.verifier_audit_file)
        if os.path.exists(verifier_path):
            raise FileExistsError(f"Refusing to overwrite verifier audit: {verifier_path}")
        calibration_path = os.path.expanduser(args.verifier_calibration_file)
        with open(calibration_path, "r", encoding="utf-8") as handle:
            verifier_calibration = json.load(handle)
        if verifier_calibration.get("schema_version") != 1:
            raise ValueError("Unsupported verifier calibration schema")
        if verifier_calibration.get("method") != "Verifier-Constrained ASCD":
            raise ValueError("Verifier calibration does not belong to Verifier-Constrained ASCD")
        if "threshold" not in verifier_calibration:
            raise ValueError("Verifier calibration has no frozen threshold")
        calibration_status = verifier_calibration.get("status")
        if args.verifier_observe_only:
            if calibration_status not in {"bootstrap_observe_only", "frozen"}:
                raise ValueError("Observe-only verifier run needs a bootstrap or frozen calibration")
        elif calibration_status != "frozen":
            raise ValueError("Constrained verifier run requires a frozen calibration")
        config = args.contrastive_decoding_config
        if not bool(getattr(config, "if_cd", False)):
            raise ValueError("Verifier-Constrained ASCD requires fixed ASCD, not no-CD")
        config.verifier_constrained_enabled = True
        config.verifier_observe_only = bool(args.verifier_observe_only)
        config.verifier_threshold = float(verifier_calibration["threshold"])
        config.verifier_calibration = verifier_calibration
        config.verifier_tokenizer = tokenizer
        config.verifier_max_new_tokens = int(args.max_new_tokens)
        config.verifier_records = []
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
            or detector_policy.get("status") != "frozen"
        ):
            raise ValueError("Detector-Grounded ASCD requires its frozen policy schema")
        if detector_policy.get("parent") != "fixed_ascd_pos0625":
            raise ValueError("Detector-Grounded ASCD policy has the wrong ASCD parent")
        if detector_policy.get("runtime_detector") != "google/owlv2-base-patch16-ensemble":
            raise ValueError("Detector-Grounded ASCD policy has the wrong detector")
        config = args.contrastive_decoding_config
        if not bool(getattr(config, "if_cd", False)):
            raise ValueError("Detector-Grounded ASCD requires fixed ASCD, not no-CD")
        if int(detector_policy.get("top_k", 0)) < 1 or "threshold" not in detector_policy:
            raise ValueError("Detector-Grounded ASCD policy is missing threshold or top_k")
        model_path = Path(os.path.expanduser(args.detector_grounded_model_path))
        if not model_path.is_dir():
            raise FileNotFoundError(f"Missing local OWLv2 checkpoint: {model_path}")
        expected_hashes = detector_policy.get("calibration_model_file_sha256")
        if not isinstance(expected_hashes, dict) or not expected_hashes:
            raise ValueError("Detector policy is missing calibration model hashes")
        for file_name, expected_hash in expected_hashes.items():
            candidate = model_path / str(file_name)
            if not candidate.is_file() or detector_sha256(candidate) != str(expected_hash):
                raise ValueError(f"OWLv2 checkpoint hash mismatch: {candidate}")
        config.detector_grounded_enabled = True
        config.detector_threshold = float(detector_policy["threshold"])
        config.detector_top_k = int(detector_policy["top_k"])
        config.detector_policy = detector_policy
        config.detector_tokenizer = tokenizer
        config.detector_records = []
    if args.soft_grounded_audit_file:
        soft_audit_path = os.path.expanduser(args.soft_grounded_audit_file)
        if os.path.exists(soft_audit_path):
            raise FileExistsError(f"Refusing to overwrite soft audit: {soft_audit_path}")
        with open(os.path.expanduser(args.soft_grounded_policy_file), "r", encoding="utf-8") as handle:
            soft_policy = json.load(handle)
        if soft_policy.get("schema_version") != 1 or soft_policy.get("method") != "Calibrated Soft-Grounded ASCD" or soft_policy.get("status") != "frozen":
            raise ValueError("Soft-Grounded ASCD requires its frozen policy schema")
        if soft_policy.get("parent") != "fixed_ascd_pos0625" or soft_policy.get("runtime_detector") != "google/owlv2-base-patch16-ensemble":
            raise ValueError("Soft-Grounded ASCD policy parent or detector mismatch")
        if int(soft_policy.get("top_k", 0)) < 1:
            raise ValueError("Soft-Grounded ASCD policy is missing top_k")
        model_path = Path(os.path.expanduser(args.soft_grounded_model_path))
        expected_hashes = soft_policy.get("calibration_model_file_sha256")
        if not model_path.is_dir() or not isinstance(expected_hashes, dict) or not expected_hashes:
            raise ValueError("Soft-Grounded ASCD policy/model is incomplete")
        for file_name, expected_hash in expected_hashes.items():
            candidate = model_path / str(file_name)
            if not candidate.is_file() or detector_sha256(candidate) != str(expected_hash):
                raise ValueError(f"OWLv2 checkpoint hash mismatch: {candidate}")
        config = args.contrastive_decoding_config
        if not bool(getattr(config, "if_cd", False)):
            raise ValueError("Soft-Grounded ASCD requires fixed ASCD, not no-CD")
        config.soft_grounded_enabled = True
        config.soft_grounded_top_k = int(soft_policy["top_k"])
        config.soft_grounded_probability_slope = float(soft_policy["probability_slope"])
        config.soft_grounded_probability_intercept = float(soft_policy["probability_intercept"])
        config.soft_grounded_probability_clip_min = float(soft_policy["probability_clip_min"])
        config.soft_grounded_probability_clip_max = float(soft_policy["probability_clip_max"])
        config.soft_grounded_policy = soft_policy
        config.soft_grounded_tokenizer = tokenizer
    if args.alias_guard_observe_audit_file:
        alias_audit_path = os.path.expanduser(args.alias_guard_observe_audit_file)
        if os.path.exists(alias_audit_path):
            raise FileExistsError(f"Refusing to overwrite alias observation audit: {alias_audit_path}")
        with open(os.path.expanduser(args.alias_guard_hard_policy_file), "r", encoding="utf-8") as handle:
            alias_policy = json.load(handle)
        if (alias_policy.get("schema_version") != 1 or alias_policy.get("method") != "Detector-Grounded ASCD"
                or alias_policy.get("status") != "frozen" or alias_policy.get("parent") != "fixed_ascd_pos0625"
                or alias_policy.get("runtime_detector") != "google/owlv2-base-patch16-ensemble"):
            raise ValueError("Alias observation requires the existing frozen Detector-Grounded ASCD policy")
        if int(alias_policy.get("top_k", 0)) < 1 or "threshold" not in alias_policy:
            raise ValueError("Alias observation hard policy is missing threshold or top_k")
        model_path = Path(os.path.expanduser(args.alias_guard_model_path))
        expected_hashes = alias_policy.get("calibration_model_file_sha256")
        if not model_path.is_dir() or not isinstance(expected_hashes, dict) or not expected_hashes:
            raise ValueError("Alias observation model/policy is incomplete")
        for file_name, expected_hash in expected_hashes.items():
            candidate = model_path / str(file_name)
            if not candidate.is_file() or detector_sha256(candidate) != str(expected_hash):
                raise ValueError(f"OWLv2 checkpoint hash mismatch: {candidate}")
        config = args.contrastive_decoding_config
        if not bool(getattr(config, "if_cd", False)):
            raise ValueError("Alias observation requires fixed ASCD, not no-CD")
        config.alias_guard_observe_enabled = True
        config.alias_guard_hard_threshold = float(alias_policy["threshold"])
        config.alias_guard_top_k = int(alias_policy["top_k"])
        config.alias_guard_policy = alias_policy
        config.alias_guard_tokenizer = tokenizer
        config.alias_guard_records = []
    if args.soft_grounded_audit_file:
        config.soft_grounded_records = []
    if args.detector_comparative_audit_file:
        comparative_audit_path = os.path.expanduser(args.detector_comparative_audit_file)
        if os.path.exists(comparative_audit_path):
            raise FileExistsError(
                f"Refusing to overwrite comparative detector audit: {comparative_audit_path}"
            )
        config = args.contrastive_decoding_config
        if not bool(getattr(config, "if_cd", False)):
            raise ValueError("Comparative Evidence Reversion requires fixed ASCD, not no-CD")
        model_path = Path(os.path.expanduser(args.detector_comparative_model_path))
        if not model_path.is_dir():
            raise FileNotFoundError(f"Missing local OWLv2 checkpoint: {model_path}")
        config.detector_comparative_enabled = True
        config.detector_comparative_observe_only = bool(
            args.detector_comparative_observe_only
        )
        config.detector_comparative_tokenizer = tokenizer
        config.detector_comparative_max_new_tokens = int(args.max_new_tokens)
        config.detector_comparative_records = []
        if args.detector_comparative_observe_only:
            config.detector_comparative_threshold = 0.0
            config.detector_comparative_policy = {
                "schema_version": 1,
                "method": "Object-Local Comparative Evidence Reversion ASCD",
                "status": "bootstrap_observe_only",
                "parent": "fixed_ascd_pos0625",
                "runtime_detector": "google/owlv2-base-patch16-ensemble",
            }
        else:
            policy_path = os.path.expanduser(args.detector_comparative_policy_file)
            with open(policy_path, "r", encoding="utf-8") as handle:
                comparative_policy = json.load(handle)
            if (
                comparative_policy.get("schema_version") != 1
                or comparative_policy.get("method") != "Object-Local Comparative Evidence Reversion ASCD"
                or comparative_policy.get("status") != "frozen"
            ):
                raise ValueError("Comparative Evidence Reversion requires its frozen policy schema")
            if comparative_policy.get("parent") != "fixed_ascd_pos0625":
                raise ValueError("Comparative Evidence Reversion policy has the wrong ASCD parent")
            if comparative_policy.get("runtime_detector") != "google/owlv2-base-patch16-ensemble":
                raise ValueError("Comparative Evidence Reversion policy has the wrong detector")
            if "threshold" not in comparative_policy:
                raise ValueError("Comparative Evidence Reversion policy is missing threshold")
            expected_hashes = comparative_policy.get("runtime_model_file_sha256")
            if not isinstance(expected_hashes, dict) or not expected_hashes:
                raise ValueError("Comparative Evidence Reversion policy is missing model hashes")
            for file_name, expected_hash in expected_hashes.items():
                candidate = model_path / str(file_name)
                if not candidate.is_file() or detector_sha256(candidate) != str(expected_hash):
                    raise ValueError(f"OWLv2 checkpoint hash mismatch: {candidate}")
            config.detector_comparative_threshold = float(comparative_policy["threshold"])
            config.detector_comparative_policy = comparative_policy
    if args.context_entropy_audit_file:
        audit_path = os.path.expanduser(args.context_entropy_audit_file)
        if os.path.exists(audit_path):
            raise FileExistsError(f"Refusing to overwrite context entropy audit: {audit_path}")
        args.contrastive_decoding_config.context_entropy_enabled = True
        args.contrastive_decoding_config.context_entropy_top_k = 3
        args.contrastive_decoding_config.context_entropy_beta = 10.0
        args.contrastive_decoding_config.context_entropy_records = []
    if args.margin_stats_file:
        margin_path = os.path.expanduser(args.margin_stats_file)
        if os.path.exists(margin_path):
            raise FileExistsError(f"Refusing to overwrite existing margin file: {margin_path}")
        args.contrastive_decoding_config.record_margin_values = True
        args.contrastive_decoding_config.recorded_margin_values = []
    diagnostic_file = None
    if args.diagnostics_file:
        diagnostics_path = os.path.expanduser(args.diagnostics_file)
        os.makedirs(os.path.dirname(diagnostics_path) or ".", exist_ok=True)
        diagnostic_file = open(diagnostics_path, "w")
        args.contrastive_decoding_config.diagnostics_enabled = True
        args.contrastive_decoding_config.diagnostic_file_handle = diagnostic_file
        args.contrastive_decoding_config.diagnostic_tokenizer = tokenizer
        args.contrastive_decoding_config.diagnostics_top_k = args.diagnostics_top_k
        args.contrastive_decoding_config.diagnostics_run_name = args.diagnostics_run_name
        set_self_denoise_attn_attr(
            model2modify,
            args.contrastive_attn_type,
            {"diagnostics_enabled": True},
        )
        print(f"Writing greedy token diagnostics to: {diagnostics_path}")
    forced_plan = {}
    forced_audit_file = None
    args.contrastive_decoding_config.forced_token_audit_enabled = False
    if args.forced_token_plan:
        forced_plan = load_forced_token_plan(args.forced_token_plan)
        audit_path = os.path.expanduser(args.forced_token_audit_file)
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        forced_audit_file = open(audit_path, "w")
        args.contrastive_decoding_config.forced_token_audit_enabled = True
        args.contrastive_decoding_config.forced_token_audit_file_handle = (
            forced_audit_file
        )
        print(
            f"Loaded forced-token plans: {len(forced_plan)}; "
            f"audit events: {audit_path}"
        )

    if getattr(model, "hf_device_map", None):
        print(f"Model already dispatched with device_map: {model.hf_device_map}")
    else:
        model.to(device='cuda')
    if args.verifier_audit_file:
        args.contrastive_decoding_config.verifier_runtime = CLIPVisualVerifier(
            args.verifier_clip_model_path, device="cuda"
        )
        print(
            "Verifier-Constrained ASCD: loaded frozen local CLIP verifier "
            f"from {args.verifier_clip_model_path}; "
            f"threshold={args.contrastive_decoding_config.verifier_threshold:.8f}; "
            f"observe_only={args.verifier_observe_only}"
        )
    if args.detector_grounded_audit_file:
        args.contrastive_decoding_config.detector_runtime = Owlv2ObjectRuntime(
            args.detector_grounded_model_path, device="cuda"
        )
        print(
            "Detector-Grounded ASCD: loaded frozen local OWLv2 detector from "
            f"{args.detector_grounded_model_path}; "
            f"threshold={args.contrastive_decoding_config.detector_threshold:.8f}; "
            f"top_k={args.contrastive_decoding_config.detector_top_k}"
        )

    if args.alias_guard_observe_audit_file:
        config = args.contrastive_decoding_config
        config.alias_guard_canonical_runtime = Owlv2ObjectRuntime(
            args.alias_guard_model_path, device="cuda"
        )
        config.alias_guard_alias_runtime = Owlv2AliasRuntime(
            args.alias_guard_model_path, device="cuda"
        )
        print("OIAG-ASCD: loaded frozen canonical and alias OWLv2 observe-only runtimes")
    if args.soft_grounded_audit_file:
        args.contrastive_decoding_config.soft_grounded_runtime = Owlv2ObjectRuntime(
            args.soft_grounded_model_path, device="cuda"
        )
        print("Soft-Grounded ASCD: loaded frozen local OWLv2 calibration policy")
    if args.detector_comparative_audit_file:
        args.contrastive_decoding_config.detector_comparative_runtime = Owlv2ObjectRuntime(
            args.detector_comparative_model_path, device="cuda"
        )
        print(
            "Comparative Evidence Reversion ASCD: loaded local OWLv2 verifier from "
            f"{args.detector_comparative_model_path}; "
            f"observe_only={args.detector_comparative_observe_only}"
        )
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

        image = Image.open(os.path.join(args.image_folder, file_name))
        image_sizes = [image.size]
        if image.layers == 1:
            image = Image.merge("RGB", (image, image, image))
        elif image.mode == "CMYK":
            image = image.convert("RGB")
        if args.verifier_audit_file:
            args.contrastive_decoding_config.verifier_image_id = int(id)
            args.contrastive_decoding_config.verifier_runtime.set_image(image)
        if args.detector_comparative_audit_file:
            config = args.contrastive_decoding_config
            config.detector_comparative_image_id = int(id)
            config.detector_comparative_runtime.set_image(image)
        if args.detector_grounded_audit_file:
            config = args.contrastive_decoding_config
            config.detector_image_id = int(id)
            support_scores = config.detector_runtime.set_image(image)
            config.detector_generated_token_ids = []
            config.detector_events = []
            config.detector_decoding_steps = 0
            config.detector_object_candidates = 0
            config.detector_masked_candidates = 0
            config.detector_selection_changes = 0
            config.detector_no_finite_protections = 0
            config.detector_support_score_min = float(min(support_scores.values()))
            config.detector_support_score_max = float(max(support_scores.values()))

        if args.alias_guard_observe_audit_file:
            config = args.contrastive_decoding_config
            config.alias_guard_image_id = int(id)
            canonical_scores = config.alias_guard_canonical_runtime.set_image(image)
            config.alias_guard_alias_runtime.set_image(image)
            config.alias_guard_generated_token_ids = []
            config.alias_guard_events = []
            config.alias_guard_decoding_steps = 0
            config.alias_guard_object_candidates = 0
            config.alias_guard_would_hard_mask = 0
            config.alias_guard_canonical_score_min = float(min(canonical_scores.values()))
            config.alias_guard_canonical_score_max = float(max(canonical_scores.values()))

        if args.soft_grounded_audit_file:
            config = args.contrastive_decoding_config
            support_scores = config.soft_grounded_runtime.set_image(image)
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

        if hasattr(model.config, "mm_use_im_start_end") and model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        if args.vista_audit_file:
            # The released LLaVA null prompt keeps the exact text tokens and
            # removes only the image placeholder/visual token sequence.
            input_ids_vista_null = input_ids[input_ids != IMAGE_TOKEN_INDEX].unsqueeze(0)
        if args.crops_audit_file:
            language_conv = conv_templates[args.conv_mode].copy()
            language_conv.append_message(language_conv.roles[0], qs.split("\n", 1)[-1])
            language_conv.append_message(language_conv.roles[1], None)
            input_ids_crops_language = tokenizer(
                language_conv.get_prompt(), return_tensors="pt", add_special_tokens=True,
            ).input_ids.cuda()
        if args.inter_audit_file:
            empty_qs = DEFAULT_IMAGE_TOKEN
            if hasattr(model.config, "mm_use_im_start_end") and model.config.mm_use_im_start_end:
                empty_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            empty_conv = conv_templates[args.conv_mode].copy()
            empty_conv.append_message(empty_conv.roles[0], empty_qs)
            empty_conv.append_message(empty_conv.roles[1], None)
            input_ids_inter_empty = tokenizer_image_token(
                empty_conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        if args.selfaug_audit_file:
            image_tensor_selfaug = augment_image(
                image_tensor.unsqueeze(0),
                args.contrastive_decoding_config.selfaug_applied_aug,
                crop_ratio=args.contrastive_decoding_config.selfaug_crop_ratio,
                mask_ratio=args.contrastive_decoding_config.selfaug_mask_ratio,
                noise_step=args.contrastive_decoding_config.selfaug_noise_step,
            )[0]
        if args.mfcd_audit_file:
            high_pass_image = gaussian_high_pass_filter(image, cutoff=0.1, device="cuda")
            low_pass_image = gaussian_low_pass_filter(image, cutoff=0.1, device="cuda")
            image_tensor_mfcd_high = image_processor.preprocess(
                high_pass_image, return_tensors="pt"
            )["pixel_values"][0]
            image_tensor_mfcd_low = image_processor.preprocess(
                low_pass_image, return_tensors="pt"
            )["pixel_values"][0]
            args.contrastive_decoding_config.mfcd_image_id = int(id)
        if args.inter_audit_file:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(20260813 + int(id))
            image_tensor_inter_random = torch.rand(
                image_tensor.shape, generator=generator, dtype=torch.float32
            ).to(image_tensor.dtype)
            args.contrastive_decoding_config.inter_image_id = int(id)
        if args.fuzzycd_audit_file:
            image_tensors_fuzzycd = [
                image_processor.preprocess(filtered, return_tensors="pt")["pixel_values"][0]
                for filtered in sharpen_images(image)
            ]
            args.contrastive_decoding_config.fuzzycd_image_id = int(id)
        
        # compute sys, image token length
        _, pos_in_batch = torch.where(input_ids==-200)
        sys_len = pos_in_batch[0].item()
        if "tinyllava" in str(type(model)):
            img_len = (model.vision_tower.config.image_size // model.vision_tower.config.patch_size)**2
            if model.config.vision_feature_select_strategy == "patch":
                img_len -= 1
        else:
            img_len = (model.model.vision_tower.config.image_size // model.model.vision_tower.config.patch_size)**2
        if args.context_entropy_audit_file:
            expanded_prompt_length = int(input_ids.shape[-1] - 1 + img_len)
            args.contrastive_decoding_config.context_entropy_image_start = int(sys_len)
            args.contrastive_decoding_config.context_entropy_image_length = int(img_len)
            args.contrastive_decoding_config.context_entropy_instruction_end = expanded_prompt_length
            args.contrastive_decoding_config.context_entropy_image_id = int(id)
            args.contrastive_decoding_config.context_entropy_step = 0
        if args.deco_audit_file:
            args.contrastive_decoding_config.deco_image_id = int(id)
            args.contrastive_decoding_config.deco_step = 0
        if args.only_audit_file:
            args.contrastive_decoding_config.only_image_id = int(id)
            args.contrastive_decoding_config.only_step = 0
        if args.mole_audit_file:
            expanded_prompt_length = int(input_ids.shape[-1] - 1 + img_len)
            args.contrastive_decoding_config.mole_prompt_end = expanded_prompt_length - 1
            args.contrastive_decoding_config.mole_image_id = int(id)
            args.contrastive_decoding_config.mole_step = 0
        if args.sumgd_audit_file:
            args.contrastive_decoding_config.sumgd_image_id = int(id)
        if args.crops_audit_file:
            args.contrastive_decoding_config.crops_image_id = int(id)
        if args.cei_audit_file:
            args.contrastive_decoding_config.cei_image_id = int(id)
        if args.dive_audit_file:
            args.contrastive_decoding_config.dive_image_id = int(id)
        if args.vhr_audit_file:
            args.contrastive_decoding_config.vhr_image_id = int(id)
        if args.selfaug_audit_file:
            args.contrastive_decoding_config.selfaug_image_id = int(id)
        if args.vista_audit_file:
            args.contrastive_decoding_config.vista_image_id = int(id)
        clearsight_state = None
        if args.clearsight_audit_file:
            frozen = clearsight_configuration()
            clearsight_state = {
                "enabled": True,
                "target_layers": frozen["target_layers"],
                "enhancement_multiplier": frozen["enhancement_multiplier"],
                "suppression_multiplier": frozen["suppression_multiplier"],
                "application_count": 0,
                "events": [],
            }
        
        wrapper_attrs = {"sys_len": sys_len, "img_len": img_len}
        if clearsight_state is not None:
            wrapper_attrs["clearsight_state"] = clearsight_state
        set_self_denoise_attn_attr(model2modify, args.contrastive_attn_type, wrapper_attrs)
        if diagnostic_file is not None:
            args.contrastive_decoding_config.diagnostic_sample_index = i
            args.contrastive_decoding_config.diagnostic_image_id = id
            args.contrastive_decoding_config.diagnostic_generated_token_ids = []
        
        args.contrastive_decoding_config.forced_token_spec = (
            dict(forced_plan[int(id)]) if int(id) in forced_plan else None
        )

        with torch.inference_mode():
            generation_kwargs = {}
            if args.mfcd_audit_file:
                generation_kwargs.update({
                    "images_mfcd_high": image_tensor_mfcd_high.unsqueeze(0).to(
                        dtype=torch.float16, device="cuda", non_blocking=True
                    ),
                    "images_mfcd_low": image_tensor_mfcd_low.unsqueeze(0).to(
                        dtype=torch.float16, device="cuda", non_blocking=True
                    ),
                })
            if args.inter_audit_file:
                generation_kwargs.update({
                    "images_inter_random": image_tensor_inter_random.unsqueeze(0).to(
                        dtype=torch.float16, device="cuda", non_blocking=True
                    ),
                    "input_ids_inter_empty": input_ids_inter_empty,
                })
            if args.fuzzycd_audit_file:
                generation_kwargs["images_fuzzycd"] = torch.stack(image_tensors_fuzzycd).to(
                    dtype=torch.float16, device="cuda", non_blocking=True
                )
            if args.crops_audit_file:
                generation_kwargs["input_ids_crops_language"] = input_ids_crops_language
            if args.selfaug_audit_file:
                generation_kwargs["images_cd"] = image_tensor_selfaug.unsqueeze(0).to(
                    dtype=torch.float16, device="cuda", non_blocking=True
                )
            if args.vista_audit_file:
                generation_kwargs["input_ids_vista_null"] = input_ids_vista_null
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
                **generation_kwargs)
            
        forced_spec = args.contrastive_decoding_config.forced_token_spec
        if forced_spec is not None and not forced_spec.get("_applied", False):
            raise RuntimeError(
                f"Forced-token plan for image_id={id} was not applied; "
                f"target_step={forced_spec['step']} max_new_tokens={args.max_new_tokens}."
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if args.detector_grounded_audit_file:
            config = args.contrastive_decoding_config
            generated = list(config.detector_generated_token_ids)
            eos_value = model.generation_config.eos_token_id
            eos_ids = {int(eos_value)} if isinstance(eos_value, int) else {int(value) for value in eos_value}
            terminated_by_eos = bool(generated and generated[-1] in eos_ids)
            config.detector_records.append({
                "image_id": int(id),
                "parent": "fixed_ascd",
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
        if args.alias_guard_observe_audit_file:
            config = args.contrastive_decoding_config
            generated = list(config.alias_guard_generated_token_ids)
            eos_value = model.generation_config.eos_token_id
            eos_ids = {int(eos_value)} if isinstance(eos_value, int) else {int(value) for value in eos_value}
            terminated_by_eos = bool(generated and generated[-1] in eos_ids)
            config.alias_guard_records.append({
                "image_id": int(id), "parent": "fixed_ascd", "mode": "observe_only",
                "hard_threshold": float(config.alias_guard_hard_threshold), "top_k": int(config.alias_guard_top_k),
                "generated_tokens": len(generated), "terminated_by_eos": terminated_by_eos,
                "hit_max_new_tokens": len(generated) >= int(args.max_new_tokens) and not terminated_by_eos,
                "decoding_steps": int(config.alias_guard_decoding_steps),
                "object_candidates_considered": int(config.alias_guard_object_candidates),
                "would_hard_mask_candidates": int(config.alias_guard_would_hard_mask),
                "canonical_score_min": float(config.alias_guard_canonical_score_min),
                "canonical_score_max": float(config.alias_guard_canonical_score_max),
                "events": config.alias_guard_events,
            })
        if args.soft_grounded_audit_file:
            config = args.contrastive_decoding_config
            generated = list(config.soft_grounded_generated_token_ids)
            eos_value = model.generation_config.eos_token_id
            eos_ids = {int(eos_value)} if isinstance(eos_value, int) else {int(value) for value in eos_value}
            terminated_by_eos = bool(generated and generated[-1] in eos_ids)
            config.soft_grounded_records.append({
                "image_id": int(id), "parent": "fixed_ascd", "top_k": int(config.soft_grounded_top_k),
                "generated_tokens": len(generated), "terminated_by_eos": terminated_by_eos,
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
        if clearsight_state is not None:
            if int(clearsight_state["application_count"]) == 0:
                raise RuntimeError(f"ClearSight VAF was not applied for image_id={id}")
            args.contrastive_decoding_config.clearsight_records.append({
                "image_id": int(id),
                "application_count": int(clearsight_state["application_count"]),
                "events": clearsight_state["events"],
            })
        
        ans_file.write(json.dumps({
                                "image_id": id,
                                "caption": outputs
                                }) + "\n")
        ans_file.flush()
        if diagnostic_file is not None:
            diagnostic_file.flush()
        if forced_audit_file is not None:
            forced_audit_file.flush()

    ans_file.close()
    if diagnostic_file is not None:
        diagnostic_file.close()
    if forced_audit_file is not None:
        forced_audit_file.close()
    if args.margin_stats_file:
        values = args.contrastive_decoding_config.recorded_margin_values
        if not values:
            raise RuntimeError("No adaptive margin values were recorded")
        margin_path = os.path.expanduser(args.margin_stats_file)
        os.makedirs(os.path.dirname(margin_path) or ".", exist_ok=True)
        with open(margin_path, "x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "records": values}, handle, indent=2)
            handle.write("\n")
    if args.context_entropy_audit_file:
        records = args.contrastive_decoding_config.context_entropy_records
        if not records:
            raise RuntimeError("No context entropy records were produced")
        audit_path = os.path.expanduser(args.context_entropy_audit_file)
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        with open(audit_path, "x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "records": records}, handle, indent=2)
            handle.write("\n")
    if args.deco_audit_file:
        records = args.contrastive_decoding_config.deco_records
        if not records:
            raise RuntimeError("No DeCo records were produced")
        audit_path = os.path.expanduser(args.deco_audit_file)
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        with open(audit_path, "x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "mode": args.deco_mode, "records": records}, handle, indent=2)
            handle.write("\n")
    if args.only_audit_file:
        records = args.contrastive_decoding_config.only_records
        if not records:
            raise RuntimeError("No ONLY records were produced")
        only_path = os.path.expanduser(args.only_audit_file)
        os.makedirs(os.path.dirname(only_path) or ".", exist_ok=True)
        with open(only_path, "x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "mode": args.only_mode, "records": records}, handle, indent=2)
            handle.write("\n")
    if args.mole_audit_file:
        records = args.contrastive_decoding_config.mole_records
        if not records:
            raise RuntimeError("No MoLE records were produced")
        mole_path = os.path.expanduser(args.mole_audit_file)
        os.makedirs(os.path.dirname(mole_path) or ".", exist_ok=True)
        with open(mole_path, "x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "mode": args.mole_mode, "records": records}, handle, indent=2)
            handle.write("\n")
    if args.sumgd_audit_file:
        records = args.contrastive_decoding_config.sumgd_records
        if not records:
            raise RuntimeError("No SumGD records were produced")
        sumgd_path = os.path.expanduser(args.sumgd_audit_file)
        os.makedirs(os.path.dirname(sumgd_path) or ".", exist_ok=True)
        with open(sumgd_path, "x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "mode": args.sumgd_mode, "records": records}, handle, indent=2)
            handle.write("\n")
    if args.allpath_audit_file:
        allpath_path = os.path.expanduser(args.allpath_audit_file)
        os.makedirs(os.path.dirname(allpath_path) or ".", exist_ok=True)
        write_configuration(allpath_path)
    if args.mfcd_audit_file:
        records = args.contrastive_decoding_config.mfcd_records
        if not records:
            raise RuntimeError("No MFCD records were produced")
        mfcd_path = os.path.expanduser(args.mfcd_audit_file)
        os.makedirs(os.path.dirname(mfcd_path) or ".", exist_ok=True)
        with open(mfcd_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "mode": args.mfcd_mode,
                "configuration": mfcd_configuration(),
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.selfaug_audit_file:
        records = args.contrastive_decoding_config.selfaug_records
        if len(records) != len(data):
            raise RuntimeError(f"Self-Aug audit records={len(records)} data={len(data)}")
        selfaug_path = os.path.expanduser(args.selfaug_audit_file)
        os.makedirs(os.path.dirname(selfaug_path) or ".", exist_ok=True)
        with open(selfaug_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "mode": args.selfaug_mode,
                "source_commit": "ff3c77c7711eade2e1d78091c8942dd0e5391d7d",
                "configuration": selfaug_configuration(),
                "sas": args.contrastive_decoding_config.selfaug_sas_payload,
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.vista_audit_file:
        records = args.contrastive_decoding_config.vista_records
        if len(records) != len(data):
            raise RuntimeError(f"VISTA audit records={len(records)} data={len(data)}")
        vista_path = os.path.expanduser(args.vista_audit_file)
        os.makedirs(os.path.dirname(vista_path) or ".", exist_ok=True)
        with open(vista_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "mode": args.vista_mode,
                "source_commit": "efcf499919e066755e7c33778fbfd864c204329c",
                "configuration": vista_configuration(),
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.clearsight_audit_file:
        records = args.contrastive_decoding_config.clearsight_records
        if len(records) != len(data):
            raise RuntimeError(f"ClearSight audit records={len(records)} data={len(data)}")
        clearsight_path = os.path.expanduser(args.clearsight_audit_file)
        os.makedirs(os.path.dirname(clearsight_path) or ".", exist_ok=True)
        with open(clearsight_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "mode": args.clearsight_mode,
                "source_commit": CLEARSIGHT_SOURCE_COMMIT,
                "configuration": clearsight_configuration(),
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.verifier_audit_file:
        records = args.contrastive_decoding_config.verifier_records
        if len(records) != len(data):
            raise RuntimeError(
                "Verifier-Constrained ASCD must emit exactly one audit record "
                f"per image: records={len(records)} images={len(data)}"
            )
        verifier_path = os.path.expanduser(args.verifier_audit_file)
        os.makedirs(os.path.dirname(verifier_path) or ".", exist_ok=True)
        with open(verifier_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "method": "Verifier-Constrained ASCD",
                "mode": "observe_only" if args.verifier_observe_only else "constrained",
                "clip_model_path": os.path.abspath(args.verifier_clip_model_path),
                "calibration": args.contrastive_decoding_config.verifier_calibration,
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.detector_grounded_audit_file:
        records = args.contrastive_decoding_config.detector_records
        if len(records) != len(data):
            raise RuntimeError(
                "Detector-Grounded ASCD must emit exactly one audit record "
                f"per image: records={len(records)} images={len(data)}"
            )
        detector_path = os.path.expanduser(args.detector_grounded_audit_file)
        os.makedirs(os.path.dirname(detector_path) or ".", exist_ok=True)
        with open(detector_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "method": "Detector-Grounded ASCD",
                "detector_model_path": os.path.abspath(args.detector_grounded_model_path),
                "policy": args.contrastive_decoding_config.detector_policy,
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.alias_guard_observe_audit_file:
        records = args.contrastive_decoding_config.alias_guard_records
        if len(records) != len(data):
            raise RuntimeError(f"OIAG alias audit mismatch: records={len(records)} images={len(data)}")
        alias_path = os.path.expanduser(args.alias_guard_observe_audit_file)
        os.makedirs(os.path.dirname(alias_path) or ".", exist_ok=True)
        with open(alias_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "method": "Object-Instance Alias Guard ASCD",
                "mode": "observe_only",
                "detector_model_path": os.path.abspath(args.alias_guard_model_path),
                "hard_detector_policy": args.contrastive_decoding_config.alias_guard_policy,
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
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
                "detector_model_path": os.path.abspath(args.soft_grounded_model_path),
                "policy": args.contrastive_decoding_config.soft_grounded_policy,
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.detector_comparative_audit_file:
        records = args.contrastive_decoding_config.detector_comparative_records
        if len(records) != len(data):
            raise RuntimeError(
                "Comparative Evidence Reversion ASCD must emit exactly one audit record "
                f"per image: records={len(records)} images={len(data)}"
            )
        comparative_path = os.path.expanduser(args.detector_comparative_audit_file)
        os.makedirs(os.path.dirname(comparative_path) or ".", exist_ok=True)
        with open(comparative_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "method": "Object-Local Comparative Evidence Reversion ASCD",
                "mode": "observe_only" if args.detector_comparative_observe_only else "constrained",
                "detector_model_path": os.path.abspath(args.detector_comparative_model_path),
                "policy": args.contrastive_decoding_config.detector_comparative_policy,
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.inter_audit_file:
        records = args.contrastive_decoding_config.inter_records
        if not records:
            raise RuntimeError("No INTER records were produced")
        inter_path = os.path.expanduser(args.inter_audit_file)
        os.makedirs(os.path.dirname(inter_path) or ".", exist_ok=True)
        with open(inter_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "mode": args.inter_mode,
                "configuration": inter_configuration(),
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.fuzzycd_audit_file:
        records = args.contrastive_decoding_config.fuzzycd_records
        if not records:
            raise RuntimeError("No FuzzyCD records were produced")
        fuzzycd_path = os.path.expanduser(args.fuzzycd_audit_file)
        os.makedirs(os.path.dirname(fuzzycd_path) or ".", exist_ok=True)
        with open(fuzzycd_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "mode": args.fuzzycd_mode,
                "configuration": fuzzycd_configuration(args.contrastive_decoding_config.fuzzycd_calibration),
                "calibration_file": os.path.abspath(os.path.expanduser(args.fuzzycd_calibration_file)),
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.crops_audit_file:
        records = args.contrastive_decoding_config.crops_records
        if not records:
            raise RuntimeError("No CRoPS records were produced")
        crops_path = os.path.expanduser(args.crops_audit_file)
        os.makedirs(os.path.dirname(crops_path) or ".", exist_ok=True)
        with open(crops_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "mode": args.crops_mode,
                "configuration": crops_configuration(),
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.cei_audit_file:
        records = args.contrastive_decoding_config.cei_records
        if not records:
            raise RuntimeError("No CEI records were produced")
        cei_path = os.path.expanduser(args.cei_audit_file)
        os.makedirs(os.path.dirname(cei_path) or ".", exist_ok=True)
        with open(cei_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "mode": args.cei_mode,
                "source_commit": "625707fa29ae620eb9ccb6f0023045ad0dd39261",
                "configuration": cei_configuration(),
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.dive_audit_file:
        records = args.contrastive_decoding_config.dive_records
        if not records:
            raise RuntimeError("No DiVE records were produced")
        dive_path = os.path.expanduser(args.dive_audit_file)
        os.makedirs(os.path.dirname(dive_path) or ".", exist_ok=True)
        with open(dive_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "mode": args.dive_mode,
                "source": "ACL-2026-paper-faithful-slow-adaptation",
                "configuration": dive_configuration(),
                "records": records,
            }, handle, indent=2)
            handle.write("\n")
    if args.vhr_audit_file:
        records = args.contrastive_decoding_config.vhr_records
        if not records:
            raise RuntimeError("No VHR records were produced")
        vhr_path = os.path.expanduser(args.vhr_audit_file)
        os.makedirs(os.path.dirname(vhr_path) or ".", exist_ok=True)
        with open(vhr_path, "x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "mode": args.vhr_mode,
                "source_commit": "f0db54a7eae62b4b8d1d585636a446ed40799512",
                "configuration": {
                    "augmentation_ratio": 2.0,
                    "last_layers": 14,
                    "include_layer_one": True,
                    "outlier_filter": True,
                },
                "records": records,
            }, handle, indent=2)
            handle.write("\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector-comparative-audit-file", type=str, default="",
        help="JSON audit for default-off object-local Comparative Evidence Reversion ASCD.",
    )
    parser.add_argument(
        "--detector-comparative-policy-file", type=str, default="",
        help="Frozen object-local comparative policy from the independent calibration split.",
    )
    parser.add_argument(
        "--detector-comparative-model-path", type=str,
        default="/root/projects/ASCD-main/models/owlv2-base-patch16-ensemble",
        help="Local official OWLv2 checkpoint used only as the comparative verifier.",
    )
    parser.add_argument(
        "--detector-comparative-observe-only", action="store_true", default=False,
        help="Record comparative candidates but emit exactly Fixed-ASCD tokens for calibration.",
    )
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
    parser.add_argument(
        "--exclude-image-ids-file",
        type=str,
        default=None,
        help=(
            "Optional JSON/JSONL whose image ids are excluded before seeded "
            "sampling; accepts CHAIR detail JSON directly."
        ),
    )

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
    parser.add_argument(
        "--diagnostics-file",
        type=str,
        default=None,
        help="Optional JSONL path for greedy token-level diagnostics. Disabled by default.",
    )
    parser.add_argument(
        "--margin-stats-file",
        type=str,
        default="",
        help="Optional compact raw-margin JSON output. Disabled by default.",
    )
    parser.add_argument("--context-entropy-audit-file", type=str, default="")
    parser.add_argument("--deco-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--deco-audit-file", type=str, default="")
    parser.add_argument("--only-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--only-audit-file", type=str, default="")
    parser.add_argument("--mole-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--mole-audit-file", type=str, default="")
    parser.add_argument("--sumgd-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--sumgd-audit-file", type=str, default="")
    parser.add_argument("--allpath-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--allpath-audit-file", type=str, default="")
    parser.add_argument("--mfcd-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--mfcd-audit-file", type=str, default="")
    parser.add_argument("--inter-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--inter-audit-file", type=str, default="")
    parser.add_argument("--fuzzycd-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--fuzzycd-audit-file", type=str, default="")
    parser.add_argument("--fuzzycd-calibration-file", type=str, default="")
    parser.add_argument("--crops-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--crops-audit-file", type=str, default="")
    parser.add_argument("--cei-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--cei-audit-file", type=str, default="")
    parser.add_argument("--dive-mode", choices=("off", "official", "hybrid"), default="off")
    parser.add_argument("--dive-audit-file", type=str, default="")
    parser.add_argument("--vhr-mode", choices=("off", "official"), default="off")
    parser.add_argument("--vhr-audit-file", type=str, default="")
    parser.add_argument("--selfaug-mode", choices=("off", "official"), default="off")
    parser.add_argument("--selfaug-audit-file", type=str, default="")
    parser.add_argument("--selfaug-sas-file", type=str, default="")
    parser.add_argument("--vista-mode", choices=("off", "official"), default="off")
    parser.add_argument("--vista-audit-file", type=str, default="")
    parser.add_argument("--clearsight-mode", choices=("off", "official"), default="off")
    parser.add_argument("--clearsight-audit-file", type=str, default="")
    parser.add_argument(
        "--verifier-audit-file", type=str, default="",
        help="JSON audit for the standalone Verifier-Constrained ASCD policy.",
    )
    parser.add_argument(
        "--verifier-calibration-file", type=str, default="",
        help="Frozen JSON threshold artifact from an independent observe-only split.",
    )
    parser.add_argument(
        "--verifier-clip-model-path", type=str,
        default="/root/projects/ASCD-main/models/clip-vit-large-patch14-336",
        help="Local OpenAI CLIP checkpoint used only as the external verifier.",
    )
    parser.add_argument(
        "--verifier-observe-only", action="store_true", default=False,
        help="Record candidate evidence but keep Fixed-ASCD tokens for calibration.",
    )
    parser.add_argument(
        "--detector-grounded-audit-file", type=str, default="",
        help="JSON audit for the standalone Detector-Grounded ASCD policy.",
    )
    parser.add_argument(
        "--detector-grounded-policy-file", type=str, default="",
        help="Frozen Detector-Grounded ASCD policy from the independent calibration split.",
    )
    parser.add_argument(
        "--detector-grounded-model-path", type=str,
        default="/root/projects/ASCD-main/models/owlv2-base-patch16-ensemble",
        help="Local official OWLv2 checkpoint used only as an external object detector.",
    )
    parser.add_argument("--soft-grounded-audit-file", type=str, default="")
    parser.add_argument("--soft-grounded-policy-file", type=str, default="")
    parser.add_argument(
        "--soft-grounded-model-path", type=str,
        default="/root/projects/ASCD-main/models/owlv2-base-patch16-ensemble",
    )
    parser.add_argument("--alias-guard-observe-audit-file", type=str, default="")
    parser.add_argument("--alias-guard-hard-policy-file", type=str, default="")
    parser.add_argument(
        "--alias-guard-model-path", type=str,
        default="/root/projects/ASCD-main/models/owlv2-base-patch16-ensemble",
    )
    parser.add_argument(
        "--diagnostics-top-k",
        type=int,
        default=5,
        help="Number of positive/negative branch top tokens saved per decoding step.",
    )
    parser.add_argument(
        "--diagnostics-run-name",
        type=str,
        default=None,
        help="Optional run identifier written into every diagnostic record.",
    )
    parser.add_argument(
        "--forced-token-plan",
        type=str,
        default=None,
        help="Optional pre-registered JSONL plan for one greedy token intervention per image.",
    )
    parser.add_argument(
        "--forced-token-audit-file",
        type=str,
        default=None,
        help="JSONL output for applied forced-token intervention events.",
    )
    parser.add_argument(
        "--neutral-directional-diagnostics",
        action="store_true",
        default=False,
        help="Add an independent unmodified neutral forward to greedy diagnostics.",
    )

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
        args.contrastive_decoding_config.neutral_directional_diagnostics = (
            args.neutral_directional_diagnostics
        )
    else:
        args.contrastive_decoding_config = None
        print("The config file for contrastive decoding is not loaded!")

    eval_model(args)
