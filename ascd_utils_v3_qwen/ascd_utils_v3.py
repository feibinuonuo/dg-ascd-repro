from typing import Dict
import re, os
import torch
from . import *
from .ascd_models_v3_qwen import AttnSteerConfig
from dataclasses import dataclass, field
from typing import Optional, Sequence


def get_parent_module(model, module_name):
    components = module_name.split('.')
    parent_module = model
    for comp in components[:-1]:
        parent_module = getattr(parent_module, comp)
    return parent_module, components[-1]

def _replace_attention_with_contrastive_attn(model, type_contrastive_attn, selected_layers, extra_args={}):
    attn_class = find_attn_class(model)
    
    for name, module in model.named_modules():
        if isinstance(module, attn_class):
            match = re.search(MODULE_NAME_TEMPLATE[type(model)], name)
            layer_id = int(match.group(1))
            if not hasattr(module, 'layer_idx'):
                setattr(module, 'layer_idx', layer_id)
            if layer_id not in selected_layers:
                continue
            parent_module, attr_name = get_parent_module(model, name)
            new_module = CONTRASTIVE_ATTN_MAPPING[type_contrastive_attn][attn_class](module, **extra_args)
            setattr(parent_module, attr_name, new_module)

def parse_layer_selection(input_string, total_layer_num):
    if re.match(r"^f(\d+)$", input_string):  # Matches "f4"
        count = int(re.match(r"^f(\d+)$", input_string).group(1))
        if count > total_layer_num:
            raise ValueError(f"f{count} exceeds the total number of layers {total_layer_num}.")
        return list(range(count))
    
    elif re.match(r"^l(\d+)$", input_string):  # Matches "l4"
        count = int(re.match(r"^l(\d+)$", input_string).group(1))
        if count > total_layer_num:
            raise ValueError(f"l{count} exceeds the total number of layers {total_layer_num}.")
        return list(range(total_layer_num - count, total_layer_num))
    
    elif input_string == "all":  # Matches "all"
        return list(range(total_layer_num))
    
    elif re.match(r"^(\d+)-(\d+)$", input_string):  # Matches "1-3"
        start, end = map(int, re.match(r"^(\d+)-(\d+)$", input_string).groups())
        if start > end or end >= total_layer_num:
            raise ValueError(f"Range {start}-{end} exceeds the total number of layers {total_layer_num}.")
        return list(range(start, end + 1))
    
    elif re.match(r"^(\d+,)+\d+$", input_string):  # Matches "1,3,5"
        indices = list(map(int, input_string.split(',')))
        if any(i >= total_layer_num for i in indices):
            raise ValueError(f"Some indices exceed the total number of layers {total_layer_num}.")
        return indices
    
    elif re.match(r"^\d+$", input_string):  # Matches a single number, e.g., "5"
        index = int(input_string)
        if index >= total_layer_num:
            raise ValueError(f"Index {index} exceeds the total number of layers {total_layer_num}.")
        return [index]
    
    else:
        raise ValueError(
            "Invalid input format! Please use one of the following formats:\n"
            "- 'f4': The first 4 layers.\n"
            "- 'l4': The last 4 layers.\n"
            "- 'all': All layers.\n"
            "- '1-3': A range of layers (e.g., layers 1 to 3).\n"
            "- '1,3,5': Specific layers (e.g., layers 1, 3, 5)."
            "- '5': A single layer (e.g., layer 5)."
        )
    

def replace_attention_with_contrastive_attn(model, type_contrastive_attn, attn_steer_configs, **kwargs):
    
    layer_applied = kwargs.get("layer_applied", None)
    total_layer_num = model.config.num_hidden_layers

    selected_layers = parse_layer_selection(layer_applied, total_layer_num) if layer_applied else []
    return _replace_attention_with_contrastive_attn(model, type_contrastive_attn, selected_layers,
                                                    {"attn_steer_configs": attn_steer_configs})


def replace_denoise_attn(model, yaml_configs, **kwargs):

    contrastive_attn_type = kwargs.get('contrastive_attn_type', None)
    contrastive_layer_ids = kwargs.get('contrastive_layer_ids', None)
    context_entropy_enabled = bool(kwargs.get('context_entropy_enabled', False))

    attn_steer_config1, hall_head_score_map1 = _generate_attn_steer_config_wo_head_map(yaml_configs[0])
    attn_steer_config2, hall_head_score_map2 = _generate_attn_steer_config_wo_head_map(yaml_configs[1])

    attn_steer_configs = [attn_steer_config1, attn_steer_config2]
    if context_entropy_enabled:
        attn_steer_configs.append(AttnSteerConfig(modify_attn=False))

    if contrastive_attn_type:
        replace_attention_with_contrastive_attn(model,
                                                contrastive_attn_type,
                                                tuple(attn_steer_configs),
                                                layer_applied=contrastive_layer_ids)

    model.contrastive_attn_type = contrastive_attn_type
    if hasattr(model, "model"):
        model.model.contrastive_attn_type = contrastive_attn_type
    attn_class = find_attn_class(model)
    denosie_attn_class = CONTRASTIVE_ATTN_MAPPING[contrastive_attn_type][attn_class] if attn_class in list(CONTRASTIVE_ATTN_MAPPING[contrastive_attn_type].keys()) else attn_class
    for name, module in model.named_modules():
        if isinstance(module, denosie_attn_class):
            layer_idx = module.original_module.layer_idx
            
            module.attn_steer_configs[0].hall_head_mask = hall_head_score_map1[layer_idx] if isinstance(hall_head_score_map1, torch.Tensor) else None
            module.attn_steer_configs[1].hall_head_mask = hall_head_score_map2[layer_idx] if isinstance(hall_head_score_map2, torch.Tensor) else None

def generate_topk_mask(matrix, k):
    flat_matrix = matrix.view(-1)  # Shape: (1024,)

    topk_values, topk_indices = torch.topk(flat_matrix, k, largest=True, sorted=False)  # 找最大值的 top-k

    mask = torch.zeros_like(flat_matrix, dtype=torch.bool)  # Shape: (1024,)
    mask[topk_indices] = True

    mask = mask.view(matrix.size())  # Shape: (32, 32)

    return mask

def _generate_attn_steer_config_wo_head_map(yaml_config):
    steer_mode = getattr(yaml_config, "steer_mode", "abs")
    attn_steer_config = AttnSteerConfig(modify_attn=getattr(yaml_config, "modify_attn", True),
                                        method_steer_sys=yaml_config.method_steer_sys,
                                        method_steer_vis=yaml_config.method_steer_vis,
                                        method_steer_text=yaml_config.method_steer_text,
                                        upscale_sys=yaml_config.upscale_sys,
                                        downscale_vis=yaml_config.downscale_vis,
                                        upscale_text=yaml_config.upscale_text,
                                        topk_vis_token_for_downscale=yaml_config.topk_vis_token_for_downscale,
                                        hall_head_mask=None,
                                        steer_mode=steer_mode)
    hall_head_score_map = yaml_config.hall_head_score_map
    topk = yaml_config.topk_hall_heads

    if hall_head_score_map is not None and hall_head_score_map != "None" and hall_head_score_map != "none" and os.path.exists(hall_head_score_map):
        hall_head_score_map = torch.load(hall_head_score_map)
        hall_head_score_map = generate_topk_mask(hall_head_score_map, k=topk) if hall_head_score_map.dtype is not torch.bool else hall_head_score_map
        assert isinstance(hall_head_score_map, torch.Tensor) and hall_head_score_map.dtype == torch.bool, "The head map must be a torch.Tensor of bool dtype!"

    return attn_steer_config, hall_head_score_map


def set_self_attn_attr(model, attr: Dict):
    attn_class = find_attn_class(model)

    for name, module in model.named_modules():
        if isinstance(module, attn_class):
            for k, v in attr.items():
                setattr(module, k, v)


def set_self_denoise_attn_attr(model, contrastive_attn_type: str, attr: Dict):
    attn_class = find_attn_class(model)
    denosie_attn_class = CONTRASTIVE_ATTN_MAPPING[contrastive_attn_type][attn_class] if attn_class in list(CONTRASTIVE_ATTN_MAPPING[contrastive_attn_type].keys()) else attn_class

    for name, module in model.named_modules():
        if isinstance(module, denosie_attn_class):
            for k, v in attr.items():
                setattr(module, k, v)
