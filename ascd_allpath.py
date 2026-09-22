"""Frozen NeurIPS 2025 AllPath LLaVA/CHAIR head configuration."""

import json
from collections import defaultdict


FORMAT_HEADS_SHA256 = "f554eb65137ce7b11ae947013c1a8f53cbdf7cf440c8bceb39682fb53f210151"
IMAGE_HEADS_SHA256 = "0755945f605f00eb1af674b6c923c60aadd686b30729926fce4fed0c95f7c49f"

# Official CHAIR defaults after the released overlap-removal rule.  The good
# list contains 40 format-path and 50 image-path entries; one duplicate head
# occurs across those lists, yielding 89 unique good heads.
HALLU_HEADS = {
    15: (8, 10, 25), 16: (10, 29), 17: (1, 8), 18: (9,),
    19: (1, 12, 18, 27), 21: (24,), 22: (3, 17, 20, 26, 29),
    23: (19,), 24: (4,), 25: (12,), 26: (2, 9, 16, 19, 28),
    27: (29,), 28: (21,), 29: (4, 8, 10, 15), 30: (5, 9, 10),
    31: (0, 3, 8, 22, 26),
}

GOOD_HEADS = {
    5: (0, 11, 15), 6: (3, 15, 19, 28), 7: (8, 16, 22, 26),
    8: (2, 6, 20), 9: (17, 27, 28), 10: (6, 17), 12: (14, 21, 31),
    13: (8, 18), 14: (13, 20, 27), 15: (22,), 16: (4, 5, 9, 24),
    17: (9, 11, 16, 17, 18, 31), 18: (6, 15, 30),
    19: (0, 4, 6, 7, 9, 10, 14, 26),
    20: (5, 10, 12, 18, 28, 29), 21: (1, 2, 9, 16, 22, 30, 31),
    22: (4, 10, 16, 23, 30), 23: (5,), 24: (3, 5, 15, 16, 17, 20, 29),
    25: (19, 21), 26: (12, 15, 18, 25), 27: (4, 19, 20),
    28: (20,), 29: (19, 29), 30: (17,), 31: (16,),
}


def allpath_configuration():
    return {
        "source_commit": "c8792510f12886848b3f7585db0246af9ac16013",
        "format_heads_sha256": FORMAT_HEADS_SHA256,
        "image_heads_sha256": IMAGE_HEADS_SHA256,
        "official_selection_counts": {
            "hallu_format": 40, "good_format": 40,
            "hallu_image": 0, "good_image": 50,
        },
        "unique_head_counts": {
            "hallu": sum(len(v) for v in HALLU_HEADS.values()),
            "good": sum(len(v) for v in GOOD_HEADS.values()),
        },
        "in_scale": 2.0,
        "de_scale": 0.0,
        "norm_distribution": False,
        "hallu_heads": {str(k): list(v) for k, v in HALLU_HEADS.items()},
        "good_heads": {str(k): list(v) for k, v in GOOD_HEADS.items()},
    }


def write_configuration(path):
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(allpath_configuration(), handle, indent=2)
        handle.write("\n")


def install_allpath_on_wrappers(model, enabled):
    wrappers = {}
    for module in model.modules():
        if not hasattr(module, "attn_steer_configs"):
            continue
        layer = int(getattr(module.original_module, "layer_idx", -1))
        wrappers[layer] = module
    if enabled and sorted(wrappers) != list(range(32)):
        raise RuntimeError("AllPath requires exactly 32 contiguous LLaVA layers")
    for layer, module in wrappers.items():
        module.allpath_enabled = bool(enabled)
        module.allpath_hallu_heads = HALLU_HEADS.get(layer, ())
        module.allpath_good_heads = GOOD_HEADS.get(layer, ())
        module.allpath_in_scale = 2.0
        module.allpath_de_scale = 0.0
