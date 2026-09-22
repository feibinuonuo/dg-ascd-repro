
from typing import List, Optional, Tuple, Union
import warnings
import torch
from torch import nn
import math, re
import torch.nn.functional as F

from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
# from .ascd_models_v3 import AttnSteerConfig

from dataclasses import dataclass, field
from typing import Optional, Sequence

@dataclass
class AttnSteerConfig:
    modify_attn: bool = True
    method_steer_sys: str = "none"
    method_steer_vis: str = "none"
    method_steer_text: str = "none"
    upscale_sys: float = 0.0
    downscale_vis: float = 0.0
    upscale_text: float = 0.0
    topk_vis_token_for_downscale: float = 1.0
    hall_head_mask: Optional[torch.Tensor] = None
    steer_mode: str = "abs"


def steer_func(obj, src, steer_factor: float, steer_mode: str = "abs"):
    if steer_mode == "abs":
        obj += steer_factor * torch.abs(src)
    elif steer_mode == "non-neg":
        obj += steer_factor * torch.abs(src)
    else:
        raise NotImplementedError(f"The steer mode {steer_mode} is not implemented!")
    
class MistralAllAttentionDenoiseBase(nn.Module):
    def __init__(self,
                 original_module,
                 attn_steer_configs: Union[AttnSteerConfig, Sequence[AttnSteerConfig]]):
        super().__init__()

        self.original_module = original_module

        self.cur_config_id = 0
        self.attn_steer_configs = attn_steer_configs

        self.sys_len = 34
        self.img_len = 576

class MistralAttentionDenoiseV1(MistralAllAttentionDenoiseBase):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attn = self.original_module
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if attn.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {attn.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, attn.layer_idx)
        cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

        if attn_weights.size() != (bsz, attn.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, attn.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        ##### modification #####
        cur_config = self.attn_steer_configs[self.cur_config_id]
        if cur_config.modify_attn:
            if cur_config.method_steer_sys == "all":
                attn_weights[:, :, -1, :self.sys_len] += cur_config.upscale_sys * torch.abs(attn_weights[:, :, -1, :self.sys_len])
            elif cur_config.method_steer_sys == "hall-head":
                assert cur_config.hall_head_mask is not None, "You are trying to apply tp hallucination heads. But no valid head-mask-map has been set."
                attn_weights[:, cur_config.hall_head_mask, -1, :self.sys_len] += cur_config.upscale_sys * torch.abs(attn_weights[:, cur_config.hall_head_mask, -1, :self.sys_len])
                # print("method_steer_sys==hall-head")
            elif cur_config.method_steer_sys == "none":
                pass
            else:
                raise ValueError(f"method_steer_sys is {cur_config.method_steer_sys}, should be one of ['all', 'hall-head', 'none']")

            if cur_config.method_steer_vis == "all":
                vis_attn = attn_weights[:, :, -1, self.sys_len:self.sys_len+self.img_len]
                _, topk_indices = torch.topk(vis_attn, int(self.img_len * cur_config.topk_vis_token_for_downscale), dim=-1)
                topk_mask = torch.zeros_like(vis_attn, dtype=torch.bool)
                topk_mask.scatter_(-1, topk_indices, True)

                attn_weights[:, :, -1, self.sys_len:self.sys_len+self.img_len][topk_mask] -= cur_config.downscale_vis * torch.abs(attn_weights[:, :, -1, self.sys_len:self.sys_len+self.img_len][topk_mask])
            elif cur_config.method_steer_vis == "hall-head":
                vis_attn = attn_weights[:, cur_config.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len]
                _, topk_indices = torch.topk(vis_attn, int(self.img_len * cur_config.topk_vis_token_for_downscale), dim=-1)
                topk_mask = torch.zeros_like(vis_attn, dtype=torch.bool)
                topk_mask.scatter_(-1, topk_indices, True)
                # print("method_steer_vis==hall-head")
                assert cur_config.hall_head_mask is not None, "You are trying to apply tp hallucination heads. But no valid head-mask-map has been set."
                # attn_weights[:, self.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len][topk_mask] -= self.downscale_vis * torch.abs(attn_weights[:, self.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len][topk_mask])
                attn_weights[:, cur_config.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len] -= (
                    cur_config.downscale_vis * torch.abs(vis_attn) * topk_mask
                )
            elif cur_config.method_steer_vis == "none":
                pass
            else:
                raise ValueError(f"method_steer_vis is {cur_config.method_steer_vis}, should be one of ['all', 'hall-head', 'none']")

            if cur_config.method_steer_text == "all":
                attn_weights[:, :, -1, self.sys_len+self.img_len:] += cur_config.upscale_text * torch.abs(attn_weights[:, :, -1, self.sys_len+self.img_len:])
            elif cur_config.method_steer_text == "hall-head":
                # print("method_steer_text==hall-head")
                assert cur_config.hall_head_mask is not None, "You are trying to apply tp hallucination heads. But no valid head-mask-map has been set."
                attn_weights[:, cur_config.hall_head_mask, -1, self.sys_len+self.img_len:] += cur_config.upscale_text * torch.abs(attn_weights[:, cur_config.hall_head_mask, -1, self.sys_len+self.img_len:])
            elif cur_config.method_steer_text == "none":
                pass
            else:
                raise ValueError(f"method_steer_text is {cur_config.method_steer_text}, should be one of ['all', 'hall-head', 'none']")

        ########################

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, attn.num_heads, q_len, attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn.num_heads, q_len, attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

        attn_output = attn.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


