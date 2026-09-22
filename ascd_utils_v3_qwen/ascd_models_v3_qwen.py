
from typing import List, Optional, Tuple, Union
import warnings
import torch
from torch import nn
import math, re
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import  Qwen2_5_VLRotaryEmbedding, apply_multimodal_rotary_pos_emb, repeat_kv
from transformers.utils import logging
from transformers.cache_utils import Cache
logger = logging.get_logger(__name__)

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
    

class Qwen2_5_VLAttentionDenoiseBase(nn.Module):
    def __init__(self,
                 original_module,
                 attn_steer_configs: Union[AttnSteerConfig, Sequence[AttnSteerConfig]]):
        super().__init__()

        self.original_module = original_module

        self.cur_config_id = 0
        self.attn_steer_configs = attn_steer_configs

        self.sys_len = 32
        self.img_len = 728

    def _capture_context_entropy(self, attn_weights: torch.Tensor) -> None:
        if not bool(getattr(self, "context_entropy_capture_enabled", False)):
            return
        if int(getattr(self.original_module, "layer_idx", -1)) != int(
            getattr(self, "context_entropy_layer_id", -2)
        ):
            return
        last_query = attn_weights[:, :, -1, :].detach().float()
        key_len = last_query.shape[-1]
        image_start = int(self.sys_len)
        image_end = image_start + int(self.img_len)
        instruction_end = int(getattr(self, "context_entropy_instruction_end"))
        candidate_position = key_len - 1
        if not (
            0 <= image_start < image_end <= instruction_end
            <= candidate_position < key_len
        ):
            raise ValueError(
                "invalid context entropy boundaries: "
                f"image={image_start}:{image_end} "
                f"instruction_end={instruction_end} key_length={key_len}"
            )
        instruction_parts = []
        if image_start:
            instruction_parts.append(last_query[..., :image_start])
        if instruction_end > image_end:
            instruction_parts.append(last_query[..., image_end:instruction_end])
        if not instruction_parts:
            raise ValueError("context entropy instruction span is empty")
        visual = last_query[..., image_start:image_end].mean(dim=-1)
        instruction = torch.cat(instruction_parts, dim=-1).mean(dim=-1)
        if candidate_position > instruction_end:
            history = last_query[..., instruction_end:candidate_position].mean(dim=-1)
        else:
            history = torch.zeros_like(visual)
        self._context_entropy_components_by_config = {
            int(self.cur_config_id): torch.stack(
                (visual[0], instruction[0], history[0]), dim=-1
            ).detach()
        }


class Qwen2_5_VLAttentionDenoiseV1(Qwen2_5_VLAttentionDenoiseBase):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attn = self.original_module
        bsz, q_len, _ = hidden_states.size()

        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, attn.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        ##### modification #####
        cur_config = self.attn_steer_configs[self.cur_config_id]
        steer_mode = getattr(cur_config, "steer_mode", "abs")
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
                ############ test random topk token ######################
                # num_random_tokens = int(self.img_len * cur_config.topk_vis_token_for_downscale)
                # batch_size, num_heads, img_len = vis_attn.shape
                # topk_indices = torch.stack([
                #     torch.randperm(img_len, device=vis_attn.device)[:num_random_tokens]
                #     for _ in range(batch_size * num_heads)
                # ], dim=0).view(batch_size, num_heads, num_random_tokens)  # 变形回 (batch, heads, num_random_tokens)
                # topk_indices = torch.rand(vis_attn.shape[:-1], device=vis_attn.device).argsort(dim=-1)[:, :, :num_random_tokens]
                ##########################################################
                topk_mask = torch.zeros_like(vis_attn, dtype=torch.bool)
                topk_mask.scatter_(-1, topk_indices, True)

                if steer_mode == "non-neg":
                    attn_weights[:, :, -1, :] -= attn_weights[:, :, -1, :].min()
                attn_weights[:, :, -1, self.sys_len:self.sys_len+self.img_len][topk_mask] -= cur_config.downscale_vis * torch.abs(attn_weights[:, :, -1, self.sys_len:self.sys_len+self.img_len][topk_mask])
            elif cur_config.method_steer_vis == "hall-head":
                vis_attn = attn_weights[:, cur_config.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len]
                _, topk_indices = torch.topk(vis_attn, int(self.img_len * cur_config.topk_vis_token_for_downscale), dim=-1)
                topk_mask = torch.zeros_like(vis_attn, dtype=torch.bool)
                topk_mask.scatter_(-1, topk_indices, True)
                # print("method_steer_vis==hall-head")
                assert cur_config.hall_head_mask is not None, "You are trying to apply tp hallucination heads. But no valid head-mask-map has been set."
                # attn_weights[:, self.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len][topk_mask] -= self.downscale_vis * torch.abs(attn_weights[:, self.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len][topk_mask])
                if steer_mode == "non-neg":
                    attn_weights[:, :, -1, :] -= attn_weights[:, :, -1, :].min()
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

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self._capture_context_entropy(attn_weights)
        attn_weights = nn.functional.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, attn.num_heads, q_len, attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn.num_heads, q_len, attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = attn.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
