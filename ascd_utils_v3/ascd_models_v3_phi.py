
from typing import List, Optional, Tuple, Union
import warnings
import torch
from torch import nn
import math, re
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Optional, Sequence

from transformers.models.phi.modeling_phi import PhiForCausalLM
# from .ascd_models_v3 import AttnSteerConfig

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

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PhiAllAttentionDenoiseBase(nn.Module):
    def __init__(self,
                 original_module,
                 attn_steer_configs: Union[AttnSteerConfig, Sequence[AttnSteerConfig]]):
        super().__init__()

        self.original_module = original_module

        self.cur_config_id = 0
        self.attn_steer_configs = attn_steer_configs

        self.sys_len = 32
        self.img_len = 728

class PhiAttentionDenoiseV1(PhiAllAttentionDenoiseBase):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attn = self.original_module
        cur_config = self.attn_steer_configs[self.cur_config_id]

        if not cur_config.modify_attn:
            return attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        if attn.qk_layernorm:
            query_states = attn.q_layernorm(query_states)
            key_states = attn.k_layernorm(key_states)

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

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : attn.rotary_emb.dim],
            query_states[..., attn.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : attn.rotary_emb.dim],
            key_states[..., attn.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": attn.rotary_emb.dim}
            key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)

        # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
        attn_weights = torch.matmul(
            query_states.to(torch.float32), key_states.to(torch.float32).transpose(2, 3)
        ) / math.sqrt(attn.head_dim)

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
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, attn.num_heads, q_len, attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn.num_heads, q_len, attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

        attn_output = attn.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class PhiSdpaAttentionDenoiseV1(PhiAttentionDenoiseV1):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
