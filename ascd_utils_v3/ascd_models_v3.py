
from typing import List, Optional, Tuple, Union
import warnings
import torch
from torch import nn
import math, re
import torch.nn.functional as F

from ascd_only import build_text_enhanced_attention
from ascd_dive import head_visual_evidence, vlac_score
from ascd_vhr import reinforce_heads, select_vision_aware_heads
from ascd_clearsight import apply_vaf_logits

from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
from lavis.models.blip2_models.modeling_llama import apply_rotary_pos_emb as apply_rotary_pos_emb_from_lavis
# from transformers.cache_utils import Cache, logging
# from flash_attn import flash_attn_func

# logger = logging.get_logger(__name__)

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
        updated = obj + steer_factor * torch.abs(src)
    elif steer_mode == "non-neg":
        updated = obj + steer_factor * torch.abs(src)
    else:
        raise NotImplementedError(f"The steer mode {steer_mode} is not implemented!")
    obj.copy_(updated)

class LlamaAllAttentionDenoiseBase(nn.Module):

    def __init__(self,
                 original_module,
                 attn_steer_configs: Union[AttnSteerConfig, Sequence[AttnSteerConfig]]):
        super().__init__()

        self.original_module = original_module

        self.cur_config_id = 0
        self.attn_steer_configs = attn_steer_configs

        self.sys_len = 35
        self.img_len = 576

    def _apply_clearsight_vaf_logits(self, attn_weights: torch.Tensor, attn) -> torch.Tensor:
        """Apply released ClearSight/VAF only when an explicit per-image state exists."""
        state = getattr(self, "clearsight_state", None)
        if not isinstance(state, dict) or not bool(state.get("enabled", False)):
            return attn_weights
        layer_index = int(getattr(attn, "layer_idx", -1))
        if layer_index not in set(int(x) for x in state["target_layers"]):
            return attn_weights
        _, event = apply_vaf_logits(
            attn_weights,
            layer_index=layer_index,
            sys_len=int(self.sys_len),
            img_len=int(self.img_len),
            enhancement_multiplier=float(state["enhancement_multiplier"]),
            suppression_multiplier=float(state["suppression_multiplier"]),
        )
        state["application_count"] = int(state.get("application_count", 0)) + 1
        state.setdefault("events", []).append(event)
        return attn_weights

    def _apply_vhr(self, attn_output: torch.Tensor, attn) -> torch.Tensor:
        """Capture text-only heads or apply released per-sample VHR selection."""
        mode = getattr(self, "vhr_branch_mode", None)
        state = getattr(self, "vhr_branch_state", None)
        if mode not in {"text_contrast", "visual"} or not isinstance(state, dict):
            return attn_output
        layer_index = int(getattr(attn, "layer_idx", -1))
        if layer_index not in state["target_layers"]:
            return attn_output
        current = attn_output[0, :, -1, :].detach()
        if mode == "text_contrast":
            state.setdefault("text_head_outputs", {})[layer_index] = current
            return attn_output

        selected_by_layer = state.setdefault("selected_heads", {})
        if layer_index not in selected_by_layer:
            text_output = state.get("text_head_outputs", {}).get(layer_index)
            if text_output is None:
                raise RuntimeError(f"VHR missing text contrast at layer {layer_index}")
            selected, event = select_vision_aware_heads(
                text_output,
                current,
                apply_outlier_filter=bool(state["outlier_filter"]),
            )
            if selected.numel() == 0:
                raise RuntimeError(f"VHR selected no heads at layer {layer_index}")
            selected_by_layer[layer_index] = selected
            state.setdefault("layer_events", {})[layer_index] = event
        return reinforce_heads(
            attn_output,
            selected_by_layer[layer_index],
            augmentation_ratio=float(state["augmentation_ratio"]),
        )

    def _capture_dive_evidence(
        self,
        attn_weights: torch.Tensor,
        value_states: torch.Tensor,
        attn,
    ) -> None:
        """Capture paper-defined V-LAC and projected visual evidence when enabled."""
        if getattr(self, "dive_branch_mode", None) != "direct":
            return
        state = getattr(self, "dive_branch_state", None)
        if not isinstance(state, dict):
            return
        layer_index = int(getattr(attn, "layer_idx", -1))
        candidates = set(int(x) for x in state.get("candidate_layers", []))
        selected = state.get("selected_layers")
        target_layers = candidates if selected is None else set(int(x) for x in selected)
        if layer_index not in target_layers:
            return

        key_length = int(attn_weights.shape[-1])
        image_start = min(max(int(self.sys_len), 0), key_length)
        image_end = min(image_start + int(self.img_len), key_length)
        if image_end <= image_start:
            raise ValueError("DiVE visual-token interval is empty")
        last_attention = attn_weights[:, :, -1:, :]
        if selected is None:
            score = vlac_score(
                last_attention[0, :, 0, image_start:image_end]
            ).detach()
            state.setdefault("vlac_scores", {})[layer_index] = score

        head_evidence = head_visual_evidence(
            last_attention,
            value_states,
            image_start,
            image_end - image_start,
            epsilon=float(state.get("epsilon", 1e-6)),
        )
        bsz, _, q_len, _ = head_evidence.shape
        projected_input = head_evidence.transpose(1, 2).contiguous().reshape(
            bsz, q_len, attn.hidden_size
        )
        if attn.config.pretraining_tp > 1:
            pieces = projected_input.split(
                attn.hidden_size // attn.config.pretraining_tp, dim=2
            )
            weights = attn.o_proj.weight.split(
                attn.hidden_size // attn.config.pretraining_tp, dim=1
            )
            projected = sum(F.linear(pieces[i], weights[i]) for i in range(len(weights)))
        else:
            projected = attn.o_proj(projected_input)
        state.setdefault("evidence", {})[layer_index] = projected[:, -1, :].detach()

    def _apply_crops_attention_mask(
        self, attn_weights: torch.Tensor, attn
    ) -> torch.Tensor:
        """Apply a shared CRoPS key mask from the frozen aggregate layer onward."""
        mode = getattr(self, "crops_branch_mode", None)
        state = getattr(self, "crops_branch_state", None)
        if mode not in {"visual_statistical", "language_prior"} or not isinstance(state, dict):
            return attn_weights
        layer_index = int(getattr(attn, "layer_idx", -1))
        aggregate_layer = int(state.get("aggregate_layer", 2))
        key_mask = state.get("key_mask")
        if layer_index < aggregate_layer or key_mask is None:
            return attn_weights
        key_length = int(attn_weights.shape[-1])
        if int(key_mask.shape[-1]) != key_length:
            raise ValueError(
                f"CRoPS key-mask length mismatch: mask={key_mask.shape[-1]} key={key_length}"
            )
        return attn_weights.masked_fill(
            ~key_mask[:, None, None, :].to(attn_weights.device),
            torch.finfo(attn_weights.dtype).min,
        )

    def _capture_crops_attention_mask(
        self, attn_weights: torch.Tensor, attn
    ) -> None:
        """Build the released bottom-attention mask from layer aggregate_layer-1."""
        mode = getattr(self, "crops_branch_mode", None)
        state = getattr(self, "crops_branch_state", None)
        if mode not in {"visual_statistical", "language_prior"} or not isinstance(state, dict):
            return
        aggregate_layer = int(state.get("aggregate_layer", 2))
        if int(getattr(attn, "layer_idx", -1)) != aggregate_layer - 1:
            return
        last_attention = attn_weights.mean(dim=1)[0, -1].detach()
        key_length = int(last_attention.shape[-1])
        if mode == "visual_statistical":
            image_start = min(max(int(self.sys_len), 0), key_length)
            image_end = min(image_start + int(self.img_len), key_length)
            image_length = image_end - image_start
            if image_length <= 0:
                raise ValueError("CRoPS visual branch has no image tokens")
            keep = max(1, round(float(state.get("visual_keep_fraction", 0.25)) * image_length))
            selected = last_attention[image_start:image_end].topk(
                keep, largest=False
            ).indices + image_start
            key_mask = torch.ones(
                (1, key_length), dtype=torch.bool, device=attn_weights.device
            )
            key_mask[:, image_start:image_end] = False
            key_mask[:, selected] = True
        else:
            keep = min(max(1, int(state["minimum_text_tokens"])), key_length)
            selected = last_attention.topk(keep, largest=False).indices
            key_mask = torch.zeros(
                (1, key_length), dtype=torch.bool, device=attn_weights.device
            )
            key_mask[:, selected] = True
        state["key_mask"] = key_mask
        state["selected_key_count"] = int(key_mask.sum().item())
        state["key_length"] = key_length

    def _capture_mole_prompt_mass(self, attn_weights: torch.Tensor) -> None:
        """Capture MoLE's last-head attention mass over the original prompt."""
        if not bool(getattr(self, "mole_capture_enabled", False)):
            return
        prompt_end = int(getattr(self, "mole_prompt_end", -1))
        key_length = int(attn_weights.shape[-1])
        if not 1 <= prompt_end <= key_length:
            raise ValueError(
                f"invalid MoLE prompt boundary: end={prompt_end} key={key_length}"
            )
        mass = attn_weights[0, -1, -1, :prompt_end].detach().float().sum()
        if not hasattr(self, "_mole_prompt_mass_by_config"):
            self._mole_prompt_mass_by_config = {}
        self._mole_prompt_mass_by_config[int(self.cur_config_id)] = mass

    def _allpath_project(self, attn_output: torch.Tensor, attn):
        """Apply the released AllPath per-head o_proj weighting exactly."""
        if not bool(getattr(self, "allpath_enabled", False)):
            return None
        hallu = [int(value) for value in self.allpath_hallu_heads]
        good = [int(value) for value in self.allpath_good_heads]
        overlap = set(hallu) & set(good)
        if overlap:
            hallu = [value for value in hallu if value not in overlap]
            good = [value for value in good if value not in overlap]
        bsz, num_heads, q_len, head_dim = attn_output.shape
        if num_heads != attn.num_heads or head_dim != attn.head_dim:
            raise ValueError("AllPath attention shape is inconsistent with the module")
        slices = attn.o_proj.weight.T.reshape(num_heads, head_dim, attn.hidden_size)
        projected = attn_output @ slices
        scale = torch.ones(
            bsz, num_heads, q_len, device=projected.device, dtype=projected.dtype
        )
        scale[:, hallu, :] = float(self.allpath_de_scale)
        scale[:, good, :] = float(self.allpath_in_scale)
        return (scale.unsqueeze(-1) * projected).sum(dim=1)

    def _capture_only_branch(
        self,
        attn_weights: torch.Tensor,
        value_states: torch.Tensor,
        attn,
    ) -> None:
        """Capture ONLY's inert side branch without changing normal attention."""
        if not bool(getattr(self, "only_capture_enabled", False)):
            return
        if int(getattr(attn, "layer_idx", -1)) != int(
            getattr(self, "only_layer_index", 0)
        ):
            return
        head_output, info = build_text_enhanced_attention(
            attn_weights,
            value_states,
            image_start=self.sys_len,
            image_length=self.img_len,
        )
        bsz, _, q_len, _ = head_output.shape
        projected_input = head_output.transpose(1, 2).contiguous().reshape(
            bsz, q_len, attn.hidden_size
        )
        if attn.config.pretraining_tp > 1:
            pieces = projected_input.split(
                attn.hidden_size // attn.config.pretraining_tp, dim=2
            )
            weights = attn.o_proj.weight.split(
                attn.hidden_size // attn.config.pretraining_tp, dim=1
            )
            projected = sum(F.linear(pieces[i], weights[i]) for i in range(len(weights)))
        else:
            projected = attn.o_proj(projected_input)
        if not hasattr(self, "_only_branch_by_config"):
            self._only_branch_by_config = {}
        self._only_branch_by_config[int(self.cur_config_id)] = {
            "attention_output": projected,
            "entropy_ratios": info["entropy_ratios"],
            "removed_heads": info["removed_heads"],
        }

    def _capture_attention_diagnostics(self, attn_weights: torch.Tensor) -> None:
        """Cache compact last-query attention statistics when requested.

        Full token diagnostics retain four layerwise statistics.  The visual
        preservation gate only retains image-token mass, avoiding entropy and
        host synchronization in ordinary method runs.  Both paths are absent
        from the original ASCD configuration and therefore default to off.
        """
        diagnostics_enabled = bool(getattr(self, "diagnostics_enabled", False))
        visual_preservation_enabled = bool(
            getattr(self, "visual_preservation_enabled", False)
        )
        context_entropy_enabled = bool(
            getattr(self, "context_entropy_capture_enabled", False)
        ) and int(getattr(self.original_module, "layer_idx", -1)) == int(
            getattr(self, "context_entropy_layer_id", -2)
        )
        if not (diagnostics_enabled or visual_preservation_enabled or context_entropy_enabled):
            return

        last_query = attn_weights[:, :, -1, :].detach().float()
        key_len = last_query.shape[-1]
        image_start = min(max(int(self.sys_len), 0), key_len)
        image_end = min(max(image_start + int(self.img_len), image_start), key_len)

        image_weights = last_query[..., image_start:image_end]
        image_mass = image_weights.sum(dim=-1)

        if context_entropy_enabled:
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
            visual_mean = image_weights.mean(dim=-1)
            instruction_mean = torch.cat(instruction_parts, dim=-1).mean(dim=-1)
            if candidate_position > instruction_end:
                history_mean = last_query[..., instruction_end:candidate_position].mean(dim=-1)
            else:
                history_mean = torch.zeros_like(visual_mean)
            self._context_entropy_components_by_config = {
                int(self.cur_config_id): torch.stack(
                    (visual_mean[0], instruction_mean[0], history_mean[0]), dim=-1
                ).detach()
            }

        payload = {
            "layer_id": int(getattr(self.original_module, "layer_idx", -1)),
            "image_mass": image_mass.mean(dim=-1),
        }
        if diagnostics_enabled:
            system_mass = last_query[..., :image_start].sum(dim=-1)
            history_mass = last_query[..., image_end:].sum(dim=-1)
            if image_weights.shape[-1] > 1:
                normalized_image = image_weights / image_mass.unsqueeze(-1).clamp_min(1e-12)
                image_entropy = -(
                    normalized_image * normalized_image.clamp_min(1e-12).log()
                ).sum(dim=-1) / math.log(image_weights.shape[-1])
            else:
                image_entropy = torch.zeros_like(image_mass)
            payload.update(
                {
                    "system_mass": system_mass.mean(dim=-1),
                    "history_mass": history_mass.mean(dim=-1),
                    "image_entropy": image_entropy.mean(dim=-1),
                }
            )
        if not (diagnostics_enabled or visual_preservation_enabled):
            return
        if not hasattr(self, "_diagnostic_attn_by_config"):
            self._diagnostic_attn_by_config = {}
        self._diagnostic_attn_by_config[int(self.cur_config_id)] = payload

class LlamaAttentionDenoiseV1(LlamaAllAttentionDenoiseBase):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attn = self.original_module
        bsz, q_len, _ = hidden_states.size()

        if attn.config.pretraining_tp > 1:
            key_value_slicing = (attn.num_key_value_heads * attn.head_dim) // attn.config.pretraining_tp
            query_slices = attn.q_proj.weight.split(
                (attn.num_heads * attn.head_dim) // attn.config.pretraining_tp, dim=0
            )
            key_slices = attn.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = attn.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(attn.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(attn.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(attn.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

        past_key_value = getattr(attn, "past_key_value", past_key_value)
        cos, sin = attn.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)
        attn_weights = self._apply_clearsight_vaf_logits(attn_weights, attn)

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

        # upcast attention to fp32
        attn_weights = self._apply_crops_attention_mask(attn_weights, attn)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self._capture_crops_attention_mask(attn_weights, attn)
        self._capture_only_branch(attn_weights, value_states, attn)
        self._capture_dive_evidence(attn_weights, value_states, attn)
        self._capture_mole_prompt_mass(attn_weights)
        self._capture_attention_diagnostics(attn_weights)
        attn_weights = nn.functional.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self._apply_vhr(attn_output, attn)

        if attn_output.size() != (bsz, attn.num_heads, q_len, attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn.num_heads, q_len, attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        allpath_output = self._allpath_project(attn_output, attn)
        if allpath_output is not None:
            attn_output = allpath_output
            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights, past_key_value

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

        if attn.config.pretraining_tp > 1:
            attn_output = attn_output.split(attn.hidden_size // attn.config.pretraining_tp, dim=2)
            o_proj_slices = attn.o_proj.weight.split(attn.hidden_size // attn.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(attn.config.pretraining_tp)])
        else:
            attn_output = attn.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaAttentionDenoiseV2(LlamaAllAttentionDenoiseBase):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attn = self.original_module
        bsz, q_len, _ = hidden_states.size()

        if attn.config.pretraining_tp > 1:
            key_value_slicing = (attn.num_key_value_heads * attn.head_dim) // attn.config.pretraining_tp
            query_slices = attn.q_proj.weight.split(
                (attn.num_heads * attn.head_dim) // attn.config.pretraining_tp, dim=0
            )
            key_slices = attn.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = attn.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(attn.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(attn.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(attn.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

        past_key_value = getattr(attn, "past_key_value", past_key_value)
        cos, sin = attn.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)
        attn_weights = self._apply_clearsight_vaf_logits(attn_weights, attn)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        ##### modification #####
        cur_config = self.attn_steer_configs[self.cur_config_id]
        if cur_config.modify_attn:
            if cur_config.method_steer_sys == "all":
                attn_weights[:, :, -1, :self.sys_len] += cur_config.upscale_sys * torch.relu(attn_weights[:, :, -1, :self.sys_len])
            elif cur_config.method_steer_sys == "hall-head":
                assert cur_config.hall_head_mask is not None, "You are trying to apply tp hallucination heads. But no valid head-mask-map has been set."
                attn_weights[:, cur_config.hall_head_mask, -1, :self.sys_len] += cur_config.upscale_sys * torch.relu(attn_weights[:, cur_config.hall_head_mask, -1, :self.sys_len])
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

                attn_weights[:, :, -1, self.sys_len:self.sys_len+self.img_len][topk_mask] -= cur_config.downscale_vis * torch.relu(attn_weights[:, :, -1, self.sys_len:self.sys_len+self.img_len][topk_mask])
            elif cur_config.method_steer_vis == "hall-head":
                vis_attn = attn_weights[:, cur_config.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len]
                _, topk_indices = torch.topk(vis_attn, int(self.img_len * cur_config.topk_vis_token_for_downscale), dim=-1)
                topk_mask = torch.zeros_like(vis_attn, dtype=torch.bool)
                topk_mask.scatter_(-1, topk_indices, True)
                # print("method_steer_vis==hall-head")
                assert cur_config.hall_head_mask is not None, "You are trying to apply tp hallucination heads. But no valid head-mask-map has been set."
                # attn_weights[:, self.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len][topk_mask] -= self.downscale_vis * torch.abs(attn_weights[:, self.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len][topk_mask])
                attn_weights[:, cur_config.hall_head_mask, -1, self.sys_len:self.sys_len+self.img_len] -= (
                    cur_config.downscale_vis * torch.relu(vis_attn) * topk_mask
                )
            elif cur_config.method_steer_vis == "none":
                pass
            else:
                raise ValueError(f"method_steer_vis is {cur_config.method_steer_vis}, should be one of ['all', 'hall-head', 'none']")

            if cur_config.method_steer_text == "all":
                attn_weights[:, :, -1, self.sys_len+self.img_len:] += cur_config.upscale_text * torch.relu(attn_weights[:, :, -1, self.sys_len+self.img_len:])
            elif cur_config.method_steer_text == "hall-head":
                # print("method_steer_text==hall-head")
                assert cur_config.hall_head_mask is not None, "You are trying to apply tp hallucination heads. But no valid head-mask-map has been set."
                attn_weights[:, cur_config.hall_head_mask, -1, self.sys_len+self.img_len:] += cur_config.upscale_text * torch.relu(attn_weights[:, cur_config.hall_head_mask, -1, self.sys_len+self.img_len:])
            elif cur_config.method_steer_text == "none":
                pass
            else:
                raise ValueError(f"method_steer_text is {cur_config.method_steer_text}, should be one of ['all', 'hall-head', 'none']")

        ########################

        # upcast attention to fp32
        attn_weights = self._apply_crops_attention_mask(attn_weights, attn)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self._capture_crops_attention_mask(attn_weights, attn)
        self._capture_only_branch(attn_weights, value_states, attn)
        self._capture_dive_evidence(attn_weights, value_states, attn)
        self._capture_mole_prompt_mass(attn_weights)
        self._capture_attention_diagnostics(attn_weights)
        attn_weights = nn.functional.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self._apply_vhr(attn_output, attn)

        if attn_output.size() != (bsz, attn.num_heads, q_len, attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn.num_heads, q_len, attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        allpath_output = self._allpath_project(attn_output, attn)
        if allpath_output is not None:
            attn_output = allpath_output
            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights, past_key_value

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

        if attn.config.pretraining_tp > 1:
            attn_output = attn_output.split(attn.hidden_size // attn.config.pretraining_tp, dim=2)
            o_proj_slices = attn.o_proj.weight.split(attn.hidden_size // attn.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(attn.config.pretraining_tp)])
        else:
            attn_output = attn.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# class LlamaAttentionDenoiseV3(LlamaAllAttentionDenoiseBase):

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         attn = self.original_module
#         bsz, q_len, _ = hidden_states.size()

#         if attn.config.pretraining_tp > 1:
#             key_value_slicing = (attn.num_key_value_heads * attn.head_dim) // attn.config.pretraining_tp
#             query_slices = attn.q_proj.weight.split(
#                 (attn.num_heads * attn.head_dim) // attn.config.pretraining_tp, dim=0
#             )
#             key_slices = attn.k_proj.weight.split(key_value_slicing, dim=0)
#             value_slices = attn.v_proj.weight.split(key_value_slicing, dim=0)

#             query_states = [F.linear(hidden_states, query_slices[i]) for i in range(attn.config.pretraining_tp)]
#             query_states = torch.cat(query_states, dim=-1)

#             key_states = [F.linear(hidden_states, key_slices[i]) for i in range(attn.config.pretraining_tp)]
#             key_states = torch.cat(key_states, dim=-1)

#             value_states = [F.linear(hidden_states, value_slices[i]) for i in range(attn.config.pretraining_tp)]
#             value_states = torch.cat(value_states, dim=-1)

#         else:
#             query_states = attn.q_proj(hidden_states)
#             key_states = attn.k_proj(hidden_states)
#             value_states = attn.v_proj(hidden_states)

#         query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

#         past_key_value = getattr(attn, "past_key_value", past_key_value)
#         cos, sin = attn.rotary_emb(value_states, position_ids)
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

#         key_states = repeat_kv(key_states, attn.num_key_value_groups)
#         value_states = repeat_kv(value_states, attn.num_key_value_groups)

#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

#         if attention_mask is not None:  # no matter the length, we just slice it
#             causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#             attn_weights = attn_weights + causal_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, attn.num_heads, q_len, attn.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, attn.num_heads, q_len, attn.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()

#         attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

#         if attn.config.pretraining_tp > 1:
#             attn_output = attn_output.split(attn.hidden_size // attn.config.pretraining_tp, dim=2)
#             o_proj_slices = attn.o_proj.weight.split(attn.hidden_size // attn.config.pretraining_tp, dim=1)
#             attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(attn.config.pretraining_tp)])
#         else:
#             attn_output = attn.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value



# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb_from_lavis(q, k, cos, sin, position_ids):
#     gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
#     gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
#     cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
#     sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


class LlamaAttentionDenoiseV1InLavis(LlamaAllAttentionDenoiseBase):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attn = self.original_module

        bsz, q_len, _ = hidden_states.size()

        query_states = attn.q_proj(hidden_states).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states = attn.k_proj(hidden_states).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb_from_lavis(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)
        attn_weights = self._apply_clearsight_vaf_logits(attn_weights, attn)

        if attn_weights.size() != (bsz, attn.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * attn.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

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
        attn_weights = self._apply_crops_attention_mask(attn_weights, attn)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self._capture_crops_attention_mask(attn_weights, attn)
        self._capture_mole_prompt_mass(attn_weights)
        self._capture_attention_diagnostics(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, attn.num_heads, q_len, attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn.num_heads, q_len, attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

        attn_output = attn.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
