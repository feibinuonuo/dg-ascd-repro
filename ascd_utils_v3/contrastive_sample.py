import copy
import inspect
import json
import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

import transformers
# from transformers.cache_utils import Cache, DynamicCache, StaticCache
# from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from transformers.utils import ModelOutput, logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
# from transformers.generation.candidate_generator import (
#     AssistedCandidateGenerator,
#     CandidateGenerator,
#     PromptLookupCandidateGenerator,
#     _crop_past_key_values,
#     _prepare_attention_mask,
#     _prepare_token_type_ids,
# )
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateBeamOutput,
    _split_model_inputs,
    stack_model_outputs,
    GenerateBeamEncoderDecoderOutput,
    GenerateBeamDecoderOnlyOutput
)

from . import *
from ascd_context_entropy import (
    calibrate_top_candidates,
    contextual_entropy_from_component_means,
)
from ascd_deco import deco_calibrate
from ascd_only import build_text_enhanced_logits, only_calibrate
from ascd_mole import mole_calibrate
from ascd_sumgd import IMAGE_POS, find_aligned_word, generate_pos_tags
from ascd_mfcd import mfcd_calibrate
from ascd_inter import inter_calibrate
from ascd_fuzzycd import fuzzycd_calibrate
from ascd_crops_runtime import greedy_search as crops_greedy_search
from ascd_cei_runtime import greedy_search as cei_greedy_search, injection as cei_injection
from ascd_dive_runtime import greedy_search as dive_greedy_search
from ascd_vhr import target_layer_indices
from ascd_selfaug_runtime import greedy_search as selfaug_greedy_search
from ascd_vista_runtime import greedy_search as vista_greedy_search
from ascd_verifier_constrained import (
    build_audit_record,
    decide_verifier_constraint,
    terminal_chair_object,
)
from ascd_detector_grounded import apply_detector_object_mask, terminal_decoded_chair_object
from ascd_soft_grounded import apply_soft_grounded_object_penalty
from ascd_alias_guard import observe_alias_guard_candidates
from ascd_detector_comparative import (
    build_comparative_audit_record,
    decide_comparative_reversion,
)


@contextmanager
def _sumgd_unmodified_attention(model):
    """Temporarily disable ASCD attention changes for SumGD's own expert."""
    saved = []
    seen = set()
    for module in model.modules():
        configs = getattr(module, "attn_steer_configs", None)
        if configs is None:
            continue
        for config in configs:
            if id(config) in seen:
                continue
            seen.add(id(config))
            saved.append((config, bool(getattr(config, "modify_attn", False))))
            config.modify_attn = False
    try:
        yield
    finally:
        for config, value in saved:
            config.modify_attn = value


def _sumgd_embeddings(self, base_embeddings, generated_ids):
    if generated_ids:
        token_ids = torch.tensor(
            [generated_ids], device=base_embeddings.device, dtype=torch.long
        )
        token_embeddings = self.get_input_embeddings()(token_ids)
        return torch.cat((base_embeddings, token_embeddings), dim=1)
    return base_embeddings


def _vhr_wrappers(model):
    return [module for module in model.modules() if hasattr(module, "attn_steer_configs")]


def _initialize_vhr(self, model_kwargs):
    """Released one-time text-contrast pass followed by fixed per-layer heads."""
    if not bool(getattr(self.cd_config, "vhr_enabled", False)):
        return None
    if bool(getattr(self.cd_config, "if_cd", False)):
        raise ValueError("Official VHR-only requires contrastive decoding disabled")
    if "inputs_embeds" not in model_kwargs:
        raise ValueError("VHR requires expanded multimodal embeddings")
    wrappers = _vhr_wrappers(self.model)
    if not wrappers:
        raise RuntimeError("VHR found no attention wrappers")
    total_layers = len(wrappers)
    target_layers = target_layer_indices(
        total_layers,
        last_layers=int(self.cd_config.vhr_last_layers),
        include_layer_one=bool(self.cd_config.vhr_include_layer_one),
    )
    image_start = int(wrappers[0].sys_len)
    image_length = int(wrappers[0].img_len)
    embeddings = model_kwargs["inputs_embeds"]
    image_end = image_start + image_length
    if not 0 <= image_start < image_end <= embeddings.shape[1]:
        raise ValueError("VHR visual-token span is invalid")
    text_embeddings = torch.cat(
        (embeddings[:, :image_start], embeddings[:, image_end:]), dim=1
    )
    state = {
        "target_layers": target_layers,
        "augmentation_ratio": float(self.cd_config.vhr_augmentation_ratio),
        "outlier_filter": bool(self.cd_config.vhr_outlier_filter),
        "text_head_outputs": {},
        "selected_heads": {},
        "layer_events": {},
    }
    for wrapper in wrappers:
        wrapper.cur_config_id = 0
        wrapper.vhr_branch_mode = "text_contrast"
        wrapper.vhr_branch_state = state
    self(
        inputs_embeds=text_embeddings,
        attention_mask=torch.ones(
            text_embeddings.shape[:2], device=text_embeddings.device, dtype=torch.long
        ),
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    if set(state["text_head_outputs"]) != set(target_layers):
        raise RuntimeError("VHR text contrast did not capture every target layer")
    for wrapper in wrappers:
        wrapper.vhr_branch_mode = "visual"
    return state


def _finalize_vhr(self, state, generated_tokens):
    if state is None:
        return
    if set(state["selected_heads"]) != set(state["target_layers"]):
        raise RuntimeError("VHR visual pass did not select heads for every target layer")
    self.cd_config.vhr_records.append({
        "image_id": int(self.cd_config.vhr_image_id),
        "parent": "unmodified",
        "target_layers": list(state["target_layers"]),
        "selected_heads": {
            str(layer): [int(value) for value in state["selected_heads"][layer].tolist()]
            for layer in state["target_layers"]
        },
        "layer_events": {
            str(layer): state["layer_events"][layer]
            for layer in state["target_layers"]
        },
        "generated_tokens": int(generated_tokens),
        "text_contrast_forwards": 1,
    })
    for wrapper in _vhr_wrappers(self.model):
        wrapper.vhr_branch_mode = None
        wrapper.vhr_branch_state = None


def _sumgd_forward(self, embeddings, config_id=0, unmodified=False):
    switch_attn_steer_id(self.model, int(config_id))
    attention_mask = torch.ones(
        embeddings.shape[:2], device=embeddings.device, dtype=torch.long
    )
    if unmodified:
        with _sumgd_unmodified_attention(self.model):
            return self(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
    return self(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )


def _sumgd_parent_logits(self, embeddings):
    positive = _sumgd_forward(self, embeddings, config_id=0)
    positive_logits = positive.logits[:, -1, :]
    if not bool(getattr(self.cd_config, "if_cd", False)):
        return positive_logits
    negative = _sumgd_forward(self, embeddings, config_id=1)
    negative_logits = negative.logits[:, -1, :]
    alpha = get_contrastive_alpha(self.cd_config, positive_logits, model=self.model)
    beta = self.cd_config.cd_beta if self.cd_config.cd_beta is not None else 0.1
    cutoff = math.log(float(beta)) + positive_logits.max(dim=-1, keepdim=True).values
    return ((1 + alpha) * positive_logits - alpha * negative_logits).masked_fill(
        positive_logits < cutoff, -float("inf")
    )


def _cei_context_embedding(self, embeddings):
    """Extract the final-layer last-prompt representation from the positive branch."""
    switch_attn_steer_id(self.model, 0)
    attention_mask = torch.ones(
        embeddings.shape[:2], device=embeddings.device, dtype=torch.long
    )
    output = self(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=True,
    )
    return output.hidden_states[-1][:, -1, :]


def _cei_parent_logits(
    self, embeddings, context_embedding, injection_layer, cei_alpha, event_sink
):
    """Apply CEI to the direct branch and optionally retain Fixed ASCD contrast."""
    switch_attn_steer_id(self.model, 0)
    attention_mask = torch.ones(
        embeddings.shape[:2], device=embeddings.device, dtype=torch.long
    )
    with cei_injection(
        self.model, context_embedding, injection_layer, cei_alpha, event_sink
    ):
        positive_logits = self(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        ).logits[:, -1, :]
    if not bool(getattr(self.cd_config, "if_cd", False)):
        return positive_logits

    switch_attn_steer_id(self.model, 1)
    negative_logits = self(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    ).logits[:, -1, :]
    alpha = get_contrastive_alpha(self.cd_config, positive_logits, model=self.model)
    beta = self.cd_config.cd_beta if self.cd_config.cd_beta is not None else 0.1
    cutoff = math.log(float(beta)) + positive_logits.max(dim=-1, keepdim=True).values
    return ((1 + alpha) * positive_logits - alpha * negative_logits).masked_fill(
        positive_logits < cutoff, -float("inf")
    )


def _dive_parent_logits(self, positive_logits, embeddings):
    """Retain the supplied DiVE positive branch or add Fixed-ASCD contrast."""
    if not bool(getattr(self.cd_config, "if_cd", False)):
        return positive_logits
    negative_logits = _sumgd_forward(self, embeddings, config_id=1).logits[:, -1, :]
    alpha = get_contrastive_alpha(self.cd_config, positive_logits, model=self.model)
    beta = self.cd_config.cd_beta if self.cd_config.cd_beta is not None else 0.1
    cutoff = math.log(float(beta)) + positive_logits.max(dim=-1, keepdim=True).values
    return ((1 + alpha) * positive_logits - alpha * negative_logits).masked_fill(
        positive_logits < cutoff, -float("inf")
    )


def _sumgd_generate_summary(self, tokenizer, caption, eos_token_ids, max_tokens):
    prompt = f"USER: Summarize the following caption briefly.\nCaption: {caption} ASSISTANT:"
    summary_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(next(self.parameters()).device)
    prompt_length = int(summary_ids.shape[-1])
    generated = []
    terminated_by_eos = False
    for _ in range(int(max_tokens)):
        embeddings = self.get_input_embeddings()(summary_ids)
        output = _sumgd_forward(self, embeddings, config_id=0, unmodified=True)
        token = int(torch.argmax(output.logits[:, -1, :], dim=-1)[0].item())
        summary_ids = torch.cat(
            (summary_ids, torch.tensor([[token]], device=summary_ids.device)), dim=-1
        )
        generated.append(token)
        if token in eos_token_ids:
            terminated_by_eos = True
            break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    retokenized = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids[0].tolist()
    return retokenized, text, prompt_length, len(generated), terminated_by_eos


def _sumgd_greedy_search(
    self,
    input_ids,
    logits_processor,
    stopping_criteria,
    pad_token_id,
    eos_token_id,
    streamer,
    model_kwargs,
):
    """Faithful, cache-free SumGD-S state machine for batch-size one."""
    if input_ids.shape[0] != 1:
        raise ValueError("SumGD currently requires batch_size=1")
    if "inputs_embeds" not in model_kwargs:
        raise ValueError("SumGD requires the expanded multimodal prompt embeddings")
    tokenizer = getattr(self.cd_config, "sumgd_tokenizer", None)
    if tokenizer is None:
        raise ValueError("SumGD tokenizer is not configured")
    base_embeddings = model_kwargs["inputs_embeds"]
    max_new_tokens = int(getattr(self.cd_config, "sumgd_max_new_tokens", 512))
    max_summary_tokens = int(getattr(self.cd_config, "sumgd_max_summary_tokens", 128))
    eos_ids = [int(eos_token_id)] if isinstance(eos_token_id, int) else [int(x) for x in eos_token_id]
    eos_set = set(eos_ids)

    generated = []
    first_end = -1
    target_index = -1
    summary_checkpoint = None
    summary_ids = []
    summary_text = ""
    summary_events = []
    interventions = []
    parent_forwards = 0
    rollback_tokens = 0
    target_checks = 0

    while len(generated) < max_new_tokens:
        embeddings = _sumgd_embeddings(self, base_embeddings, generated)
        parent_logits = _sumgd_parent_logits(self, embeddings)
        parent_forwards += 1
        output_prefix = torch.cat(
            (
                input_ids,
                torch.tensor([generated], device=input_ids.device, dtype=torch.long),
            ),
            dim=-1,
        )
        parent_scores = logits_processor(output_prefix, parent_logits)
        candidate = int(torch.argmax(parent_scores, dim=-1)[0].item())
        generated.append(candidate)

        tagged = generate_pos_tags(tokenizer, generated)
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if first_end < 0:
            if decoded.endswith(".") and tagged and tagged[-1][0]:
                first_end = int(tagged[-1][0][0])
                target_index = first_end
        else:
            match = find_aligned_word(tagged, target_index)
            if match is None:
                target_index += 1
            else:
                word_index, aligned, word, pos = match
                enough_future = word_index + 2 < len(tagged)
                at_eos = candidate in eos_set
                if enough_future or at_eos:
                    target_checks += 1
                    original_token = int(generated[target_index])
                    if pos in IMAGE_POS:
                        if summary_checkpoint != first_end:
                            completed = tokenizer.decode(
                                generated[: first_end + 1], skip_special_tokens=True
                            )
                            (
                                summary_ids,
                                summary_text,
                                prompt_tokens,
                                summary_tokens,
                                summary_terminated_by_eos,
                            ) = (
                                _sumgd_generate_summary(
                                    self, tokenizer, completed, eos_set, max_summary_tokens
                                )
                            )
                            summary_checkpoint = first_end
                            summary_events.append({
                                "sentence_end_token_index": int(first_end),
                                "completed_caption": completed,
                                "summary": summary_text,
                                "summary_prompt_tokens": int(prompt_tokens),
                                "summary_generation_tokens": int(summary_tokens),
                                "terminated_by_eos": bool(summary_terminated_by_eos),
                                "truncated": not bool(summary_terminated_by_eos),
                            })
                        current_sentence = generated[first_end + 1 : target_index]
                        context_ids = summary_ids + current_sentence
                        summary_embeddings = _sumgd_embeddings(
                            self, base_embeddings, context_ids
                        )
                        summary_output = _sumgd_forward(
                            self, summary_embeddings, config_id=0, unmodified=True
                        )
                        summary_logits = summary_output.logits[:, -1, :]
                        prefix_before_target = torch.cat(
                            (
                                input_ids,
                                torch.tensor(
                                    [generated[:target_index]],
                                    device=input_ids.device,
                                    dtype=torch.long,
                                ),
                            ),
                            dim=-1,
                        )
                        replacement = int(
                            torch.argmax(
                                logits_processor(prefix_before_target, summary_logits),
                                dim=-1,
                            )[0].item()
                        )
                        removed = len(generated) - target_index
                        rollback_tokens += removed
                        generated = generated[:target_index] + [replacement]
                        interventions.append({
                            "token_index": int(target_index),
                            "word": word,
                            "pos": pos,
                            "aligned_token_indices": [int(x) for x in aligned],
                            "original_token_id": original_token,
                            "summary_token_id": replacement,
                            "changed": original_token != replacement,
                            "rolled_back_tokens": int(removed),
                            "sentence_end_token_index": int(first_end),
                        })
                        if tokenizer.convert_ids_to_tokens(replacement) == ".":
                            first_end = target_index
                        target_index += 1
                        if streamer is not None:
                            streamer.put(torch.tensor([replacement]).cpu())
                        continue
                    if word == ".":
                        first_end = int(aligned[0])
                    target_index += 1

        if candidate in eos_set:
            break
        if streamer is not None:
            streamer.put(torch.tensor([candidate]).cpu())

    if streamer is not None:
        streamer.end()
    result = torch.cat(
        (
            input_ids,
            torch.tensor([generated], device=input_ids.device, dtype=torch.long),
        ),
        dim=-1,
    )
    if stopping_criteria(result, None).any() and generated and generated[-1] not in eos_set:
        pass
    record = {
        "image_id": int(getattr(self.cd_config, "sumgd_image_id")),
        "parent": (
            "unmodified" if not bool(getattr(self.cd_config, "if_cd", False))
            else "adaptive_recall_ascd" if bool(getattr(self.cd_config, "adaptive_alpha", False))
            else "fixed_ascd"
        ),
        "generated_tokens": len(generated),
        "parent_forward_steps": parent_forwards,
        "target_checks": target_checks,
        "rollback_tokens": rollback_tokens,
        "summary_events": summary_events,
        "interventions": interventions,
        "intervention_count": len(interventions),
        "changed_intervention_count": sum(event["changed"] for event in interventions),
    }
    if not hasattr(self.cd_config, "sumgd_records"):
        self.cd_config.sumgd_records = []
    self.cd_config.sumgd_records.append(record)
    return result


def _mfcd_greedy_search(
    self,
    input_ids,
    logits_processor,
    stopping_criteria,
    eos_token_id,
    streamer,
    model_kwargs,
):
    """Cache-free synchronized three-visual-branch MFCD greedy decoding."""
    if input_ids.shape[0] != 1:
        raise ValueError("MFCD currently requires batch_size=1")
    required = ("inputs_embeds", "inputs_embeds_mfcd_high", "inputs_embeds_mfcd_low")
    if any(name not in model_kwargs for name in required):
        raise ValueError("MFCD requires original, high-pass, and low-pass embeddings")
    bases = [model_kwargs[name] for name in required]
    if len({tuple(value.shape) for value in bases}) != 1:
        raise ValueError("MFCD visual branches have inconsistent embedding shapes")
    max_new_tokens = int(getattr(self.cd_config, "mfcd_max_new_tokens", 512))
    eos_ids = [int(eos_token_id)] if isinstance(eos_token_id, int) else [int(x) for x in eos_token_id]
    eos_set = set(eos_ids)
    generated = []
    events = []

    while len(generated) < max_new_tokens:
        branch_logits = []
        for base in bases:
            embeddings = _sumgd_embeddings(self, base, generated)
            branch_logits.append(_sumgd_parent_logits(self, embeddings))
        calibrated, event = mfcd_calibrate(
            branch_logits[0], branch_logits[1], branch_logits[2],
            high_alpha=float(getattr(self.cd_config, "mfcd_high_alpha", 1.0)),
            low_alpha=float(getattr(self.cd_config, "mfcd_low_alpha", 1.0)),
            beta=float(getattr(self.cd_config, "mfcd_beta", 0.3)),
        )
        prefix = torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device, dtype=torch.long)),
            dim=-1,
        )
        scores = logits_processor(prefix, calibrated)
        token = int(torch.argmax(scores, dim=-1)[0].item())
        event.update({
            "step": len(generated),
            "selected_token_id": token,
            "selected_changed_from_original_top": token != event["original_top_token_id"],
        })
        events.append(event)
        generated.append(token)
        if streamer is not None:
            streamer.put(torch.tensor([token]).cpu())
        result = torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device, dtype=torch.long)),
            dim=-1,
        )
        if token in eos_set or stopping_criteria(result, None).any():
            break

    if streamer is not None:
        streamer.end()
    record = {
        "image_id": int(getattr(self.cd_config, "mfcd_image_id")),
        "parent": "fixed_ascd" if bool(getattr(self.cd_config, "if_cd", False)) else "unmodified",
        "generated_tokens": len(generated),
        "forward_calls": len(generated) * (6 if bool(getattr(self.cd_config, "if_cd", False)) else 3),
        "changed_steps": sum(event["selected_changed_from_original_top"] for event in events),
        "events": events,
    }
    self.cd_config.mfcd_records.append(record)
    return result


def _inter_greedy_search(
    self, input_ids, logits_processor, stopping_criteria, eos_token_id,
    streamer, model_kwargs,
):
    """Cache-free synchronized four-coalition INTER greedy decoding."""
    required = (
        "inputs_embeds", "inputs_embeds_inter_random_image",
        "inputs_embeds_inter_empty_text", "inputs_embeds_inter_random_empty",
    )
    if input_ids.shape[0] != 1 or any(name not in model_kwargs for name in required):
        raise ValueError("INTER requires batch one and all four coalition embeddings")
    bases = [model_kwargs[name] for name in required]
    max_new_tokens = int(getattr(self.cd_config, "inter_max_new_tokens", 512))
    eos_ids = [int(eos_token_id)] if isinstance(eos_token_id, int) else [int(x) for x in eos_token_id]
    eos_set = set(eos_ids)
    generated, events = [], []
    while len(generated) < max_new_tokens:
        embeddings = [_sumgd_embeddings(self, base, generated) for base in bases]
        # INTER's inclusion-exclusion contrast must be formed from finite direct
        # logits. ASCD's plausibility mask contains -inf and therefore cannot be
        # subtracted across coalitions. For the hybrid, add the finite INTER
        # correction to the ordinary ASCD parent distribution only afterwards.
        direct_logits = [
            _sumgd_forward(self, value, config_id=0).logits[:, -1, :]
            for value in embeddings
        ]
        parent_logits = (
            _sumgd_parent_logits(self, embeddings[0])
            if bool(getattr(self.cd_config, "if_cd", False))
            else direct_logits[0]
        )
        guided, event = inter_calibrate(
            *direct_logits,
            variance_threshold=float(getattr(self.cd_config, "inter_variance_threshold", 1.0)),
            beta=float(getattr(self.cd_config, "inter_beta", 0.1)),
            parent_logits=parent_logits,
        )
        prefix = torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device, dtype=torch.long)),
            dim=-1,
        )
        token = int(torch.argmax(logits_processor(prefix, guided), dim=-1)[0].item())
        event.update({
            "step": len(generated), "selected_token_id": token,
            "selected_changed_from_original_top": token != event["original_top_token_id"],
        })
        events.append(event)
        generated.append(token)
        if streamer is not None:
            streamer.put(torch.tensor([token]).cpu())
        result = torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device, dtype=torch.long)), dim=-1
        )
        if token in eos_set or stopping_criteria(result, None).any():
            break
    if streamer is not None:
        streamer.end()
    self.cd_config.inter_records.append({
        "image_id": int(getattr(self.cd_config, "inter_image_id")),
        "parent": "fixed_ascd" if bool(getattr(self.cd_config, "if_cd", False)) else "unmodified",
        "generated_tokens": len(generated),
        "forward_calls": len(generated) * (6 if bool(getattr(self.cd_config, "if_cd", False)) else 4),
        "active_steps": sum(event["interaction_active"] for event in events),
        "changed_steps": sum(event["selected_changed_from_original_top"] for event in events),
        "events": events,
    })
    return result


def _fuzzycd_greedy_search(
    self, input_ids, logits_processor, stopping_criteria, eos_token_id,
    streamer, model_kwargs,
):
    """Cache-free frozen four-sharpen-branch FuzzyCD greedy decoding."""
    required = ["inputs_embeds"] + [f"inputs_embeds_fuzzycd_{i}" for i in range(4)]
    if input_ids.shape[0] != 1 or any(name not in model_kwargs for name in required):
        raise ValueError("FuzzyCD requires batch one and four filtered embeddings")
    bases = [model_kwargs[name] for name in required]
    if len({tuple(value.shape) for value in bases}) != 1:
        raise ValueError("FuzzyCD image branches have inconsistent embedding shapes")
    calibration = getattr(self.cd_config, "fuzzycd_calibration", None)
    if not isinstance(calibration, dict):
        raise ValueError("FuzzyCD requires a calibration dictionary")
    max_new_tokens = int(getattr(self.cd_config, "fuzzycd_max_new_tokens", 512))
    eos_ids = [int(eos_token_id)] if isinstance(eos_token_id, int) else [int(x) for x in eos_token_id]
    eos_set = set(eos_ids)
    generated, events = [], []
    while len(generated) < max_new_tokens:
        embeddings = [_sumgd_embeddings(self, base, generated) for base in bases]
        direct_logits = [
            _sumgd_forward(self, value, config_id=0).logits[:, -1, :]
            for value in embeddings
        ]
        parent_logits = (
            _sumgd_parent_logits(self, embeddings[0])
            if bool(getattr(self.cd_config, "if_cd", False))
            else direct_logits[0]
        )
        calibrated, event = fuzzycd_calibrate(
            direct_logits[0], direct_logits[1:], calibration,
            beta=float(getattr(self.cd_config, "fuzzycd_beta", 0.1)),
            parent_logits=parent_logits,
        )
        prefix = torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device, dtype=torch.long)), dim=-1
        )
        token = int(torch.argmax(logits_processor(prefix, calibrated), dim=-1)[0].item())
        event.update({
            "step": len(generated), "selected_token_id": token,
            "selected_changed_from_parent_top": token != event["parent_top_token_id"],
        })
        events.append(event); generated.append(token)
        if streamer is not None:
            streamer.put(torch.tensor([token]).cpu())
        result = torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device, dtype=torch.long)), dim=-1
        )
        if token in eos_set or stopping_criteria(result, None).any():
            break
    if streamer is not None:
        streamer.end()
    self.cd_config.fuzzycd_records.append({
        "image_id": int(getattr(self.cd_config, "fuzzycd_image_id")),
        "parent": "fixed_ascd" if bool(getattr(self.cd_config, "if_cd", False)) else "unmodified",
        "generated_tokens": len(generated),
        "forward_calls": len(generated) * (7 if bool(getattr(self.cd_config, "if_cd", False)) else 5),
        "changed_steps": sum(event["selected_changed_from_parent_top"] for event in events),
        "events": events,
    })
    return result


def _set_mole_capture_mode(model, config):
    enabled = bool(getattr(config, "mole_enabled", False))
    prompt_end = int(getattr(config, "mole_prompt_end", -1))
    matches = 0
    for module in model.modules():
        if not hasattr(module, "attn_steer_configs"):
            continue
        matches += 1
        module.mole_capture_enabled = enabled
        module.mole_prompt_end = prompt_end
        if enabled:
            module._mole_prompt_mass_by_config = {}
    if enabled and matches < 1:
        raise RuntimeError("MoLE found no wrapped attention layers")


def _collect_mole_prompt_masses(model, config_id=0):
    masses = []
    for module in model.modules():
        if not hasattr(module, "attn_steer_configs"):
            continue
        layer = int(getattr(module.original_module, "layer_idx", -1))
        values = getattr(module, "_mole_prompt_mass_by_config", None)
        if values is None or int(config_id) not in values:
            raise RuntimeError(f"MoLE missing prompt mass for layer {layer}")
        masses.append((layer, values[int(config_id)]))
    masses.sort(key=lambda item: item[0])
    expected = list(range(len(masses)))
    if [layer for layer, _ in masses] != expected:
        raise RuntimeError("MoLE attention layers are not contiguous from zero")
    return [value for _, value in masses]


def _mole_rerank(self, parent_logits, outputs):
    config = self.cd_config
    if not bool(getattr(config, "mole_enabled", False)):
        return parent_logits
    if outputs.hidden_states is None:
        raise RuntimeError("MoLE requires output_hidden_states=True")
    masses = _collect_mole_prompt_masses(self.model, config_id=0)
    calibrated, record = mole_calibrate(
        parent_logits=parent_logits,
        original_final_logits=outputs.logits[:, -1, :],
        hidden_states=outputs.hidden_states,
        lm_head=self.lm_head,
        prompt_attention_masses=masses,
        top_n=int(getattr(config, "mole_top_n", 5)),
        final_layers=int(getattr(config, "mole_final_layers", 3)),
    )
    record.update({
        "image_id": int(getattr(config, "mole_image_id")),
        "step": int(getattr(config, "mole_step", 0)),
        "parent": "fixed_ascd" if bool(getattr(config, "if_cd", False)) else "unmodified",
        "released_tmp": float(getattr(config, "mole_tmp", 50.0)),
        "released_w_exp": float(getattr(config, "mole_w_exp", 0.2)),
        "released_branch": "top2_of_three",
    })
    if not hasattr(config, "mole_records"):
        config.mole_records = []
    config.mole_records.append(record)
    config.mole_step = record["step"] + 1
    return calibrated


def _set_only_capture_mode(model, config):
    enabled = bool(getattr(config, "only_enabled", False))
    matches = 0
    for module in model.modules():
        if not hasattr(module, "attn_steer_configs"):
            continue
        module.only_capture_enabled = enabled
        module.only_layer_index = int(getattr(config, "only_layer_index", 0))
        if int(getattr(module.original_module, "layer_idx", -1)) == module.only_layer_index:
            matches += 1
            if enabled:
                module._only_branch_by_config = {}
    if enabled and matches != 1:
        raise RuntimeError(f"ONLY expected one selected attention layer, found {matches}")


def _collect_only_branch(model, config_id=0):
    matches = []
    for module in model.modules():
        branches = getattr(module, "_only_branch_by_config", None)
        if branches and int(config_id) in branches:
            matches.append(branches[int(config_id)])
    if len(matches) != 1:
        raise RuntimeError(f"ONLY expected one captured branch, found {len(matches)}")
    return matches[0]


def _only_rerank(self, parent_logits, outputs):
    config = self.cd_config
    if not bool(getattr(config, "only_enabled", False)):
        return parent_logits
    branch = _collect_only_branch(self.model, config_id=0)
    text_enhanced_logits = build_text_enhanced_logits(
        text_enhanced_attention_output=branch["attention_output"],
        hidden_states=outputs.hidden_states,
        final_decoder_layer=self.model.layers[-1],
        final_norm=self.model.norm,
        lm_head=self.lm_head,
    ).to(parent_logits.dtype)
    calibrated, record = only_calibrate(
        parent_logits=parent_logits,
        text_enhanced_logits=text_enhanced_logits,
        positive_alpha=float(getattr(config, "only_positive_alpha", 3.0)),
        negative_alpha=float(getattr(config, "only_negative_alpha", 1.0)),
        beta=float(getattr(config, "only_beta", 0.1)),
        tvd_threshold=float(getattr(config, "only_tvd_threshold", 0.25)),
    )
    ratios = branch["entropy_ratios"].detach().float().cpu()
    removed = branch["removed_heads"].detach().cpu()
    record.update({
        "image_id": int(getattr(config, "only_image_id")),
        "step": int(getattr(config, "only_step", 0)),
        "parent": "fixed_ascd" if bool(getattr(config, "if_cd", False)) else "unmodified",
        "intervention_layer": int(getattr(config, "only_layer_index", 0)),
        "removed_head_indices": [int(x) for x in torch.where(removed)[0].tolist()],
        "removed_head_count": int(removed.sum().item()),
        "entropy_ratio_min": float(ratios.min().item()),
        "entropy_ratio_mean": float(ratios.mean().item()),
        "entropy_ratio_max": float(ratios.max().item()),
    })
    if not hasattr(config, "only_records"):
        config.only_records = []
    config.only_records.append(record)
    config.only_step = record["step"] + 1
    return calibrated


def _deco_rerank(self, parent_logits, outputs):
    config = self.cd_config
    if not bool(getattr(config, "deco_enabled", False)):
        return parent_logits
    if outputs.hidden_states is None:
        raise RuntimeError("DeCo requires output_hidden_states=True")
    calibrated, record = deco_calibrate(
        parent_logits=parent_logits,
        hidden_states=outputs.hidden_states,
        norm=self.model.norm,
        lm_head=self.lm_head,
        early_exit_layers=getattr(config, "deco_early_exit_layers", range(20, 29)),
        alpha=float(getattr(config, "deco_alpha", 0.6)),
        threshold_top_p=float(getattr(config, "deco_threshold_top_p", 0.9)),
        threshold_top_k=int(getattr(config, "deco_threshold_top_k", 20)),
    )
    record["image_id"] = int(getattr(config, "deco_image_id"))
    record["step"] = int(getattr(config, "deco_step", 0))
    record["parent"] = "fixed_ascd" if bool(getattr(config, "if_cd", False)) else "unmodified"
    if not hasattr(config, "deco_records"):
        config.deco_records = []
    config.deco_records.append(record)
    config.deco_step = record["step"] + 1
    return calibrated


def _configure_context_entropy_capture(model, config, enabled):
    layer_ids = [
        int(module.original_module.layer_idx)
        for module in model.modules()
        if hasattr(module, "attn_steer_configs")
        and hasattr(module, "original_module")
        and hasattr(module.original_module, "layer_idx")
    ]
    if enabled and not layer_ids:
        raise RuntimeError("no wrapped attention layers available for context entropy")
    for module in model.modules():
        if not hasattr(module, "attn_steer_configs"):
            continue
        module.context_entropy_capture_enabled = bool(enabled)
        if enabled:
            module.context_entropy_layer_id = max(layer_ids)
            module.context_entropy_instruction_end = int(
                getattr(config, "context_entropy_instruction_end")
            )
            if int(getattr(module.original_module, "layer_idx", -1)) == max(layer_ids):
                module._context_entropy_components_by_config = {}


def _collect_context_entropy_components(model):
    matches = []
    for module in model.modules():
        snapshots = getattr(module, "_context_entropy_components_by_config", None)
        if snapshots and 2 in snapshots:
            matches.append(snapshots[2])
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one final-layer context entropy snapshot, found {len(matches)}"
        )
    return matches[0]

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


def _context_entropy_rerank(self, input_ids, scores, base_model_kwargs):
    config = self.cd_config
    if not getattr(config, "context_entropy_enabled", False):
        return scores
    if scores.shape[0] != 1:
        raise ValueError("context entropy reranking requires batch_size=1")
    top_k = int(getattr(config, "context_entropy_top_k", 3))
    if top_k < 2:
        raise ValueError("context_entropy_top_k must be >= 2")
    values, ids = torch.topk(scores, k=top_k, dim=-1)
    if not torch.isfinite(values).all():
        raise ValueError("context entropy top candidates must have finite scores")
    candidate_ids = [int(value) for value in ids[0].tolist()]
    entropies = []
    distributions = []
    try:
        for candidate_id in candidate_ids:
            candidate_input_ids = torch.cat(
                (input_ids, torch.tensor([[candidate_id]], device=input_ids.device)), dim=-1
            )
            kwargs = {
                key: value for key, value in base_model_kwargs.items()
                if key not in {"past_key_values", "cache_position", "inputs_embeds", "position_ids"}
            }
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = torch.ones_like(candidate_input_ids)
            candidate_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **kwargs)
            candidate_inputs["use_cache"] = False
            switch_attn_steer_id(self.model, 2)
            _configure_context_entropy_capture(self.model, config, True)
            candidate_outputs = self(
                **candidate_inputs, return_dict=True, output_attentions=False
            )
            entropy, distribution = contextual_entropy_from_component_means(
                _collect_context_entropy_components(self.model)
            )
            entropies.append(entropy)
            distributions.append(distribution)
            del candidate_outputs
    finally:
        _configure_context_entropy_capture(self.model, config, False)
    beta = float(getattr(config, "context_entropy_beta", 10.0))
    calibrated = calibrate_top_candidates(scores, candidate_ids, entropies, beta)
    selected = int(torch.argmax(calibrated, dim=-1)[0].item())
    if not hasattr(config, "context_entropy_records"):
        config.context_entropy_records = []
    config.context_entropy_records.append({
        "image_id": int(getattr(config, "context_entropy_image_id")),
        "step": int(getattr(config, "context_entropy_step", 0)),
        "candidate_ids": candidate_ids,
        "base_scores": [float(value) for value in values[0].detach().float().cpu().tolist()],
        "contextual_entropies": entropies,
        "context_distributions": distributions,
        "calibrated_scores": [
            float(calibrated[0, candidate_id].detach().float().item())
            for candidate_id in candidate_ids
        ],
        "selected_token_id": selected,
    })
    config.context_entropy_step = int(getattr(config, "context_entropy_step", 0)) + 1
    return calibrated


def switch_attn_steer_id(model, attn_steer_id: int):
    ### model: the language model of llava
    attn_class = find_attn_class(model)
    # denosie_attn_class = CONTRASTIVE_ATTN_MAPPING[model.contrastive_attn_type][attn_class]
    denosie_attn_class = CONTRASTIVE_ATTN_MAPPING[model.contrastive_attn_type][attn_class] if attn_class in list(CONTRASTIVE_ATTN_MAPPING[model.contrastive_attn_type].keys()) else attn_class
    for name, module in model.named_modules():
        if isinstance(module, denosie_attn_class):
            assert attn_steer_id < len(module.attn_steer_configs), f"The attn_steer_id {attn_steer_id} exceeds the max number of attn_steer_config!"
            module.cur_config_id = attn_steer_id


def _set_attention_capture_mode(model, cd_config) -> None:
    """Configure compact attention caching once before decoding begins."""
    diagnostics_enabled = bool(getattr(cd_config, "diagnostics_enabled", False))
    visual_preservation_enabled = bool(
        getattr(cd_config, "visual_preservation_gate", False)
    )
    for module in model.modules():
        if hasattr(module, "_capture_attention_diagnostics"):
            module.diagnostics_enabled = diagnostics_enabled
            module.visual_preservation_enabled = visual_preservation_enabled


def _collect_positive_image_mass(model) -> torch.Tensor:
    """Return all-layer positive-branch image mass without a CPU sync."""
    layer_masses = []
    for module in model.modules():
        snapshots = getattr(module, "_diagnostic_attn_by_config", None)
        if snapshots and 0 in snapshots and "image_mass" in snapshots[0]:
            layer_masses.append(snapshots[0]["image_mass"])
    if not layer_masses:
        raise RuntimeError(
            "visual_preservation_gate requires cached positive image attention; "
            "check eager attention replacement and contrastive_layer_ids."
        )
    return torch.stack(layer_masses, dim=0).detach().float().mean(dim=0)


def _alpha_component_summary(cd_config):
    components = getattr(cd_config, "_last_alpha_components", None)
    if not components:
        return None
    return {
        name: float(torch.as_tensor(value).detach().float().reshape(-1)[0].item())
        for name, value in components.items()
    }


def get_contrastive_alpha(cd_config, logits: torch.Tensor, model=None):
    """Return a fixed, confidence-gated, or evidence-preserving coefficient.

    The original ASCD path uses one global ``cd_alpha`` for every decoding
    step.  When enabled, the coefficient is larger for low-margin tokens and
    smaller for already-confident tokens, reducing over-suppression of visual
    evidence while retaining stronger correction at uncertain steps.
    """
    base_alpha = getattr(cd_config, "cd_alpha", None)
    base_alpha = 0.5 if base_alpha is None else float(base_alpha)
    alpha = base_alpha
    margin = None
    if getattr(cd_config, "adaptive_alpha", False):
        alpha_min = float(getattr(cd_config, "adaptive_alpha_min", base_alpha))
        alpha_max = float(getattr(cd_config, "adaptive_alpha_max", base_alpha))
        if alpha_min > alpha_max:
            raise ValueError("adaptive_alpha_min must be <= adaptive_alpha_max")

        margin_threshold = float(getattr(cd_config, "adaptive_alpha_margin", 1.0))
        margin_temperature = max(
            float(getattr(cd_config, "adaptive_alpha_temperature", 0.5)), 1e-6
        )
        top2 = torch.topk(logits.detach().float(), k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        uncertainty = torch.sigmoid((margin_threshold - margin) / margin_temperature)
        alpha = alpha_min + (alpha_max - alpha_min) * uncertainty.unsqueeze(-1)

    if getattr(cd_config, "record_margin_values", False):
        if margin is None:
            raise ValueError("record_margin_values requires adaptive_alpha=True")
        if not hasattr(cd_config, "recorded_margin_values"):
            cd_config.recorded_margin_values = []
        image_id = getattr(cd_config, "margin_image_id", None)
        if image_id is None:
            raise ValueError("margin_image_id must be set while recording margins")
        step = int(getattr(cd_config, "margin_step", 0))
        cd_config.recorded_margin_values.append({
            "image_id": int(image_id),
            "step": step,
            "margin": float(margin.detach().float().reshape(-1)[0].item()),
            "alpha": float(torch.as_tensor(alpha).detach().float().reshape(-1)[0].item()),
        })
        cd_config.margin_step = step + 1

    if not getattr(cd_config, "visual_preservation_gate", False):
        cd_config._last_alpha_components = None
        return alpha
    if model is None:
        raise ValueError("model is required when visual_preservation_gate=True")

    mass_center = float(
        getattr(cd_config, "visual_preservation_mass_center", 0.1392475516)
    )
    mass_temperature = float(
        getattr(cd_config, "visual_preservation_mass_temperature", 0.0178608781)
    )
    if mass_temperature <= 0:
        raise ValueError("visual_preservation_mass_temperature must be > 0")

    alpha_floor = float(
        getattr(cd_config, "visual_preservation_alpha_min", 0.15)
    )
    configured_min = (
        float(getattr(cd_config, "adaptive_alpha_min", base_alpha))
        if getattr(cd_config, "adaptive_alpha", False)
        else base_alpha
    )
    if alpha_floor < 0 or alpha_floor > configured_min:
        raise ValueError(
            "visual_preservation_alpha_min must be non-negative and no greater "
            "than the base gate's minimum alpha"
        )

    image_mass = _collect_positive_image_mass(model).unsqueeze(-1)
    preserve_score = torch.sigmoid((image_mass - mass_center) / mass_temperature)
    alpha_tensor = torch.as_tensor(
        alpha, dtype=image_mass.dtype, device=image_mass.device
    )
    final_alpha = alpha_floor + (alpha_tensor - alpha_floor) * (1.0 - preserve_score)
    cd_config._last_alpha_components = {
        "base_alpha": alpha_tensor.detach(),
        "positive_image_mass": image_mass.detach(),
        "preserve_score": preserve_score.detach(),
        "final_alpha": final_alpha.detach(),
    }
    return final_alpha


def _collect_attention_diagnostics(model, config_id: int):
    """Collect one compact attention snapshot from every instrumented layer."""
    layer_payloads = []
    for module in model.modules():
        snapshots = getattr(module, "_diagnostic_attn_by_config", None)
        if snapshots and config_id in snapshots:
            layer_payloads.append(snapshots[config_id])

    if not layer_payloads:
        return {"available": False, "num_layers": 0}

    layer_payloads.sort(key=lambda item: item["layer_id"])
    metric_names = ("system_mass", "image_mass", "history_mass", "image_entropy")
    matrix = torch.stack(
        [
            torch.stack([payload[name][0] for name in metric_names])
            for payload in layer_payloads
        ]
    ).detach().float().cpu()

    result = {
        "available": True,
        "num_layers": len(layer_payloads),
        "layer_ids": [payload["layer_id"] for payload in layer_payloads],
    }
    for metric_index, metric_name in enumerate(metric_names):
        values = matrix[:, metric_index].tolist()
        result[metric_name] = float(matrix[:, metric_index].mean().item())
        result[f"layerwise_{metric_name}"] = values
    return result


def _js_divergence(prob_p: torch.Tensor, prob_q: torch.Tensor) -> torch.Tensor:
    midpoint = 0.5 * (prob_p + prob_q)
    return 0.5 * (
        (
            prob_p
            * (prob_p.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())
        ).sum(dim=-1)
        + (
            prob_q
            * (prob_q.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())
        ).sum(dim=-1)
    )


def build_token_diagnostic_record(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    alpha,
    selected_token_id: int,
    top_k: int = 5,
):
    """Compute distribution-level diagnostics without retaining full logits."""
    if positive_logits.shape[0] != 1 or negative_logits.shape[0] != 1:
        raise ValueError("Token diagnostics currently support batch_size=1 only.")

    positive = positive_logits.detach().float()
    negative = negative_logits.detach().float()
    positive_log_prob = torch.log_softmax(positive, dim=-1)
    negative_log_prob = torch.log_softmax(negative, dim=-1)
    positive_prob = positive_log_prob.exp()
    negative_prob = negative_log_prob.exp()

    vocab_size = positive.shape[-1]
    entropy_scale = math.log(max(vocab_size, 2))
    positive_entropy = -(positive_prob * positive_log_prob).sum(dim=-1)
    negative_entropy = -(negative_prob * negative_log_prob).sum(dim=-1)

    positive_top2 = torch.topk(positive, k=min(2, vocab_size), dim=-1)
    negative_top2 = torch.topk(negative, k=min(2, vocab_size), dim=-1)
    positive_margin = (
        positive_top2.values[:, 0] - positive_top2.values[:, 1]
        if vocab_size > 1
        else torch.zeros(1, device=positive.device)
    )
    negative_margin = (
        negative_top2.values[:, 0] - negative_top2.values[:, 1]
        if vocab_size > 1
        else torch.zeros(1, device=negative.device)
    )

    mask = candidate_mask.detach().bool()
    positive_truncated = torch.softmax(
        positive.masked_fill(~mask, -float("inf")), dim=-1
    )
    negative_truncated = torch.softmax(
        negative.masked_fill(~mask, -float("inf")), dim=-1
    )

    selected = int(selected_token_id)
    alpha_value = float(torch.as_tensor(alpha).detach().float().reshape(-1)[0].item())
    positive_top_id = int(positive_top2.indices[0, 0].item())
    negative_top_id = int(negative_top2.indices[0, 0].item())
    k = max(1, min(int(top_k), vocab_size))
    positive_top = torch.topk(positive_prob, k=k, dim=-1)
    negative_top = torch.topk(negative_prob, k=k, dim=-1)

    return {
        "alpha_t": alpha_value,
        "positive_margin": float(positive_margin[0].item()),
        "negative_margin": float(negative_margin[0].item()),
        "positive_entropy": float(positive_entropy[0].item()),
        "negative_entropy": float(negative_entropy[0].item()),
        "positive_normalized_entropy": float((positive_entropy[0] / entropy_scale).item()),
        "negative_normalized_entropy": float((negative_entropy[0] / entropy_scale).item()),
        "branch_js_divergence": float(_js_divergence(positive_prob, negative_prob)[0].item()),
        "candidate_js_divergence": float(
            _js_divergence(positive_truncated, negative_truncated)[0].item()
        ),
        "candidate_count": int(mask[0].sum().item()),
        "positive_top_token_id": positive_top_id,
        "negative_top_token_id": negative_top_id,
        "top_token_agreement": bool(positive_top_id == negative_top_id),
        "selected_token_id": selected,
        "ascd_changed_top_token": bool(selected != positive_top_id),
        "selected_positive_logit": float(positive[0, selected].item()),
        "selected_negative_logit": float(negative[0, selected].item()),
        "selected_signed_branch_gap": float(
            (positive[0, selected] - negative[0, selected]).item()
        ),
        "selected_contrastive_correction": float(
            (alpha_value * (positive[0, selected] - negative[0, selected])).item()
        ),
        "positive_top_ids": [int(value) for value in positive_top.indices[0].tolist()],
        "positive_top_probs": [float(value) for value in positive_top.values[0].tolist()],
        "negative_top_ids": [int(value) for value in negative_top.indices[0].tolist()],
        "negative_top_probs": [float(value) for value in negative_top.values[0].tolist()],
    }


def build_neutral_directional_diagnostic(
    positive_logits: torch.Tensor,
    neutral_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    selected_token_id: int,
):
    """Build compact three-branch diagnostics for one greedy decoding step.

    The rival is frozen in advance as the highest-positive-logit token in the
    existing positive cutoff set, excluding the selected token.  If that set
    has no alternative, the positive runner-up is used and the fallback is
    recorded.  Relative gaps avoid dependence on arbitrary logit offsets.
    """
    if any(logits.shape[0] != 1 for logits in (positive_logits, neutral_logits, negative_logits)):
        raise ValueError("Neutral directional diagnostics require batch_size=1.")
    if not (positive_logits.shape == neutral_logits.shape == negative_logits.shape):
        raise ValueError("Positive, neutral, and negative logits must have identical shapes.")

    positive = positive_logits.detach().float()
    neutral = neutral_logits.detach().float()
    negative = negative_logits.detach().float()
    mask = candidate_mask.detach().bool().clone()
    selected = int(selected_token_id)
    mask[0, selected] = False
    rival_from_cutoff = bool(mask[0].any().item())
    if rival_from_cutoff:
        rival = int(positive.masked_fill(~mask, -float("inf")).argmax(dim=-1)[0].item())
    else:
        fallback = positive.clone()
        fallback[0, selected] = -float("inf")
        rival = int(fallback.argmax(dim=-1)[0].item())

    def gap(logits):
        return logits[0, selected] - logits[0, rival]

    g_pos, g_neutral, g_neg = gap(positive), gap(neutral), gap(negative)
    positive_support = g_pos - g_neutral
    negative_support = g_neutral - g_neg
    correction_gap = g_pos - g_neg

    positive_prob = torch.softmax(positive, dim=-1)
    neutral_prob = torch.softmax(neutral, dim=-1)
    negative_prob = torch.softmax(negative, dim=-1)
    positive_top = int(positive.argmax(dim=-1)[0].item())
    neutral_top = int(neutral.argmax(dim=-1)[0].item())
    negative_top = int(negative.argmax(dim=-1)[0].item())

    return {
        "neutral_js": {
            "positive_neutral": float(_js_divergence(positive_prob, neutral_prob)[0].item()),
            "neutral_negative": float(_js_divergence(neutral_prob, negative_prob)[0].item()),
            "positive_negative": float(_js_divergence(positive_prob, negative_prob)[0].item()),
        },
        "positive_neutral_top_agreement": bool(positive_top == neutral_top),
        "neutral_negative_top_agreement": bool(neutral_top == negative_top),
        "three_branch_top_agreement": bool(positive_top == neutral_top == negative_top),
        "neutral_top_token_id": neutral_top,
        "neutral_top_probability": float(neutral_prob[0, neutral_top].item()),
        "ascd_selected_differs_from_neutral_top": bool(selected != neutral_top),
        "rival_token_id": rival,
        "rival_from_positive_cutoff": rival_from_cutoff,
        "g_pos": float(g_pos.item()),
        "g_neutral": float(g_neutral.item()),
        "g_neg": float(g_neg.item()),
        "positive_support": float(positive_support.item()),
        "negative_support": float(negative_support.item()),
        "correction_gap": float(correction_gap.item()),
        "directional_consistency": bool(
            torch.sign(positive_support).item() == torch.sign(negative_support).item()
        ),
        "joint_support": float(torch.minimum(positive_support, negative_support).item()),
        "support_conflict": float(torch.abs(positive_support - negative_support).item()),
    }


def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin._sample`] directly. Use generate() instead.
    For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        output_logits (`bool`, *optional*, defaults to `False`):
            Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
            more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     TopKLogitsWarper,
    ...     TemperatureLogitsWarper,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> model.generation_config.pad_token_id = model.config.eos_token_id

    >>> input_prompt = "Today is a beautiful day, and"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> # instantiate logits processors
    >>> logits_warper = LogitsProcessorList(
    ...     [
    ...         TopKLogitsWarper(50),
    ...         TemperatureLogitsWarper(0.7),
    ...     ]
    ... )

    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
    >>> outputs = model._sample(
    ...     input_ids,
    ...     logits_processor=logits_processor,
    ...     logits_warper=logits_warper,
    ...     stopping_criteria=stopping_criteria,
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
    ```"""
    # init values
    key_position = model_kwargs.pop("key_position", None)

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    _set_attention_capture_mode(self.model, self.cd_config)
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

    temp_if_vcd = True
    if hasattr(self.cd_config, "if_vcd") and self.cd_config.if_vcd:
        model_kwargs_cd = model_kwargs.copy()
        if "inputs_embeds_vcd" not in model_kwargs_cd or model_kwargs_cd['inputs_embeds_vcd'] is None:
            temp_if_vcd = False
        else:
            model_kwargs_cd['inputs_embeds'] = model_kwargs_cd.pop("inputs_embeds_vcd")
            model_kwargs_cd.pop("images_cd")
            # input_ids_cd = input_ids.clone()
            model_kwargs.pop("images_cd")
            model_kwargs.pop("inputs_embeds_vcd")

    elif hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd:
        model_kwargs_cd = model_kwargs.copy()
        input_ids_icd = input_ids.clone()
        model_kwargs_cd['inputs_embeds'] = model_kwargs_cd.pop('inputs_embeds_icd')

        generationMixin = transformers.generation.utils.GenerationMixin()
        model_kwargs_cd["attention_mask"] = generationMixin._prepare_attention_mask_for_generation(
            model_kwargs_cd.get("inputs_embeds"), pad_token_id, eos_token_id
        )
        if "inputs_embeds" in model_kwargs_cd:
            cur_len_cd = model_kwargs_cd["inputs_embeds"].shape[1]
        model_kwargs_cd["cache_position"] = torch.arange(cur_len_cd, device=input_ids_icd.device)

        model_kwargs_cd.pop("input_ids_icd")
        model_kwargs.pop("input_ids_icd")
        model_kwargs.pop("inputs_embeds_icd")

    elif hasattr(self.cd_config, "if_sid") and self.cd_config.if_sid:
        model_kwargs_cd = model_kwargs.copy()
    else:
        model_kwargs_cd = model_kwargs.copy()


    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if not (hasattr(self.cd_config, "if_vcd") and self.cd_config.if_vcd
                or hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd
                or hasattr(self.cd_config, "if_sid") and self.cd_config.if_sid):
            if hasattr(self.cd_config, "if_cd") and self.cd_config.if_cd:
                switch_attn_steer_id(self.model, 0)
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=(
                output_hidden_states
                or bool(getattr(self.cd_config, "deco_enabled", False))
            ),
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        ############# Contrastive Decoding ##############
        if self.cd_config.if_cd and temp_if_vcd:
            output_attentions_wo_img = (
                output_attentions if output_attentions is not None else self.generation_config.output_attentions
            )
            output_hidden_states_wo_img = (
                output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
            )
            if hasattr(self.cd_config, "if_vcd") and self.cd_config.if_vcd:
                model_inputs_vcd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
                outputs_cd = self(
                    **model_inputs_vcd,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]
                
            elif hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd:
                model_inputs_icd = self.prepare_inputs_for_generation(input_ids_icd, **model_kwargs_cd)
                outputs_cd = self(
                    **model_inputs_icd,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]

            elif hasattr(self.cd_config, "if_sid") and self.cd_config.if_sid:

                model_inputs_sid = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
                outputs_cd = self(
                    **model_inputs_sid,
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=output_hidden_states_wo_img,
                    key_position=key_position,
                    vad = False,
                )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]
            else:
                model_inputs_attn_steer = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
                switch_attn_steer_id(self.model, 1)
                outputs_cd = self(
                    **model_inputs_attn_steer,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]

            cd_alpha = get_contrastive_alpha(
                self.cd_config, next_token_logits, model=self.model
            )
            cd_beta = self.cd_config.cd_beta if self.cd_config.cd_beta is not None else 0.1
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            next_token_logits = cd_logits

        #########################################

        next_token_logits = _deco_rerank(self, next_token_logits, outputs)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if self.cd_config.if_cd and temp_if_vcd:
            if hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd:
                input_ids_icd = torch.cat([input_ids_icd, next_tokens[:, None]], dim=-1)

            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids
    

def apply_forced_token_audit(next_tokens, diagnostic_step, spec):
    """Apply one pre-registered token intervention after strict prefix checks.

    The spec dictionary is mutated only by setting _applied after the target
    token is replaced. The caller must provide a fresh spec per sample.
    """
    if spec is None or spec.get("_applied", False):
        return next_tokens, None
    if next_tokens.ndim != 1 or next_tokens.shape[0] != 1:
        raise ValueError("Forced-token audit requires batch_size=1.")

    target_step = int(spec["step"])
    if target_step < 0:
        raise ValueError("Forced-token target step must be non-negative.")
    original_token_id = int(next_tokens[0].item())

    if diagnostic_step < target_step:
        expected_prefix = spec.get("expected_prefix_token_ids")
        if expected_prefix is None or len(expected_prefix) != target_step:
            raise ValueError(
                "Forced-token plan must contain one expected prefix token per "
                "step before the intervention."
            )
        expected_token_id = int(expected_prefix[diagnostic_step])
        if original_token_id != expected_token_id:
            raise RuntimeError(
                "Forced-token prefix mismatch at step "
                f"{diagnostic_step}: expected={expected_token_id} "
                f"actual={original_token_id}."
            )
        return next_tokens, None

    if diagnostic_step > target_step:
        raise RuntimeError(
            f"Forced-token target step {target_step} was passed without applying it."
        )

    expected_selected_token_id = int(spec["expected_selected_token_id"])
    if original_token_id != expected_selected_token_id:
        raise RuntimeError(
            "Forced-token target mismatch at step "
            f"{diagnostic_step}: expected={expected_selected_token_id} "
            f"actual={original_token_id}."
        )
    forced_token_id = int(spec["forced_token_id"])
    if forced_token_id < 0:
        raise ValueError("Forced token id must be non-negative.")

    forced = next_tokens.clone()
    forced[0] = forced_token_id
    spec["_applied"] = True
    event = {
        "step": target_step,
        "original_token_id": original_token_id,
        "forced_token_id": forced_token_id,
        "expected_selected_token_id": expected_selected_token_id,
        "prefix_verified": True,
        "target_verified": True,
    }
    return forced, event


def _verifier_constrained_greedy_search(
    self,
    input_ids,
    logits_processor,
    stopping_criteria,
    eos_token_id,
    streamer,
    model_kwargs,
):
    """Cache-free, same-prefix Vanilla-vs-Fixed-ASCD token verification.

    Both candidate distributions are recomputed from the exact selected prefix.
    This avoids mixing an ASCD key/value cache with a Vanilla fallback token.
    The only branch decision is whether to retain Fixed ASCD's top token at a
    CHAIR-object-like disagreement; ASCD itself remains untouched.
    """
    config = self.cd_config
    if not bool(getattr(config, "if_cd", False)):
        raise ValueError("Verifier-Constrained ASCD requires Fixed ASCD (if_cd=True)")
    if "inputs_embeds" not in model_kwargs:
        raise ValueError("Verifier-Constrained ASCD requires multimodal inputs_embeds")
    tokenizer = getattr(config, "verifier_tokenizer", None)
    runtime = getattr(config, "verifier_runtime", None)
    if tokenizer is None or runtime is None:
        raise ValueError("Verifier-Constrained ASCD requires tokenizer and CLIP runtime")

    base_embeddings = model_kwargs["inputs_embeds"]
    max_new_tokens = int(getattr(config, "verifier_max_new_tokens", 512))
    if max_new_tokens < 1:
        raise ValueError("verifier_max_new_tokens must be positive")
    eos_ids = [int(eos_token_id)] if isinstance(eos_token_id, int) else [
        int(value) for value in eos_token_id
    ]
    eos_set = set(eos_ids)
    generated, events = [], []
    disagreements = eligible = fallback_count = accepted_count = 0
    observe_only = bool(getattr(config, "verifier_observe_only", False))
    threshold = float(getattr(config, "verifier_threshold"))

    while len(generated) < max_new_tokens:
        embeddings = _sumgd_embeddings(self, base_embeddings, generated)
        positive_logits = _sumgd_forward(self, embeddings, config_id=0).logits[:, -1, :]
        negative_logits = _sumgd_forward(self, embeddings, config_id=1).logits[:, -1, :]
        cd_alpha = get_contrastive_alpha(config, positive_logits, model=self.model)
        cd_beta = config.cd_beta if config.cd_beta is not None else 0.1
        cutoff = torch.log(torch.tensor(cd_beta, device=positive_logits.device)) + (
            positive_logits.max(dim=-1, keepdim=True).values
        )
        ascd_logits = ((1 + cd_alpha) * positive_logits - cd_alpha * negative_logits)
        ascd_logits = ascd_logits.masked_fill(positive_logits < cutoff, -float("inf"))
        prefix = torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device, dtype=torch.long)),
            dim=-1,
        )
        ascd_scores = logits_processor(prefix, ascd_logits)
        ascd_token = int(torch.argmax(ascd_scores, dim=-1)[0].item())

        # The unmodified candidate is evaluated at the same selected prefix,
        # not through an incompatible ASCD cache.
        vanilla_logits = _sumgd_forward(
            self, embeddings, config_id=0, unmodified=True
        ).logits[:, -1, :]
        vanilla_scores = logits_processor(prefix, vanilla_logits)
        vanilla_token = int(torch.argmax(vanilla_scores, dim=-1)[0].item())
        selected = ascd_token

        if ascd_token != vanilla_token:
            disagreements += 1
            ascd_object = terminal_chair_object(tokenizer, generated, ascd_token)
            vanilla_object = terminal_chair_object(tokenizer, generated, vanilla_token)
            if ascd_object or vanilla_object:
                eligible += 1
                ascd_text = tokenizer.decode(
                    generated + [ascd_token], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                vanilla_text = tokenizer.decode(
                    generated + [vanilla_token], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                ascd_score, vanilla_score = runtime.score_pair(ascd_text, vanilla_text)
            else:
                ascd_score = vanilla_score = None
            decision = decide_verifier_constraint(
                ascd_token_id=ascd_token,
                vanilla_token_id=vanilla_token,
                ascd_object_word=ascd_object,
                vanilla_object_word=vanilla_object,
                ascd_score=ascd_score,
                vanilla_score=vanilla_score,
                threshold=threshold,
                observe_only=observe_only,
            )
            selected = int(decision.selected_token_id)
            accepted_count += int(
                decision.eligible_object_position and decision.accepts_ascd
            )
            fallback_count += int(decision.reason == "fallback_vanilla")
            events.append(build_audit_record(
                image_id=int(getattr(config, "verifier_image_id")),
                step=len(generated),
                ascd_token_id=ascd_token,
                vanilla_token_id=vanilla_token,
                ascd_token=tokenizer.convert_ids_to_tokens(ascd_token),
                vanilla_token=tokenizer.convert_ids_to_tokens(vanilla_token),
                decision=decision,
            ))

        generated.append(selected)
        if streamer is not None:
            streamer.put(torch.tensor([selected]).cpu())
        result = torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device, dtype=torch.long)),
            dim=-1,
        )
        if selected in eos_set or bool(stopping_criteria(result, None).any().item()):
            break

    if streamer is not None:
        streamer.end()
    terminated_by_eos = bool(generated and generated[-1] in eos_set)
    if not hasattr(config, "verifier_records"):
        config.verifier_records = []
    config.verifier_records.append({
        "image_id": int(getattr(config, "verifier_image_id")),
        "parent": "fixed_ascd",
        "mode": "observe_only" if observe_only else "constrained",
        "threshold": threshold,
        "generated_tokens": len(generated),
        "terminated_by_eos": terminated_by_eos,
        "hit_max_new_tokens": len(generated) >= max_new_tokens and not terminated_by_eos,
        "candidate_disagreements": disagreements,
        "eligible_object_disagreements": eligible,
        "accepted_ascd_count": accepted_count,
        "fallback_vanilla_count": fallback_count,
        "events": events,
    })
    return result



def _detector_comparative_greedy_search(
    self,
    input_ids,
    logits_processor,
    stopping_criteria,
    eos_token_id,
    streamer,
    model_kwargs,
):
    """Same-prefix object-local OWLv2 reversion between Fixed ASCD and Vanilla.

    The forward passes are cache-free so a fallback cannot inherit an ASCD KV
    cache. The branch is inactive unless explicitly enabled by the evaluator.
    """
    config = self.cd_config
    if not bool(getattr(config, "if_cd", False)):
        raise ValueError("Comparative Evidence Reversion requires Fixed ASCD (if_cd=True)")
    if "inputs_embeds" not in model_kwargs:
        raise ValueError("Comparative Evidence Reversion requires multimodal inputs_embeds")
    tokenizer = getattr(config, "detector_comparative_tokenizer", None)
    runtime = getattr(config, "detector_comparative_runtime", None)
    if tokenizer is None or runtime is None:
        raise ValueError("Comparative Evidence Reversion requires tokenizer and OWLv2 runtime")

    base_embeddings = model_kwargs["inputs_embeds"]
    max_new_tokens = int(getattr(config, "detector_comparative_max_new_tokens", 512))
    if max_new_tokens < 1:
        raise ValueError("detector_comparative_max_new_tokens must be positive")
    eos_ids = [int(eos_token_id)] if isinstance(eos_token_id, int) else [
        int(value) for value in eos_token_id
    ]
    eos_set = set(eos_ids)
    generated, events = [], []
    disagreements = eligible = fallback_count = accepted_count = 0
    observe_only = bool(getattr(config, "detector_comparative_observe_only", False))
    threshold = float(getattr(config, "detector_comparative_threshold", 0.0))

    while len(generated) < max_new_tokens:
        embeddings = _sumgd_embeddings(self, base_embeddings, generated)
        positive_logits = _sumgd_forward(self, embeddings, config_id=0).logits[:, -1, :]
        negative_logits = _sumgd_forward(self, embeddings, config_id=1).logits[:, -1, :]
        cd_alpha = get_contrastive_alpha(config, positive_logits, model=self.model)
        cd_beta = config.cd_beta if config.cd_beta is not None else 0.1
        cutoff = torch.log(torch.tensor(cd_beta, device=positive_logits.device)) + (
            positive_logits.max(dim=-1, keepdim=True).values
        )
        ascd_logits = ((1 + cd_alpha) * positive_logits - cd_alpha * negative_logits)
        ascd_logits = ascd_logits.masked_fill(positive_logits < cutoff, -float("inf"))
        prefix = torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device, dtype=torch.long)),
            dim=-1,
        )
        ascd_scores = logits_processor(prefix, ascd_logits)
        ascd_token = int(torch.argmax(ascd_scores, dim=-1)[0].item())

        # Recompute unmodified Vanilla at precisely the selected prefix.
        vanilla_logits = _sumgd_forward(
            self, embeddings, config_id=0, unmodified=True
        ).logits[:, -1, :]
        vanilla_scores = logits_processor(prefix, vanilla_logits)
        vanilla_token = int(torch.argmax(vanilla_scores, dim=-1)[0].item())
        selected = ascd_token

        if ascd_token != vanilla_token:
            disagreements += 1
            ascd_object = terminal_decoded_chair_object(tokenizer, generated, ascd_token)
            vanilla_object = terminal_decoded_chair_object(tokenizer, generated, vanilla_token)
            ascd_support = runtime.scores.get(ascd_object) if ascd_object is not None else None
            vanilla_support = runtime.scores.get(vanilla_object) if vanilla_object is not None else None
            decision = decide_comparative_reversion(
                ascd_token_id=ascd_token,
                vanilla_token_id=vanilla_token,
                ascd_object=ascd_object,
                vanilla_object=vanilla_object,
                ascd_support=ascd_support,
                vanilla_support=vanilla_support,
                threshold=threshold,
                observe_only=observe_only,
            )
            selected = int(decision.selected_token_id)
            eligible += int(decision.eligible_object_position)
            accepted_count += int(
                decision.eligible_object_position and decision.accepts_ascd
            )
            fallback_count += int(decision.reason == "fallback_vanilla")
            events.append(build_comparative_audit_record(
                image_id=int(getattr(config, "detector_comparative_image_id")),
                step=len(generated),
                ascd_token_id=ascd_token,
                vanilla_token_id=vanilla_token,
                ascd_token=tokenizer.convert_ids_to_tokens(ascd_token),
                vanilla_token=tokenizer.convert_ids_to_tokens(vanilla_token),
                decision=decision,
            ))

        generated.append(selected)
        if streamer is not None:
            streamer.put(torch.tensor([selected]).cpu())
        result = torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device, dtype=torch.long)),
            dim=-1,
        )
        if selected in eos_set or bool(stopping_criteria(result, None).any().item()):
            break

    if streamer is not None:
        streamer.end()
    terminated_by_eos = bool(generated and generated[-1] in eos_set)
    if not hasattr(config, "detector_comparative_records"):
        config.detector_comparative_records = []
    config.detector_comparative_records.append({
        "image_id": int(getattr(config, "detector_comparative_image_id")),
        "parent": "fixed_ascd",
        "mode": "observe_only" if observe_only else "constrained",
        "threshold": None if observe_only else threshold,
        "generated_tokens": len(generated),
        "terminated_by_eos": terminated_by_eos,
        "hit_max_new_tokens": len(generated) >= max_new_tokens and not terminated_by_eos,
        "candidate_disagreements": disagreements,
        "eligible_object_disagreements": eligible,
        "accepted_ascd_count": accepted_count,
        "fallback_vanilla_count": fallback_count,
        "events": events,
    })
    return result
def _greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin._greedy_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>


    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.

        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        output_logits (`bool`, *optional*, defaults to `False`):
            Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
            for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

    >>> input_prompt = "It might be possible to"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> outputs = model._greedy_search(
    ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
    ```"""
    if bool(getattr(self.cd_config, "selfaug_enabled", False)):
        return selfaug_greedy_search(
            self, input_ids,
            logits_processor if logits_processor is not None else LogitsProcessorList(),
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
            eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streamer, model_kwargs,
        )
    if bool(getattr(self.cd_config, "vista_enabled", False)):
        return vista_greedy_search(
            self, input_ids,
            logits_processor if logits_processor is not None else LogitsProcessorList(),
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
            eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streamer, model_kwargs,
        )
    if bool(getattr(self.cd_config, "verifier_constrained_enabled", False)):
        return _verifier_constrained_greedy_search(
            self, input_ids,
            logits_processor if logits_processor is not None else LogitsProcessorList(),
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
            eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streamer, model_kwargs,
        )
    if bool(getattr(self.cd_config, "sumgd_enabled", False)):
        return _sumgd_greedy_search(
            self=self,
            input_ids=input_ids,
            logits_processor=logits_processor if logits_processor is not None else LogitsProcessorList(),
            stopping_criteria=stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streamer=streamer,
            model_kwargs=model_kwargs,
        )
    if bool(getattr(self.cd_config, "mfcd_enabled", False)):
        return _mfcd_greedy_search(
            self=self,
            input_ids=input_ids,
            logits_processor=logits_processor if logits_processor is not None else LogitsProcessorList(),
            stopping_criteria=stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
            eos_token_id=eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streamer=streamer,
            model_kwargs=model_kwargs,
        )
    if bool(getattr(self.cd_config, "inter_enabled", False)):
        return _inter_greedy_search(
            self, input_ids,
            logits_processor if logits_processor is not None else LogitsProcessorList(),
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
            eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streamer, model_kwargs,
        )
    if bool(getattr(self.cd_config, "fuzzycd_enabled", False)):
        return _fuzzycd_greedy_search(
            self, input_ids,
            logits_processor if logits_processor is not None else LogitsProcessorList(),
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
            eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streamer, model_kwargs,
        )
    if bool(getattr(self.cd_config, "crops_enabled", False)):
        return crops_greedy_search(
            self, input_ids,
            logits_processor if logits_processor is not None else LogitsProcessorList(),
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
            eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streamer, model_kwargs,
            lambda embeddings: _sumgd_parent_logits(self, embeddings),
        )
    if bool(getattr(self.cd_config, "cei_enabled", False)):
        return cei_greedy_search(
            self, input_ids,
            logits_processor if logits_processor is not None else LogitsProcessorList(),
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
            eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streamer, model_kwargs,
            lambda embeddings: _cei_context_embedding(self, embeddings),
            lambda embeddings, context, layer, alpha, sink: _cei_parent_logits(
                self, embeddings, context, layer, alpha, sink
            ),
        )
    if bool(getattr(self.cd_config, "dive_enabled", False)):
        return dive_greedy_search(
            self, input_ids,
            logits_processor if logits_processor is not None else LogitsProcessorList(),
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
            eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streamer, model_kwargs,
            lambda positive, embeddings: _dive_parent_logits(
                self, positive, embeddings
            ),
        )

    # init values
    ###################### test
    # switch_attn_steer_id(self.model, 1)
    ######################
    key_position = model_kwargs.pop("key_position", None)

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    vhr_initial_input_length = int(input_ids.shape[-1])
    vhr_state = _initialize_vhr(self, model_kwargs)
    context_entropy_base_model_kwargs = model_kwargs.copy()
    _set_attention_capture_mode(self.model, self.cd_config)
    _set_only_capture_mode(self.model, self.cd_config)
    _set_mole_capture_mode(self.model, self.cd_config)
    diagnostics_enabled = bool(getattr(self.cd_config, "diagnostics_enabled", False))
    neutral_diagnostics = bool(
        getattr(self.cd_config, "neutral_directional_diagnostics", False)
    )
    forced_token_audit = bool(
        getattr(self.cd_config, "forced_token_audit_enabled", False)
    )
    detector_grounded = bool(
        getattr(self.cd_config, "detector_grounded_enabled", False)
    )
    soft_grounded = bool(
        getattr(self.cd_config, "soft_grounded_enabled", False)
    )
    alias_guard_observe = bool(
        getattr(self.cd_config, "alias_guard_observe_enabled", False)
    )
    if sum((detector_grounded, soft_grounded, alias_guard_observe)) > 1:
        raise ValueError("Hard Detector, Soft-Grounded, and alias observation modes are mutually exclusive")
    if soft_grounded:
        if batch_size != 1 or not bool(getattr(self.cd_config, "if_cd", False)):
            raise ValueError("Soft-Grounded ASCD requires batch_size=1 Fixed ASCD")
        if getattr(self.cd_config, "soft_grounded_tokenizer", None) is None:
            raise ValueError("Soft-Grounded ASCD requires a tokenizer")
        if getattr(self.cd_config, "soft_grounded_runtime", None) is None:
            raise ValueError("Soft-Grounded ASCD requires an OWLv2 runtime")
    if alias_guard_observe:
        if batch_size != 1 or not bool(getattr(self.cd_config, "if_cd", False)):
            raise ValueError("Alias observation requires batch_size=1 Fixed ASCD")
        required = (
            "alias_guard_tokenizer", "alias_guard_canonical_runtime",
            "alias_guard_alias_runtime", "alias_guard_generated_token_ids",
        )
        if any(getattr(self.cd_config, key, None) is None for key in required):
            raise ValueError("Alias observation requires tokenizer, OWLv2 runtimes, and generated-token state")
    if forced_token_audit and not diagnostics_enabled:
        raise ValueError("Forced-token audit requires diagnostics_enabled=True.")
    if neutral_diagnostics and not diagnostics_enabled:
        raise ValueError("Neutral directional diagnostics require diagnostics_enabled=True.")
    if diagnostics_enabled:
        if batch_size != 1:
            raise ValueError("Greedy token diagnostics currently require batch_size=1.")
        if not getattr(self.cd_config, "if_cd", False):
            raise ValueError("Greedy token diagnostics require contrastive decoding (if_cd=True).")
        if getattr(self.cd_config, "diagnostic_file_handle", None) is None:
            raise ValueError("diagnostic_file_handle is required when diagnostics are enabled.")
    detector_comparative = bool(
        getattr(self.cd_config, "detector_comparative_enabled", False)
    )
    if detector_comparative:
        if batch_size != 1:
            raise ValueError("Comparative Evidence Reversion requires batch_size=1")
        if not bool(getattr(self.cd_config, "if_cd", False)):
            raise ValueError("Comparative Evidence Reversion requires Fixed ASCD")
        if getattr(self.cd_config, "detector_comparative_tokenizer", None) is None:
            raise ValueError("Comparative Evidence Reversion requires a tokenizer")
        if getattr(self.cd_config, "detector_comparative_runtime", None) is None:
            raise ValueError("Comparative Evidence Reversion requires an OWLv2 runtime")
    diagnostic_step = 0
    diagnostic_generated_token_ids = getattr(
        self.cd_config, "diagnostic_generated_token_ids", []
    )
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

    temp_if_vcd = True
    if hasattr(self.cd_config, "if_vcd") and self.cd_config.if_vcd:
        model_kwargs_cd = model_kwargs.copy()
        if "inputs_embeds_vcd" not in model_kwargs_cd or model_kwargs_cd['inputs_embeds_vcd'] is None:
            temp_if_vcd = False
        else:
            model_kwargs_cd['inputs_embeds'] = model_kwargs_cd.pop("inputs_embeds_vcd")
            model_kwargs_cd.pop("images_cd")
            # input_ids_cd = input_ids.clone()
            model_kwargs.pop("images_cd")
            model_kwargs.pop("inputs_embeds_vcd")

    elif hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd:
        model_kwargs_cd = model_kwargs.copy()
        input_ids_icd = input_ids.clone()
        model_kwargs_cd['inputs_embeds'] = model_kwargs_cd.pop('inputs_embeds_icd')

        generationMixin = transformers.generation.utils.GenerationMixin()
        model_kwargs_cd["attention_mask"] = generationMixin._prepare_attention_mask_for_generation(
            model_kwargs_cd.get("inputs_embeds"), pad_token_id, eos_token_id
        )
        if "inputs_embeds" in model_kwargs_cd:
            cur_len_cd = model_kwargs_cd["inputs_embeds"].shape[1]
        model_kwargs_cd["cache_position"] = torch.arange(cur_len_cd, device=input_ids_icd.device)

        model_kwargs_cd.pop("input_ids_icd")
        model_kwargs.pop("input_ids_icd")
        model_kwargs.pop("inputs_embeds_icd")

    elif hasattr(self.cd_config, "if_sid") and self.cd_config.if_sid:
        model_kwargs_cd = model_kwargs.copy()
    else:
        model_kwargs_cd = model_kwargs.copy()
    model_kwargs_neutral = model_kwargs.copy() if neutral_diagnostics else None
    if detector_comparative:
        detector_comparative_config = self.cd_config
        detector_comparative_generated = []
        detector_comparative_events = []
        detector_comparative_disagreements = 0
        detector_comparative_eligible = 0
        detector_comparative_accepted = 0
        detector_comparative_fallbacks = 0
        detector_comparative_vanilla_input_ids = input_ids.clone()
        detector_comparative_vanilla_kwargs = model_kwargs.copy()

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if not (hasattr(self.cd_config, "if_vcd") and self.cd_config.if_vcd
                or hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd
                or hasattr(self.cd_config, "if_sid") and self.cd_config.if_sid):
            if hasattr(self.cd_config, "if_cd") and self.cd_config.if_cd:
                switch_attn_steer_id(self.model, 0)
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=(
                output_hidden_states
                or bool(getattr(self.cd_config, "deco_enabled", False))
                or bool(getattr(self.cd_config, "only_enabled", False))
                or bool(getattr(self.cd_config, "mole_enabled", False))
            ),
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        if diagnostics_enabled:
            diagnostic_positive_logits = next_token_logits
        # if torch.argmax(next_token_logits).item() == 7254 or torch.argmax(next_token_logits).item() == 24841:
        #     print(next_token_logits[0, 7254].item(), next_token_logits[0, 13328].item(), next_token_logits[0, 24841].item(), next_token_logits[0, 7933].item(), next_token_logits[0, 4796].item(), next_token_logits[0, 4628].item(), )
        
        ############# Contrastive Decoding ##############
        if self.cd_config.if_cd and temp_if_vcd:
            output_attentions_wo_img = (
                output_attentions if output_attentions is not None else self.generation_config.output_attentions
            )
            output_hidden_states_wo_img = (
                output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
            )
            if hasattr(self.cd_config, "if_vcd") and self.cd_config.if_vcd:
                model_inputs_vcd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
                outputs_cd = self(
                    **model_inputs_vcd,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]
                
            elif hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd:
                model_inputs_icd = self.prepare_inputs_for_generation(input_ids_icd, **model_kwargs_cd)
                outputs_cd = self(
                    **model_inputs_icd,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]

            elif hasattr(self.cd_config, "if_sid") and self.cd_config.if_sid:

                model_inputs_sid = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
                outputs_cd = self(
                    **model_inputs_sid,
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=output_hidden_states_wo_img,
                    key_position=key_position,
                    vad = False,
                )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]
            else:
                model_inputs_attn_steer = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
                switch_attn_steer_id(self.model, 1)
                outputs_cd = self(
                    **model_inputs_attn_steer,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]
            if neutral_diagnostics:
                model_inputs_neutral = self.prepare_inputs_for_generation(
                    input_ids, **model_kwargs_neutral
                )
                switch_attn_steer_id(self.model, 2)
                outputs_neutral = self(
                    **model_inputs_neutral,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                diagnostic_neutral_logits = outputs_neutral.logits[:, -1, :]
            cd_alpha = get_contrastive_alpha(
                self.cd_config, next_token_logits, model=self.model
            )
            cd_beta = self.cd_config.cd_beta if self.cd_config.cd_beta is not None else 0.1
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            if diagnostics_enabled:
                diagnostic_negative_logits = next_token_logits_cd
                diagnostic_candidate_mask = next_token_logits >= cutoff
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            next_token_logits = cd_logits

        #########################################

        next_token_logits = _deco_rerank(self, next_token_logits, outputs)
        next_token_logits = _only_rerank(self, next_token_logits, outputs)
        next_token_logits = _mole_rerank(self, next_token_logits, outputs)

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens_scores = _context_entropy_rerank(
            self, input_ids, next_tokens_scores, context_entropy_base_model_kwargs
        )
        if detector_grounded:
            tokenizer = getattr(self.cd_config, "detector_tokenizer", None)
            runtime = getattr(self.cd_config, "detector_runtime", None)
            generated = getattr(self.cd_config, "detector_generated_token_ids", None)
            if tokenizer is None or runtime is None or generated is None:
                raise ValueError("Detector-Grounded ASCD requires tokenizer, OWLv2 runtime, and per-image generated-token state")
            next_tokens_scores, detector_event = apply_detector_object_mask(
                next_tokens_scores,
                tokenizer=tokenizer,
                generated_token_ids=generated,
                support_scores=runtime.scores,
                threshold=float(getattr(self.cd_config, "detector_threshold")),
                top_k=int(getattr(self.cd_config, "detector_top_k")),
            )
            object_candidates = detector_event["object_candidates"]
            self.cd_config.detector_decoding_steps += 1
            self.cd_config.detector_object_candidates += len(object_candidates)
            self.cd_config.detector_masked_candidates += len(detector_event["masked_token_ids"])
            self.cd_config.detector_selection_changes += int(detector_event["selection_changed"])
            self.cd_config.detector_no_finite_protections += int(detector_event["protected_no_finite"])
            if object_candidates:
                self.cd_config.detector_events.append(detector_event)
        elif soft_grounded:
            tokenizer = self.cd_config.soft_grounded_tokenizer
            runtime = self.cd_config.soft_grounded_runtime
            generated = getattr(self.cd_config, "soft_grounded_generated_token_ids", None)
            if generated is None:
                raise ValueError("Soft-Grounded ASCD requires per-image generated-token state")
            next_tokens_scores, soft_event = apply_soft_grounded_object_penalty(
                next_tokens_scores,
                tokenizer=tokenizer,
                generated_token_ids=generated,
                support_scores=runtime.scores,
                top_k=int(self.cd_config.soft_grounded_top_k),
                probability_slope=float(self.cd_config.soft_grounded_probability_slope),
                probability_intercept=float(self.cd_config.soft_grounded_probability_intercept),
                probability_clip_min=float(self.cd_config.soft_grounded_probability_clip_min),
                probability_clip_max=float(self.cd_config.soft_grounded_probability_clip_max),
            )
            object_candidates = soft_event["object_candidates"]
            self.cd_config.soft_grounded_decoding_steps += 1
            self.cd_config.soft_grounded_object_candidates += len(object_candidates)
            self.cd_config.soft_grounded_penalized_candidates += len(soft_event["soft_penalized_token_ids"])
            self.cd_config.soft_grounded_selection_changes += int(soft_event["selection_changed"])
            self.cd_config.soft_grounded_no_finite_protections += int(soft_event["protected_no_finite"])
            if object_candidates:
                self.cd_config.soft_grounded_events.append(soft_event)
        elif alias_guard_observe:
            alias_event = observe_alias_guard_candidates(
                next_tokens_scores,
                tokenizer=self.cd_config.alias_guard_tokenizer,
                generated_token_ids=self.cd_config.alias_guard_generated_token_ids,
                canonical_support_scores=self.cd_config.alias_guard_canonical_runtime.scores,
                alias_runtime=self.cd_config.alias_guard_alias_runtime,
                hard_threshold=float(self.cd_config.alias_guard_hard_threshold),
                top_k=int(self.cd_config.alias_guard_top_k),
            )
            object_candidates = alias_event["object_candidates"]
            self.cd_config.alias_guard_decoding_steps += 1
            self.cd_config.alias_guard_object_candidates += len(object_candidates)
            self.cd_config.alias_guard_would_hard_mask += sum(
                int(candidate["would_hard_mask"]) for candidate in object_candidates
            )
            if object_candidates:
                self.cd_config.alias_guard_events.append(alias_event)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if detector_comparative:
            # Keep an ordinary GenerationMixin cache at exactly the selected
            # prefix. The ASCD path below remains the stock cached path.
            switch_attn_steer_id(self.model, 0)
            vanilla_model_inputs = self.prepare_inputs_for_generation(
                detector_comparative_vanilla_input_ids,
                **detector_comparative_vanilla_kwargs,
            )
            with _sumgd_unmodified_attention(self.model):
                detector_comparative_vanilla_outputs = self(
                    **vanilla_model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
            vanilla_scores = logits_processor(
                detector_comparative_vanilla_input_ids,
                detector_comparative_vanilla_outputs.logits[:, -1, :],
            )
            ascd_token = int(next_tokens[0].item())
            vanilla_token = int(torch.argmax(vanilla_scores, dim=-1)[0].item())
            if ascd_token != vanilla_token:
                detector_comparative_disagreements += 1
                tokenizer = detector_comparative_config.detector_comparative_tokenizer
                runtime = detector_comparative_config.detector_comparative_runtime
                ascd_object = terminal_decoded_chair_object(
                    tokenizer, detector_comparative_generated, ascd_token
                )
                vanilla_object = terminal_decoded_chair_object(
                    tokenizer, detector_comparative_generated, vanilla_token
                )
                ascd_support = runtime.scores.get(ascd_object) if ascd_object is not None else None
                vanilla_support = runtime.scores.get(vanilla_object) if vanilla_object is not None else None
                decision = decide_comparative_reversion(
                    ascd_token_id=ascd_token,
                    vanilla_token_id=vanilla_token,
                    ascd_object=ascd_object,
                    vanilla_object=vanilla_object,
                    ascd_support=ascd_support,
                    vanilla_support=vanilla_support,
                    threshold=float(detector_comparative_config.detector_comparative_threshold),
                    observe_only=bool(detector_comparative_config.detector_comparative_observe_only),
                )
                detector_comparative_eligible += int(decision.eligible_object_position)
                detector_comparative_accepted += int(
                    decision.eligible_object_position and decision.accepts_ascd
                )
                detector_comparative_fallbacks += int(decision.reason == "fallback_vanilla")
                detector_comparative_events.append(build_comparative_audit_record(
                    image_id=int(detector_comparative_config.detector_comparative_image_id),
                    step=len(detector_comparative_generated),
                    ascd_token_id=ascd_token,
                    vanilla_token_id=vanilla_token,
                    ascd_token=tokenizer.convert_ids_to_tokens(ascd_token),
                    vanilla_token=tokenizer.convert_ids_to_tokens(vanilla_token),
                    decision=decision,
                ))
                next_tokens = torch.tensor(
                    [int(decision.selected_token_id)],
                    device=next_tokens.device,
                    dtype=next_tokens.dtype,
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        forced_event = None
        if forced_token_audit:
            next_tokens, forced_event = apply_forced_token_audit(
                next_tokens, diagnostic_step, getattr(self.cd_config, "forced_token_spec", None)
            )

        if detector_grounded:
            self.cd_config.detector_generated_token_ids.append(
                int(next_tokens[0].item())
            )
        if soft_grounded:
            self.cd_config.soft_grounded_generated_token_ids.append(
                int(next_tokens[0].item())
            )
        if alias_guard_observe:
            self.cd_config.alias_guard_generated_token_ids.append(
                int(next_tokens[0].item())
            )
        if detector_comparative:
            detector_comparative_generated.append(int(next_tokens[0].item()))

        if diagnostics_enabled:
            selected_token_id = int(next_tokens[0].item())
            diagnostic_generated_token_ids.append(selected_token_id)
            tokenizer = getattr(self.cd_config, "diagnostic_tokenizer", None)
            if tokenizer is None:
                raise ValueError("diagnostic_tokenizer is required when diagnostics are enabled.")

            if forced_event is not None:
                forced_event.update(
                    {
                        "schema_version": 1,
                        "run_name": getattr(self.cd_config, "diagnostics_run_name", None),
                        "sample_index": int(
                            getattr(self.cd_config, "diagnostic_sample_index", -1)
                        ),
                        "image_id": getattr(self.cd_config, "diagnostic_image_id", None),
                        "original_token": tokenizer.convert_ids_to_tokens(
                            forced_event["original_token_id"]
                        ),
                        "forced_token": tokenizer.convert_ids_to_tokens(
                            forced_event["forced_token_id"]
                        ),
                    }
                )
                audit_handle = getattr(self.cd_config, "forced_token_audit_file_handle", None)
                if audit_handle is None:
                    raise ValueError("forced_token_audit_file_handle is required.")
                audit_handle.write(json.dumps(forced_event, ensure_ascii=False) + "\n")

            record = build_token_diagnostic_record(
                positive_logits=diagnostic_positive_logits,
                negative_logits=diagnostic_negative_logits,
                candidate_mask=diagnostic_candidate_mask,
                alpha=cd_alpha,
                selected_token_id=selected_token_id,
                top_k=getattr(self.cd_config, "diagnostics_top_k", 5),
            )
            record.update(
                {
                    "schema_version": 1,
                    "run_name": getattr(self.cd_config, "diagnostics_run_name", None),
                    "sample_index": int(
                        getattr(self.cd_config, "diagnostic_sample_index", -1)
                    ),
                    "image_id": getattr(self.cd_config, "diagnostic_image_id", None),
                    "step": diagnostic_step,
                    "selected_token": tokenizer.convert_ids_to_tokens(selected_token_id),
                    "generated_text_so_far": tokenizer.decode(
                        diagnostic_generated_token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    "positive_attention": _collect_attention_diagnostics(self.model, 0),
                    "negative_attention": _collect_attention_diagnostics(self.model, 1),
                }
            )
            if neutral_diagnostics:
                record["neutral_directional"] = build_neutral_directional_diagnostic(
                    positive_logits=diagnostic_positive_logits,
                    neutral_logits=diagnostic_neutral_logits,
                    negative_logits=diagnostic_negative_logits,
                    candidate_mask=diagnostic_candidate_mask,
                    selected_token_id=selected_token_id,
                )
                record["neutral_attention"] = _collect_attention_diagnostics(
                    self.model, 2
                )
            if forced_event is not None:
                record["forced_token_audit"] = dict(forced_event)
            alpha_components = _alpha_component_summary(self.cd_config)
            if alpha_components is not None:
                record["alpha_components"] = alpha_components
            self.cd_config.diagnostic_file_handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            diagnostic_step += 1

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if detector_comparative:
            detector_comparative_vanilla_input_ids = torch.cat(
                [detector_comparative_vanilla_input_ids, next_tokens[:, None]], dim=-1
            )
            detector_comparative_vanilla_kwargs = self._update_model_kwargs_for_generation(
                detector_comparative_vanilla_outputs,
                detector_comparative_vanilla_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
        if self.cd_config.if_cd and temp_if_vcd:
            if hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd:
                input_ids_icd = torch.cat([input_ids_icd, next_tokens[:, None]], dim=-1)

            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if neutral_diagnostics:
                model_kwargs_neutral = self._update_model_kwargs_for_generation(
                    outputs_neutral,
                    model_kwargs_neutral,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()
    if detector_comparative:
        detector_comparative_config.detector_comparative_records.append({
            "image_id": int(detector_comparative_config.detector_comparative_image_id),
            "parent": "fixed_ascd",
            "mode": "observe_only" if detector_comparative_config.detector_comparative_observe_only else "constrained",
            "threshold": None if detector_comparative_config.detector_comparative_observe_only else float(detector_comparative_config.detector_comparative_threshold),
            "generated_tokens": len(detector_comparative_generated),
            "terminated_by_eos": bool(detector_comparative_generated and detector_comparative_generated[-1] in eos_token_id),
            "hit_max_new_tokens": len(detector_comparative_generated) >= int(getattr(detector_comparative_config, "detector_comparative_max_new_tokens", 512)) and not bool(detector_comparative_generated and detector_comparative_generated[-1] in eos_token_id),
            "candidate_disagreements": detector_comparative_disagreements,
            "eligible_object_disagreements": detector_comparative_eligible,
            "accepted_ascd_count": detector_comparative_accepted,
            "fallback_vanilla_count": detector_comparative_fallbacks,
            "events": detector_comparative_events,
        })

    _finalize_vhr(
        self, vhr_state, int(input_ids.shape[-1]) - vhr_initial_input_length
    )

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids



def _beam_search(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    sequential: Optional[bool] = None,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin._beam_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_logits (`bool`, *optional*, defaults to `False`):
            Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
            more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        sequential (`bool`, defaults to `False`):
            By default, beam search has `batch_size * num_beams` as effective batch size (see `beam_search()` for
            more details). This flag will avoid parallelizing the beam search and will instead run beam search
            sequentially.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.


    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     BeamSearchScorer,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


    >>> # lets run beam search using 3 beams
    >>> num_beams = 3
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id

    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }

    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ... )

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ...     ]
    ... )

    >>> outputs = model._beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""
    key_position = model_kwargs.pop("key_position", None)

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    sequential = sequential if sequential is not None else self.generation_config.low_memory
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape
    _set_attention_capture_mode(self.model, self.cd_config)
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False

    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

    temp_if_vcd = True
    if hasattr(self.cd_config, "if_vcd") and self.cd_config.if_vcd:
        model_kwargs_cd = model_kwargs.copy()
        if "inputs_embeds_vcd" not in model_kwargs_cd or model_kwargs_cd['inputs_embeds_vcd'] is None:
            temp_if_vcd = False
        else:
            model_kwargs_cd['inputs_embeds'] = model_kwargs_cd.pop("inputs_embeds_vcd")
            model_kwargs_cd.pop("images_cd")
            # input_ids_cd = input_ids.clone()
            model_kwargs.pop("images_cd")
            model_kwargs.pop("inputs_embeds_vcd")
        

    elif hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd:
        model_kwargs_cd = model_kwargs.copy()
        input_ids_icd = input_ids.clone()
        model_kwargs_cd['inputs_embeds'] = model_kwargs_cd.pop('inputs_embeds_icd')

        generationMixin = transformers.generation.utils.GenerationMixin()
        model_kwargs_cd["attention_mask"] = generationMixin._prepare_attention_mask_for_generation(
            model_kwargs_cd.get("inputs_embeds"), pad_token_id, eos_token_id
        )
        if "inputs_embeds" in model_kwargs_cd:
            cur_len_cd = model_kwargs_cd["inputs_embeds"].shape[1]
        model_kwargs_cd["cache_position"] = torch.arange(cur_len_cd, device=input_ids_icd.device)

        model_kwargs_cd.pop("input_ids_icd")
        model_kwargs.pop("input_ids_icd")
        model_kwargs.pop("inputs_embeds_icd")

    elif hasattr(self.cd_config, "if_sid") and self.cd_config.if_sid:
        model_kwargs_cd = model_kwargs.copy()
    else:
        model_kwargs_cd = model_kwargs.copy()


    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if not (hasattr(self.cd_config, "if_vcd") and self.cd_config.if_vcd
                or hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd
                or hasattr(self.cd_config, "if_sid") and self.cd_config.if_sid):
            if hasattr(self.cd_config, "if_cd") and self.cd_config.if_cd:
                switch_attn_steer_id(self.model, 0)
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # if sequential is True, split the input to batches of batch_size and run sequentially
        if sequential:
            if any(
                model_name in self.__class__.__name__.lower()
                for model_name in [
                    "fsmt",
                    "reformer",
                    "bloom",
                    "ctrl",
                    "gpt_bigcode",
                    "transo_xl",
                    "xlnet",
                    "cpm",
                ]
            ):
                raise RuntimeError(
                    f"Currently generation for {self.__class__.__name__} is not supported "
                    f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                )

            inputs_per_sub_batches = _split_model_inputs(
                model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
            )
            outputs_per_sub_batch = [
                self(
                    **inputs_per_sub_batch,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                for inputs_per_sub_batch in inputs_per_sub_batches
            ]

            outputs = stack_model_outputs(outputs_per_sub_batch)

        else:  # Unchanged original behavior
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        # next_token_scores = nn.functional.log_softmax(
        #     next_token_logits, dim=-1
        # )  # (batch_size * num_beams, vocab_size)

        # next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        # next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
        #     next_token_scores_processed
        # )

        ############# Contrastive Decoding ##############
        if self.cd_config.if_cd and temp_if_vcd:
            output_attentions_wo_img = (
                output_attentions if output_attentions is not None else self.generation_config.output_attentions
            )
            output_hidden_states_wo_img = (
                output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
            )
            if hasattr(self.cd_config, "if_vcd") and self.cd_config.if_vcd:
                model_inputs_vcd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)

                # if sequential is True, split the input to batches of batch_size and run sequentially
                if sequential:
                    if any(
                        model_name in self.__class__.__name__.lower()
                        for model_name in [
                            "fsmt",
                            "reformer",
                            "bloom",
                            "ctrl",
                            "gpt_bigcode",
                            "transo_xl",
                            "xlnet",
                            "cpm",
                        ]
                    ):
                        raise RuntimeError(
                            f"Currently generation for {self.__class__.__name__} is not supported "
                            f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                        )

                    inputs_per_sub_batches_vcd = _split_model_inputs(
                        model_inputs_vcd, split_size=batch_size, full_batch_size=batch_beam_size
                    )
                    outputs_per_sub_batch_vcd = [
                        self(
                            **inputs_per_sub_batch,
                            return_dict=True,
                            output_attentions=output_attentions_wo_img,
                            output_hidden_states=output_hidden_states_wo_img,
                        )
                        for inputs_per_sub_batch in inputs_per_sub_batches_vcd
                    ]

                    outputs_cd = stack_model_outputs(outputs_per_sub_batch_vcd)

                else:  # Unchanged original behavior
                    outputs_cd = self(
                        **model_inputs_vcd,
                        return_dict=True,
                        output_attentions=output_attentions_wo_img,
                        output_hidden_states=output_hidden_states_wo_img,
                    )

                next_token_logits_cd = outputs_cd.logits[:, -1, :]


            elif hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd:

                model_inputs_icd = self.prepare_inputs_for_generation(input_ids_icd, **model_kwargs_cd)

                # if sequential is True, split the input to batches of batch_size and run sequentially
                if sequential:
                    if any(
                        model_name in self.__class__.__name__.lower()
                        for model_name in [
                            "fsmt",
                            "reformer",
                            "bloom",
                            "ctrl",
                            "gpt_bigcode",
                            "transo_xl",
                            "xlnet",
                            "cpm",
                        ]
                    ):
                        raise RuntimeError(
                            f"Currently generation for {self.__class__.__name__} is not supported "
                            f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                        )

                    inputs_per_sub_batches_icd = _split_model_inputs(
                        model_inputs_icd, split_size=batch_size, full_batch_size=batch_beam_size
                    )
                    outputs_per_sub_batch_icd = [
                        self(
                            **inputs_per_sub_batch,
                            return_dict=True,
                            output_attentions=output_attentions_wo_img,
                            output_hidden_states=output_hidden_states_wo_img,
                        )
                        for inputs_per_sub_batch in inputs_per_sub_batches_icd
                    ]

                    outputs_cd = stack_model_outputs(outputs_per_sub_batch_icd)

                else:  # Unchanged original behavior
                    outputs_cd = self(
                        **model_inputs_icd,
                        return_dict=True,
                        output_attentions=output_attentions_wo_img,
                        output_hidden_states=output_hidden_states_wo_img,
                    )

                next_token_logits_cd = outputs_cd.logits[:, -1, :]

            elif hasattr(self.cd_config, "if_sid") and self.cd_config.if_sid:
                model_inputs_sid = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)

                # if sequential is True, split the input to batches of batch_size and run sequentially
                if sequential:
                    # Not tested!
                    if any(
                        model_name in self.__class__.__name__.lower()
                        for model_name in [
                            "fsmt",
                            "reformer",
                            "bloom",
                            "ctrl",
                            "gpt_bigcode",
                            "transo_xl",
                            "xlnet",
                            "cpm",
                        ]
                    ):
                        raise RuntimeError(
                            f"Currently generation for {self.__class__.__name__} is not supported "
                            f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                        )

                    inputs_per_sub_batches_sid = _split_model_inputs(
                        model_inputs_sid, split_size=batch_size, full_batch_size=batch_beam_size
                    )
                    outputs_per_sub_batch_sid = [
                        self(
                            **inputs_per_sub_batch,
                            return_dict=True,
                            output_attentions=True,
                            output_hidden_states=output_hidden_states_wo_img,
                            key_position=key_position,
                            vad = False,
                        )
                        for inputs_per_sub_batch in inputs_per_sub_batches_sid
                    ]

                    outputs_cd = stack_model_outputs(outputs_per_sub_batch_sid)
                else:  # Unchanged original behavior
                    outputs_cd = self(
                        **model_inputs_sid,
                        return_dict=True,
                        output_attentions=True,
                        output_hidden_states=output_hidden_states_wo_img,
                        key_position=key_position,
                        vad = False,
                    )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]

            else:

                model_inputs_attn_steer = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
                switch_attn_steer_id(self.model, 1)

                # if sequential is True, split the input to batches of batch_size and run sequentially
                if sequential:
                    if any(
                        model_name in self.__class__.__name__.lower()
                        for model_name in [
                            "fsmt",
                            "reformer",
                            "bloom",
                            "ctrl",
                            "gpt_bigcode",
                            "transo_xl",
                            "xlnet",
                            "cpm",
                        ]
                    ):
                        raise RuntimeError(
                            f"Currently generation for {self.__class__.__name__} is not supported "
                            f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                        )

                    inputs_per_sub_batches_attn_steer = _split_model_inputs(
                        model_inputs_attn_steer, split_size=batch_size, full_batch_size=batch_beam_size
                    )
                    outputs_per_sub_batch_attn_steer = [
                        self(
                            **inputs_per_sub_batch,
                            return_dict=True,
                            output_attentions=output_attentions_wo_img,
                            output_hidden_states=output_hidden_states_wo_img,
                        )
                        for inputs_per_sub_batch in inputs_per_sub_batches_attn_steer
                    ]

                    outputs_cd = stack_model_outputs(outputs_per_sub_batch_attn_steer)

                else:  # Unchanged original behavior
                    outputs_cd = self(
                        **model_inputs_attn_steer,
                        return_dict=True,
                        output_attentions=output_attentions_wo_img,
                        output_hidden_states=output_hidden_states_wo_img,
                    )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]

            cd_alpha = get_contrastive_alpha(
                self.cd_config, next_token_logits, model=self.model
            )
            cd_beta = self.cd_config.cd_beta if self.cd_config.cd_beta is not None else 0.1
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            next_token_logits = cd_logits

        #########################################

        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
            next_token_scores_processed
        )

        if detector_grounded:
            tokenizer = getattr(self.cd_config, "detector_tokenizer", None)
            runtime = getattr(self.cd_config, "detector_runtime", None)
            generated = getattr(self.cd_config, "detector_generated_token_ids", None)
            if tokenizer is None or runtime is None or generated is None:
                raise ValueError("Detector-Grounded ASCD requires tokenizer, OWLv2 runtime, and per-image generated-token state")
            next_tokens_scores, detector_event = apply_detector_object_mask(
                next_tokens_scores,
                tokenizer=tokenizer,
                generated_token_ids=generated,
                support_scores=runtime.scores,
                threshold=float(getattr(self.cd_config, "detector_threshold")),
                top_k=int(getattr(self.cd_config, "detector_top_k")),
            )
            object_candidates = detector_event["object_candidates"]
            self.cd_config.detector_decoding_steps += 1
            self.cd_config.detector_object_candidates += len(object_candidates)
            self.cd_config.detector_masked_candidates += len(detector_event["masked_token_ids"])
            self.cd_config.detector_selection_changes += int(detector_event["selection_changed"])
            self.cd_config.detector_no_finite_protections += int(detector_event["protected_no_finite"])
            if object_candidates:
                self.cd_config.detector_events.append(detector_event)
        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")   # beam idx
        next_tokens = next_tokens % vocab_size   # token idx

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if model_kwargs.get("past_key_values", None) is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        if self.cd_config.if_cd and temp_if_vcd:
            if hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd:
                input_ids_icd = torch.cat([input_ids_icd[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if model_kwargs_cd.get("past_key_values", None) is not None:
                model_kwargs_cd["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs_cd["past_key_values"], beam_idx
                )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
            this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return sequence_outputs["sequences"]
