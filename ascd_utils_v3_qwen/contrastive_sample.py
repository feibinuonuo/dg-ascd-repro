import copy
import os
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
# import torch.distributed as dist
from torch import nn

import transformers
# from transformers.cache_utils import Cache, DynamicCache, StaticCache
# from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
# from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
# from transformers.models.auto import (
#     MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
#     MODEL_FOR_CAUSAL_LM_MAPPING,
#     MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
#     MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
#     MODEL_FOR_VISION_2_SEQ_MAPPING,
# )
from transformers.utils import ModelOutput, logging
# from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
# from transformers.generation.candidate_generator import (
#     AssistedCandidateGenerator,
#     CandidateGenerator,
#     PromptLookupCandidateGenerator,
#     _crop_past_key_values,
#     _prepare_attention_mask,
#     _prepare_token_type_ids,
# )
# from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.cache_utils import Cache

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

from transformers.generation.configuration_utils import GenerationConfig

from . import *
from ascd_detector_grounded import apply_detector_object_mask
from ascd_soft_grounded import apply_soft_grounded_object_penalty
from ascd_context_entropy import (
    calibrate_top_candidates,
    contextual_entropy_from_component_means,
)


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


def get_contrastive_alpha(cd_config, logits: torch.Tensor):
    """Return the fixed or frozen margin-adaptive ASCD coefficient."""
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
        margin_temperature = float(
            getattr(cd_config, "adaptive_alpha_temperature", 0.5)
        )
        if margin_temperature <= 0:
            raise ValueError("adaptive_alpha_temperature must be > 0")
        top2 = torch.topk(logits.detach().float(), k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        uncertainty = torch.sigmoid(
            (margin_threshold - margin) / margin_temperature
        )
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

    if getattr(cd_config, "record_alpha_values", False):
        values = torch.as_tensor(alpha).detach().float().reshape(-1).cpu().tolist()
        if not hasattr(cd_config, "recorded_alpha_values"):
            cd_config.recorded_alpha_values = []
        cd_config.recorded_alpha_values.extend(values)
    return alpha


def switch_attn_steer_id(model, attn_steer_id: int):
    ### model: the language model of llava
    attn_class = find_attn_class(model)
    # denosie_attn_class = CONTRASTIVE_ATTN_MAPPING[model.contrastive_attn_type][attn_class]
    denosie_attn_class = CONTRASTIVE_ATTN_MAPPING[model.contrastive_attn_type][attn_class] if attn_class in list(CONTRASTIVE_ATTN_MAPPING[model.contrastive_attn_type].keys()) else attn_class
    for name, module in model.named_modules():
        if isinstance(module, denosie_attn_class):
            assert attn_steer_id < len(module.attn_steer_configs), f"The attn_steer_id {attn_steer_id} exceeds the max number of attn_steer_config!"
            module.cur_config_id = attn_steer_id


def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

    # init values
    context_entropy_base_model_kwargs = model_kwargs.copy()
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    detector_grounded = bool(
        getattr(self.cd_config, "detector_grounded_enabled", False)
    )
    soft_grounded = bool(
        getattr(self.cd_config, "soft_grounded_enabled", False)
    )
    if detector_grounded and soft_grounded:
        raise ValueError("hard and soft grounded ASCD are mutually exclusive")
    if (detector_grounded or soft_grounded) and do_sample:
        raise ValueError("Grounded ASCD requires greedy decoding")

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
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__
    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
        if getattr(self, "hf_quantizer", None) is not None:
            is_compileable &= self.hf_quantizer.is_compileable
        is_compileable = is_compileable and not generation_config.disable_compile
        if is_compileable and (
            self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

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

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

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
                outputs_cd = self(**model_inputs_attn_steer, return_dict=True)
                next_token_logits_cd = outputs_cd.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            cd_alpha = get_contrastive_alpha(self.cd_config, next_token_logits)
            cd_beta = self.cd_config.cd_beta if self.cd_config.cd_beta is not None else 0.1
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            next_token_logits = cd_logits

        #########################################

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = _context_entropy_rerank(
            self, input_ids, next_token_scores, context_entropy_base_model_kwargs
        )
        if detector_grounded:
            tokenizer = getattr(self.cd_config, "detector_tokenizer", None)
            runtime = getattr(self.cd_config, "detector_runtime", None)
            generated = getattr(
                self.cd_config, "detector_generated_token_ids", None
            )
            if tokenizer is None or runtime is None or generated is None:
                raise ValueError(
                    "Detector-Grounded ASCD requires tokenizer, OWLv2 runtime, "
                    "and per-image generated-token state"
                )
            next_token_scores, detector_event = apply_detector_object_mask(
                next_token_scores,
                tokenizer=tokenizer,
                generated_token_ids=generated,
                support_scores=runtime.scores,
                threshold=float(getattr(self.cd_config, "detector_threshold")),
                top_k=int(getattr(self.cd_config, "detector_top_k")),
            )
            object_candidates = detector_event["object_candidates"]
            self.cd_config.detector_decoding_steps += 1
            self.cd_config.detector_object_candidates += len(object_candidates)
            self.cd_config.detector_masked_candidates += len(
                detector_event["masked_token_ids"]
            )
            self.cd_config.detector_selection_changes += int(
                detector_event["selection_changed"]
            )
            self.cd_config.detector_no_finite_protections += int(
                detector_event["protected_no_finite"]
            )
            if object_candidates:
                self.cd_config.detector_events.append(detector_event)

        elif soft_grounded:
            tokenizer = getattr(self.cd_config, "soft_grounded_tokenizer", None)
            runtime = getattr(self.cd_config, "soft_grounded_runtime", None)
            generated = getattr(self.cd_config, "soft_grounded_generated_token_ids", None)
            policy = getattr(self.cd_config, "soft_grounded_policy", None)
            if tokenizer is None or runtime is None or generated is None or policy is None:
                raise ValueError(
                    "Soft-Grounded ASCD requires tokenizer, OWLv2 runtime, frozen policy, "
                    "and per-image generated-token state"
                )
            next_token_scores, soft_event = apply_soft_grounded_object_penalty(
                next_token_scores,
                tokenizer=tokenizer,
                generated_token_ids=generated,
                support_scores=runtime.scores,
                top_k=int(policy["top_k"]),
                probability_slope=float(policy["probability_slope"]),
                probability_intercept=float(policy["probability_intercept"]),
                probability_clip_min=float(policy["probability_clip_min"]),
                probability_clip_max=float(policy["probability_clip_max"]),
            )
            object_candidates = soft_event["object_candidates"]
            self.cd_config.soft_grounded_decoding_steps += 1
            self.cd_config.soft_grounded_object_candidates += len(object_candidates)
            self.cd_config.soft_grounded_penalized_candidates += len(
                soft_event["soft_penalized_token_ids"]
            )
            self.cd_config.soft_grounded_selection_changes += int(
                soft_event["selection_changed"]
            )
            self.cd_config.soft_grounded_no_finite_protections += int(
                soft_event["protected_no_finite"]
            )
            if object_candidates:
                self.cd_config.soft_grounded_events.append(soft_event)
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

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        if detector_grounded:
            self.cd_config.detector_generated_token_ids.append(
                int(next_tokens[0].item())
            )

        if soft_grounded:
            self.cd_config.soft_grounded_generated_token_ids.append(
                int(next_tokens[0].item())
            )
        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        if self.cd_config.if_cd and temp_if_vcd:
            if hasattr(self.cd_config, "if_icd") and self.cd_config.if_icd:
                input_ids_icd = torch.cat([input_ids_icd, next_tokens[:, None]], dim=-1)

            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            del outputs_cd
        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs
        

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
