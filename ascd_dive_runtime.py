"""Cache-free faithful-slow runtime for the frozen DiVE paper adaptation."""

import math
from contextlib import contextmanager

import torch

from ascd_dive import (
    candidate_layer_indices,
    likelihood_ratio_calibration,
    select_visual_evidence_layers,
    suppress_visual_evidence,
)


def _wrappers(model):
    return [module for module in model.modules() if hasattr(module, "attn_steer_configs")]


@contextmanager
def _branch(model, state=None, *, unmodified=False, config_id=0):
    wrappers = _wrappers(model)
    saved = []
    for module in wrappers:
        configs = module.attn_steer_configs
        saved.append((
            module,
            module.cur_config_id,
            getattr(module, "dive_branch_mode", None),
            getattr(module, "dive_branch_state", None),
            [bool(config.modify_attn) for config in configs],
        ))
        module.cur_config_id = int(config_id)
        module.dive_branch_mode = "direct" if state is not None else None
        module.dive_branch_state = state
        if unmodified:
            for config in configs:
                config.modify_attn = False
    try:
        yield
    finally:
        for module, old_id, old_mode, old_state, modify_values in saved:
            module.cur_config_id = old_id
            module.dive_branch_mode = old_mode
            module.dive_branch_state = old_state
            for config, value in zip(module.attn_steer_configs, modify_values):
                config.modify_attn = value


def _attention_mask(embeddings):
    return torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)


def _direct_forward(self, embeddings, state, unmodified):
    with _branch(self.model, state, unmodified=unmodified, config_id=0):
        return self(
            inputs_embeds=embeddings,
            attention_mask=_attention_mask(embeddings),
            use_cache=False,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        ).logits[:, -1, :]


def _reference_forward(self, embeddings, evidence, gamma, epsilon, unmodified):
    layers = self.model.layers if hasattr(self.model, "layers") else self.model.model.layers
    layer_index = len(layers) - 2
    hook_events = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        result = hidden.clone()
        # Clone the audit value: the assignment below mutates the last-token
        # view in `result`, which would otherwise make before/after norms alias.
        before = result[:, -1, :].clone()
        result[:, -1, :] = suppress_visual_evidence(
            before, evidence, gamma=gamma, epsilon=epsilon
        )
        hook_events.append({
            "penultimate_hidden_norm": float(before.float().norm(dim=-1)[0].item()),
            "aggregate_evidence_norm": float(evidence.float().norm(dim=-1)[0].item()),
            "suppressed_hidden_norm": float(result[:, -1, :].float().norm(dim=-1)[0].item()),
        })
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result

    handle = layers[layer_index].register_forward_hook(hook)
    try:
        with _branch(self.model, None, unmodified=unmodified, config_id=0):
            logits = self(
                inputs_embeds=embeddings,
                attention_mask=_attention_mask(embeddings),
                use_cache=False,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            ).logits[:, -1, :]
    finally:
        handle.remove()
    if len(hook_events) != 1:
        raise RuntimeError(f"DiVE reference expected one suppression hook, got {len(hook_events)}")
    return logits, hook_events[0]


def _embeddings(self, base, generated):
    if not generated:
        return base
    ids = torch.tensor([generated], device=base.device, dtype=torch.long)
    return torch.cat((base, self.get_input_embeddings()(ids)), dim=1)


def greedy_search(self, input_ids, logits_processor, stopping_criteria,
                  eos_token_id, streamer, model_kwargs, parent_logits_fn):
    if input_ids.shape[0] != 1 or "inputs_embeds" not in model_kwargs:
        raise ValueError("DiVE requires batch one and expanded multimodal embeddings")
    base = model_kwargs["inputs_embeds"]
    total_layers = len(_wrappers(self.model))
    candidates = candidate_layer_indices(
        total_layers, float(getattr(self.cd_config, "dive_exclusion_ratio", 0.05))
    )
    epsilon = float(getattr(self.cd_config, "dive_epsilon", 1e-6))
    gamma = float(getattr(self.cd_config, "dive_gamma", 0.5))
    threshold = float(getattr(self.cd_config, "dive_threshold", 0.85))
    maximum = int(getattr(self.cd_config, "dive_max_new_tokens", 512))
    unmodified = not bool(getattr(self.cd_config, "if_cd", False))
    eos = {int(eos_token_id)} if isinstance(eos_token_id, int) else {
        int(token_id) for token_id in eos_token_id
    }

    # Paper prefill: one-time V-LAC computation from the last prompt query.
    prefill_state = {
        "candidate_layers": candidates,
        "selected_layers": None,
        "vlac_scores": {},
        "evidence": {},
        "epsilon": epsilon,
    }
    _direct_forward(self, base, prefill_state, unmodified)
    if set(prefill_state["vlac_scores"]) != set(candidates):
        raise RuntimeError("DiVE prefill did not capture every candidate-layer V-LAC")
    scores = torch.zeros(total_layers, device=base.device)
    for index, score in prefill_state["vlac_scores"].items():
        scores[int(index)] = score
    selected = select_visual_evidence_layers(scores, candidates)

    generated, events = [], []
    for step in range(1, maximum + 1):
        embeddings = _embeddings(self, base, generated)
        state = {
            "candidate_layers": candidates,
            "selected_layers": selected,
            "evidence": {},
            "epsilon": epsilon,
        }
        positive = _direct_forward(self, embeddings, state, unmodified)
        if set(state["evidence"]) != set(selected):
            raise RuntimeError("DiVE direct forward did not capture all selected-layer evidence")
        evidence = torch.stack([state["evidence"][index] for index in selected]).mean(dim=0)
        parent = parent_logits_fn(positive, embeddings)
        reference, suppression_event = _reference_forward(
            self, embeddings, evidence, gamma, epsilon, unmodified
        )
        calibrated, event = likelihood_ratio_calibration(
            parent, reference, threshold=threshold
        )
        prefix = torch.cat((input_ids, torch.tensor(
            [generated], device=input_ids.device, dtype=torch.long
        )), dim=-1)
        token = int(torch.argmax(logits_processor(prefix, calibrated), dim=-1)[0].item())
        event.update(suppression_event)
        event.update({
            "step": step,
            "selected_token_id": token,
            "selected_changed_from_parent": token != int(torch.argmax(parent, dim=-1)[0].item()),
        })
        events.append(event)
        generated.append(token)
        if streamer is not None:
            streamer.put(torch.tensor([token]).cpu())
        result = torch.cat((input_ids, torch.tensor(
            [generated], device=input_ids.device, dtype=torch.long
        )), dim=-1)
        if token in eos or stopping_criteria(result, None).any():
            break

    if streamer is not None:
        streamer.end()
    self.cd_config.dive_records.append({
        "image_id": int(self.cd_config.dive_image_id),
        "parent": "fixed_ascd" if not unmodified else "unmodified",
        "candidate_layers": candidates,
        "selected_layers": selected,
        "vlac_scores": {str(i): float(scores[i].item()) for i in candidates},
        "generated_tokens": len(generated),
        "prefill_forwards": 1,
        "direct_forwards": len(generated),
        "reference_forwards": len(generated),
        "negative_forwards": len(generated) if not unmodified else 0,
        "changed_steps": sum(event["selected_changed_from_parent"] for event in events),
        "events": events,
    })
    return result
