"""Cache-free greedy runtime for the frozen combined VISTA VSV + SLA method."""

from contextlib import contextmanager
import torch
from ascd_vista import pca_direction, sla_mix, vsv_transform


def _layers(self):
    return self.model.layers if hasattr(self.model, "layers") else self.model.model.layers


@contextmanager
def _unmodified(model):
    wrappers = [m for m in model.modules() if hasattr(m, "attn_steer_configs")]
    saved = []
    for module in wrappers:
        saved.append((module, module.cur_config_id,
                      [bool(c.modify_attn) for c in module.attn_steer_configs]))
        module.cur_config_id = 0
        for config in module.attn_steer_configs:
            config.modify_attn = False
    try:
        yield
    finally:
        for module, config_id, values in saved:
            module.cur_config_id = config_id
            for config, value in zip(module.attn_steer_configs, values):
                config.modify_attn = value


def _mask(embeddings):
    return torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)


def _capture_prompt_activations(self, embeddings):
    captured, handles = [], []
    for layer in _layers(self):
        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden[:, -1, :].detach().float().cpu())
        handles.append(layer.register_forward_hook(capture))
    try:
        with _unmodified(self.model):
            self(inputs_embeds=embeddings, attention_mask=_mask(embeddings),
                 use_cache=False, return_dict=True, output_attentions=False,
                 output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(_layers(self)):
        raise RuntimeError("VISTA prompt activation capture missed transformer layers")
    return torch.cat(captured, dim=0)


@contextmanager
def _vsv_injection(self, direction, strength, event_sink):
    handles = []
    for index, layer in enumerate(_layers(self)):
        def inject(_module, _inputs, output, layer_index=index):
            transformed, event = vsv_transform(output, direction[layer_index], strength)
            event["layer"] = layer_index
            event_sink.append(event)
            return transformed
        handles.append(layer.mlp.register_forward_hook(inject))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _embeddings(self, base, generated):
    if not generated:
        return base
    ids = torch.tensor([generated], device=base.device, dtype=torch.long)
    return torch.cat((base, self.get_input_embeddings()(ids)), dim=1)


def greedy_search(self, input_ids, logits_processor, stopping_criteria,
                  eos_token_id, streamer, model_kwargs):
    required = ("inputs_embeds", "inputs_embeds_vista_null")
    if input_ids.shape[0] != 1 or any(name not in model_kwargs for name in required):
        raise ValueError("VISTA requires batch one and visual/null-prompt embeddings")
    visual_base = model_kwargs["inputs_embeds"]
    null_base = model_kwargs["inputs_embeds_vista_null"]
    negative = _capture_prompt_activations(self, null_base)
    positive = _capture_prompt_activations(self, visual_base)
    direction_cpu = pca_direction(negative, positive)
    direction = direction_cpu.to(device=visual_base.device, dtype=torch.float32)
    direction_norms = direction.float().norm(dim=-1)
    if not torch.isfinite(direction_norms).all() or torch.any(direction_norms <= 0):
        raise RuntimeError("VISTA produced a non-finite or zero VSV direction")
    strength = float(getattr(self.cd_config, "vista_vsv_lambda", 0.17))
    start = int(getattr(self.cd_config, "vista_sla_start_layer", 25))
    end = int(getattr(self.cd_config, "vista_sla_end_layer", 30))
    alpha = float(getattr(self.cd_config, "vista_sla_alpha", 0.3))
    maximum = int(getattr(self.cd_config, "vista_max_new_tokens", 512))
    if start < 0 or end >= len(_layers(self)) or start > end:
        raise ValueError("VISTA SLA layer window is invalid for this backbone")
    eos = {int(eos_token_id)} if isinstance(eos_token_id, int) else {int(x) for x in eos_token_id}
    generated, events = [], []
    while len(generated) < maximum:
        embeddings = _embeddings(self, visual_base, generated)
        injection_events = []
        with _unmodified(self.model), _vsv_injection(self, direction, strength, injection_events):
            output = self(inputs_embeds=embeddings, attention_mask=_mask(embeddings),
                          use_cache=False, return_dict=True, output_attentions=False,
                          output_hidden_states=True)
        if len(injection_events) != len(_layers(self)):
            raise RuntimeError("VISTA VSV injection did not execute once per layer")
        early = [self.lm_head(output.hidden_states[i + 1][:, -1, :])
                 for i in range(start, end + 1)]
        final_direct = output.logits[:, -1, :]
        final = sla_mix(final_direct, early, alpha)
        prefix = torch.cat((input_ids, torch.tensor([generated], device=input_ids.device,
                                                   dtype=torch.long)), dim=-1)
        token = int(torch.argmax(logits_processor(prefix, final), dim=-1)[0].item())
        direct_top = int(torch.argmax(final_direct, dim=-1)[0].item())
        schedules = [v for e in injection_events for v in (e["schedule_min"], e["schedule_max"])]
        events.append({"step": len(generated) + 1, "selected_token_id": token,
                       "vsv_final_top_token_id": direct_top,
                       "sla_top_token_id": int(torch.argmax(final, dim=-1)[0].item()),
                       "sla_changed_from_vsv_final_top": token != direct_top,
                       "vsv_schedule_min": min(schedules), "vsv_schedule_max": max(schedules),
                       "vsv_layer_events": injection_events})
        generated.append(token)
        if streamer is not None:
            streamer.put(torch.tensor([token]).cpu())
        result = torch.cat((input_ids, torch.tensor([generated], device=input_ids.device,
                                                   dtype=torch.long)), dim=-1)
        if token in eos or stopping_criteria(result, None).any():
            break
    if streamer is not None:
        streamer.end()
    self.cd_config.vista_records.append({
        "image_id": int(self.cd_config.vista_image_id), "parent": "unmodified",
        "generated_tokens": len(generated), "vsv_extraction_forwards": 2,
        "generation_forwards": len(generated),
        "direction_norm_min": float(direction_norms.min().item()),
        "direction_norm_max": float(direction_norms.max().item()),
        "positive_negative_difference_norm": float((positive - negative).norm().item()),
        "sla_changed_steps": sum(e["sla_changed_from_vsv_final_top"] for e in events),
        "events": events})
    return result
