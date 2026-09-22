"""Cache-free greedy runtime for the frozen Self-Aug adaptation."""

from contextlib import contextmanager

import torch

from ascd_selfaug import calibrate


@contextmanager
def _unmodified(model):
    wrappers = [module for module in model.modules() if hasattr(module, "attn_steer_configs")]
    saved = []
    for module in wrappers:
        saved.append((module, module.cur_config_id,
                      [bool(config.modify_attn) for config in module.attn_steer_configs]))
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


def _embeddings(self, base, generated):
    if not generated:
        return base
    ids = torch.tensor([generated], device=base.device, dtype=torch.long)
    return torch.cat((base, self.get_input_embeddings()(ids)), dim=1)


def _forward(self, embeddings):
    mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)
    with _unmodified(self.model):
        return self(inputs_embeds=embeddings, attention_mask=mask, use_cache=False,
                    return_dict=True, output_attentions=False,
                    output_hidden_states=False).logits[:, -1, :]


def greedy_search(self, input_ids, logits_processor, stopping_criteria,
                  eos_token_id, streamer, model_kwargs):
    required = ("inputs_embeds", "inputs_embeds_vcd")
    if input_ids.shape[0] != 1 or any(name not in model_kwargs for name in required):
        raise ValueError("Self-Aug requires batch one and visual/augmented embeddings")
    bases = [model_kwargs[name] for name in required]
    if tuple(bases[0].shape) != tuple(bases[1].shape):
        raise ValueError("Self-Aug visual branches have inconsistent embedding shapes")
    maximum = int(getattr(self.cd_config, "selfaug_max_new_tokens", 512))
    eos = {int(eos_token_id)} if isinstance(eos_token_id, int) else {
        int(value) for value in eos_token_id
    }
    generated, events = [], []
    while len(generated) < maximum:
        visual = _forward(self, _embeddings(self, bases[0], generated))
        augmented = _forward(self, _embeddings(self, bases[1], generated))
        final, event = calibrate(
            visual, augmented,
            alpha=float(getattr(self.cd_config, "selfaug_alpha", 1.0)),
            tau=float(getattr(self.cd_config, "selfaug_tau", 0.5)),
        )
        prefix = torch.cat((input_ids, torch.tensor(
            [generated], device=input_ids.device, dtype=torch.long)), dim=-1)
        token = int(torch.argmax(logits_processor(prefix, final), dim=-1)[0].item())
        event.update({"step": len(generated) + 1, "selected_token_id": token})
        events.append(event); generated.append(token)
        if streamer is not None:
            streamer.put(torch.tensor([token]).cpu())
        result = torch.cat((input_ids, torch.tensor(
            [generated], device=input_ids.device, dtype=torch.long)), dim=-1)
        if token in eos or stopping_criteria(result, None).any():
            break
    if streamer is not None:
        streamer.end()
    self.cd_config.selfaug_records.append({
        "image_id": int(self.cd_config.selfaug_image_id),
        "parent": "unmodified",
        "applied_aug": str(self.cd_config.selfaug_applied_aug),
        "generated_tokens": len(generated),
        "forward_calls": 2 * len(generated),
        "changed_steps": sum(event["calibrated_changed_from_visual_top"] for event in events),
        "events": events,
    })
    return result
