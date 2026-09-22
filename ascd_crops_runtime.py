"""Isolated greedy runtime for the frozen CRoPS adaptation."""

from contextlib import contextmanager

import torch

from ascd_crops import calibrate, minimum_text_tokens


def _wrappers(model):
    return [module for module in model.modules() if hasattr(module, "attn_steer_configs")]


@contextmanager
def _branch(model, mode, state, unmodified):
    wrappers = _wrappers(model)
    saved = []
    for module in wrappers:
        configs = module.attn_steer_configs
        saved.append((module, module.cur_config_id,
                      getattr(module, "crops_branch_mode", None),
                      getattr(module, "crops_branch_state", None),
                      [bool(config.modify_attn) for config in configs]))
        module.cur_config_id = 0
        module.crops_branch_mode = mode
        module.crops_branch_state = state
        if unmodified:
            for config in configs:
                config.modify_attn = False
    try:
        yield
    finally:
        for module, config_id, old_mode, old_state, modify_values in saved:
            module.cur_config_id = config_id
            module.crops_branch_mode = old_mode
            module.crops_branch_state = old_state
            for config, value in zip(module.attn_steer_configs, modify_values):
                config.modify_attn = value


def _embeddings(self, base, generated):
    if not generated:
        return base
    ids = torch.tensor([generated], device=base.device, dtype=torch.long)
    return torch.cat((base, self.get_input_embeddings()(ids)), dim=1)


def _forward(self, embeddings, mode=None, state=None, unmodified=False):
    mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)
    with _branch(self.model, mode, state, unmodified):
        return self(inputs_embeds=embeddings, attention_mask=mask, use_cache=False,
                    return_dict=True, output_attentions=False,
                    output_hidden_states=False).logits[:, -1, :]


def greedy_search(self, input_ids, logits_processor, stopping_criteria,
                  eos_token_id, streamer, model_kwargs, parent_logits_fn):
    required = ("inputs_embeds", "inputs_embeds_crops_language")
    if input_ids.shape[0] != 1 or any(key not in model_kwargs for key in required):
        raise ValueError("CRoPS requires batch one and direct/language embeddings")
    base_direct = model_kwargs["inputs_embeds"]
    base_language = model_kwargs["inputs_embeds_crops_language"]
    maximum = int(getattr(self.cd_config, "crops_max_new_tokens", 512))
    eos = {int(eos_token_id)} if isinstance(eos_token_id, int) else {int(x) for x in eos_token_id}
    generated, events = [], []
    aggregate_layer = int(getattr(self.cd_config, "crops_aggregate_layer", 2))
    keep_fraction = float(getattr(self.cd_config, "crops_visual_keep_fraction", 0.25))
    while len(generated) < maximum:
        step = len(generated) + 1
        direct_embeddings = _embeddings(self, base_direct, generated)
        language_embeddings = _embeddings(self, base_language, generated)
        direct_logits = parent_logits_fn(direct_embeddings)
        text_state = {"aggregate_layer": aggregate_layer,
                      "minimum_text_tokens": minimum_text_tokens(step)}
        language_logits = _forward(
            self, language_embeddings, "language_prior", text_state, True
        )
        visual_state = {"aggregate_layer": aggregate_layer,
                        "visual_keep_fraction": keep_fraction}
        statistical_logits = _forward(
            self, direct_embeddings, "visual_statistical", visual_state, True
        )
        final, event = calibrate(
            direct_logits, language_logits, statistical_logits, step=step,
            lambda_lang_prior=float(getattr(self.cd_config, "crops_lambda_lang_prior", 0.01)),
            alpha_stat_bias=float(getattr(self.cd_config, "crops_alpha_stat_bias", 1.0)),
            beta_cutoff=float(getattr(self.cd_config, "crops_beta_cutoff", 0.1)),
            max_threshold_plausibility_constraint=float(
                getattr(self.cd_config, "crops_max_plausibility", 0.95)),
        )
        prefix = torch.cat((input_ids, torch.tensor(
            [generated], device=input_ids.device, dtype=torch.long)), dim=-1)
        token = int(torch.argmax(logits_processor(prefix, final), dim=-1)[0].item())
        event.update({
            "selected_token_id": token,
            "selected_changed_from_direct_top": token != event["direct_top_token_id"],
            "language_selected_key_count": int(text_state.get("selected_key_count", 0)),
            "language_key_length": int(text_state.get("key_length", 0)),
            "statistical_selected_key_count": int(visual_state.get("selected_key_count", 0)),
            "statistical_key_length": int(visual_state.get("key_length", 0)),
        })
        events.append(event)
        generated.append(token)
        if streamer is not None:
            streamer.put(torch.tensor([token]).cpu())
        result = torch.cat((input_ids, torch.tensor(
            [generated], device=input_ids.device, dtype=torch.long)), dim=-1)
        if token in eos or stopping_criteria(result, None).any():
            break
    if streamer is not None:
        streamer.end()
    self.cd_config.crops_records.append({
        "image_id": int(self.cd_config.crops_image_id),
        "parent": "fixed_ascd" if bool(getattr(self.cd_config, "if_cd", False)) else "unmodified",
        "generated_tokens": len(generated),
        "forward_calls": len(generated) * (4 if bool(getattr(self.cd_config, "if_cd", False)) else 3),
        "bypass_steps": sum(event["plausibility_bypass"] for event in events),
        "changed_steps": sum(event["selected_changed_from_direct_top"] for event in events),
        "events": events,
    })
    return result
