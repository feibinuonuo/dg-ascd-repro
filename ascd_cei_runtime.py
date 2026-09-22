"""Isolated cache-free greedy runtime for the frozen Static CEI adaptation."""

from contextlib import contextmanager

import torch

from ascd_cei import blend_last_hidden


@contextmanager
def injection(model, context_embedding, layer_index, alpha, event_sink=None):
    """Temporarily install the released CEI hidden-state blending hook."""
    layers = model.layers if hasattr(model, "layers") else model.model.layers
    resolved_layer = int(layer_index)
    if resolved_layer < 0:
        resolved_layer += len(layers)
    if not 0 <= resolved_layer < len(layers):
        raise ValueError(f"CEI injection layer {layer_index} is out of range")

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        before = hidden[:, -1, :]
        blended = blend_last_hidden(hidden, context_embedding, alpha)
        if event_sink is not None:
            context = context_embedding
            if context.ndim == 1:
                context = context.unsqueeze(0)
            context = context.to(device=before.device, dtype=before.dtype)
            event_sink.append({
                "pre_hidden_norm": float(before.float().norm(dim=-1)[0].item()),
                "context_norm": float(context.float().norm(dim=-1)[0].item()),
                "post_hidden_norm": float(blended[:, -1, :].float().norm(dim=-1)[0].item()),
                "pre_context_cosine": float(torch.nn.functional.cosine_similarity(
                    before.float(), context.float(), dim=-1
                )[0].item()),
            })
        if isinstance(output, tuple):
            return (blended,) + output[1:]
        return blended

    handle = layers[resolved_layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def greedy_search(
    self,
    input_ids,
    logits_processor,
    stopping_criteria,
    eos_token_id,
    streamer,
    model_kwargs,
    context_embedding_fn,
    parent_logits_fn,
):
    """Generate with one fixed prompt-context extraction and per-step CEI."""
    if input_ids.shape[0] != 1 or "inputs_embeds" not in model_kwargs:
        raise ValueError("Static CEI requires batch one and expanded multimodal embeddings")
    base_embeddings = model_kwargs["inputs_embeds"]
    maximum = int(getattr(self.cd_config, "cei_max_new_tokens", 512))
    injection_layer = int(getattr(self.cd_config, "cei_injection_layer", 10))
    alpha = float(getattr(self.cd_config, "cei_alpha", 0.1))
    eos = {int(eos_token_id)} if isinstance(eos_token_id, int) else {
        int(token_id) for token_id in eos_token_id
    }

    context_embedding = context_embedding_fn(base_embeddings).detach()
    if context_embedding.shape != (1, base_embeddings.shape[-1]):
        raise RuntimeError(
            f"Unexpected CEI context shape {tuple(context_embedding.shape)}"
        )

    generated = []
    events = []
    for step in range(1, maximum + 1):
        if generated:
            generated_ids = torch.tensor(
                [generated], device=base_embeddings.device, dtype=torch.long
            )
            embeddings = torch.cat(
                (base_embeddings, self.get_input_embeddings()(generated_ids)), dim=1
            )
        else:
            embeddings = base_embeddings
        hook_events = []
        logits = parent_logits_fn(
            embeddings, context_embedding, injection_layer, alpha, hook_events
        )
        if len(hook_events) != 1:
            raise RuntimeError(
                f"CEI expected one injected positive forward, observed {len(hook_events)}"
            )
        prefix = torch.cat((input_ids, torch.tensor(
            [generated], device=input_ids.device, dtype=torch.long
        )), dim=-1)
        token = int(torch.argmax(logits_processor(prefix, logits), dim=-1)[0].item())
        event = dict(hook_events[0])
        event.update({"step": step, "selected_token_id": token})
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
    self.cd_config.cei_records.append({
        "image_id": int(self.cd_config.cei_image_id),
        "parent": "fixed_ascd" if bool(getattr(self.cd_config, "if_cd", False)) else "unmodified",
        "generated_tokens": len(generated),
        "context_extraction_forwards": 1,
        "injected_positive_forwards": len(generated),
        "negative_forwards": len(generated) if bool(getattr(self.cd_config, "if_cd", False)) else 0,
        "events": events,
    })
    return result
