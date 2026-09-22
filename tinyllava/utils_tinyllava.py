from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def _propagate_attn_implementation(config, attn_implementation):
    if not attn_implementation:
        return

    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        cfg._attn_implementation = attn_implementation
        cfg._attn_implementation_internal = attn_implementation


def load_tinyllava(hf_path, **kwargs):
    attn_implementation = kwargs.get("attn_implementation")
    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    _propagate_attn_implementation(config, attn_implementation)

    model = AutoModelForCausalLM.from_pretrained(hf_path, config=config, trust_remote_code=True, **kwargs)
    model.cuda()
    config = model.config
    first_attn = model.language_model.model.layers[0].self_attn
    print(f"TinyLLaVA first attention class: {first_attn.__class__.__name__}")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
    image_processor = model.vision_tower._image_processor

    return tokenizer, model, image_processor
