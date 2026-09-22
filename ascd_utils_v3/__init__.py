from transformers.models.llama.modeling_llama import LlamaAttention as LlamaAttentionFromTransformers
from transformers.models.mistral.modeling_mistral import MistralAttention
# from transformers import LlamaForCausalLM, LlamaModel
from .ascd_models_v3 import (
    LlamaAttentionDenoiseV1,
    LlamaAttentionDenoiseV1InLavis,
    LlamaAttentionDenoiseV2
    )
from .ascd_models_v3_mistral import MistralAttentionDenoiseV1
from .ascd_models_v3_phi import PhiAttentionDenoiseV1, PhiSdpaAttentionDenoiseV1
from llava.model.language_model.llava_llama import LlavaLlamaModel
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM, LlamaAttention as LlamaAttentionFromLavis, LlamaModel
from llava.model.language_model.llava_mistral import LlavaMistralModel

from transformers.models.phi.modeling_phi import PhiForCausalLM, PhiAttention, PhiSdpaAttention, PhiModel

CONTRASTIVE_ATTN_MAPPING = {
    'hall_attn_v1': {LlamaAttentionFromTransformers: LlamaAttentionDenoiseV1,
                     LlamaAttentionFromLavis: LlamaAttentionDenoiseV1InLavis,
                     MistralAttention: MistralAttentionDenoiseV1,
                     PhiAttention: PhiAttentionDenoiseV1,
                     PhiSdpaAttention: PhiSdpaAttentionDenoiseV1},
    'hall_attn_v2': {LlamaAttentionFromTransformers: LlamaAttentionDenoiseV2},
    }

MODULE_NAME_TEMPLATE = {
    LlavaLlamaModel: r"layers\.(\d+)\.self_attn",
    LlamaForCausalLM: r"model.layers\.(\d+)\.self_attn",
    LlavaMistralModel: r"layers\.(\d+)\.self_attn",
    PhiForCausalLM: r"model.layers\.(\d+)\.self_attn"
}

def find_attn_class(llm):
    if type(llm) == LlavaLlamaModel:
        return type(llm.layers[0].self_attn)
    elif type(llm) == LlamaForCausalLM:
        return type(llm.model.layers[0].self_attn)
    elif type(llm) == LlamaModel:
        return type(llm.layers[0].self_attn)
    elif type(llm) == LlavaMistralModel:
        return type(llm.layers[0].self_attn)
    elif type(llm) == PhiForCausalLM:
        return type(llm.model.layers[0].self_attn)
    elif type(llm) == PhiModel:
        return type(llm.layers[0].self_attn)
    else:
        raise NotImplementedError()
    
