
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLSdpaAttention, Qwen2_5_VLAttention

from .ascd_models_v3_qwen import  Qwen2_5_VLAttentionDenoiseV1


CONTRASTIVE_ATTN_MAPPING = {
    'hall_attn_v1': {Qwen2_5_VLSdpaAttention: Qwen2_5_VLAttentionDenoiseV1, Qwen2_5_VLAttention: Qwen2_5_VLAttentionDenoiseV1},
    'hall_attn_v2': {},
    }

MODULE_NAME_TEMPLATE = {
    Qwen2_5_VLModel: r"layers\.(\d+)\.self_attn",
}

def find_attn_class(llm):
    if type(llm) == Qwen2_5_VLModel:
        return type(llm.layers[0].self_attn)
    else:
        raise NotImplementedError()
    
