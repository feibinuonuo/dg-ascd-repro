#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        images_cd=None,   # useless here, but for the debugfree of _validate_model_kwargs()
        inputs_embeds_vcd=None,   # useless here, but for the debugfree of _validate_model_kwargs()
        inputs_embeds_icd=None,   # useless here, but for the debugfree of _validate_model_kwargs()
        input_ids_icd=None,   # useless here, but for the debugfree of _validate_model_kwargs()
        inputs_embeds_mfcd_high=None,  # consumed by the frozen MFCD greedy path
        inputs_embeds_mfcd_low=None,   # consumed by the frozen MFCD greedy path
        inputs_embeds_inter_random_image=None,
        inputs_embeds_inter_empty_text=None,
        inputs_embeds_inter_random_empty=None,
        inputs_embeds_fuzzycd_0=None,
        inputs_embeds_fuzzycd_1=None,
        inputs_embeds_fuzzycd_2=None,
        inputs_embeds_fuzzycd_3=None,
        inputs_embeds_crops_language=None,
        inputs_embeds_vista_null=None,
        key_position=None,
        vad=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        if key_position is not None and vad is not None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                key_position=key_position,
                vad=vad
            )
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        images_mfcd_high = kwargs.pop("images_mfcd_high", None)
        images_mfcd_low = kwargs.pop("images_mfcd_low", None)
        images_inter_random = kwargs.pop("images_inter_random", None)
        input_ids_inter_empty = kwargs.pop("input_ids_inter_empty", None)
        images_fuzzycd = kwargs.pop("images_fuzzycd", None)
        input_ids_crops_language = kwargs.pop("input_ids_crops_language", None)
        input_ids_vista_null = kwargs.pop("input_ids_vista_null", None)
        if images_fuzzycd is not None and (images_fuzzycd.ndim != 4 or images_fuzzycd.shape[0] != 4):
            raise ValueError("FuzzyCD requires four filtered images")
        inputs_fuzzycd = inputs.clone() if images_fuzzycd is not None else None
        if (images_inter_random is None) != (input_ids_inter_empty is None):
            raise ValueError("INTER requires both random image and empty-text input ids")
        inputs_inter_question = inputs.clone() if images_inter_random is not None else None
        if (images_mfcd_high is None) != (images_mfcd_low is None):
            raise ValueError("MFCD requires both high-pass and low-pass images")
        if images_mfcd_high is not None:
            inputs_mfcd_high = inputs.clone()
            inputs_mfcd_low = inputs.clone()
            position_ids_mfcd = position_ids.clone() if isinstance(position_ids, torch.Tensor) else None
            attention_mask_mfcd = attention_mask.clone() if isinstance(attention_mask, torch.Tensor) else None

        ############ for vcd ############
        if "images_cd" in kwargs and kwargs["images_cd"] is not None:
            inputs_vcd = inputs.clone() if isinstance(inputs, torch.Tensor) else None
            position_ids_vcd = position_ids.clone() if isinstance(position_ids, torch.Tensor) else None
            attention_mask_vcd = attention_mask.clone() if isinstance(attention_mask, torch.Tensor) else None
        #################################

        ############ for icd ############
        if "input_ids_icd" in kwargs and kwargs["input_ids_icd"] is not None:
            # inputs_icd = inputs.clone() if isinstance(inputs, torch.Tensor) else None
            position_ids_icd = position_ids.clone() if isinstance(position_ids, torch.Tensor) else None
            attention_mask_icd = attention_mask.clone() if isinstance(attention_mask, torch.Tensor) else None
        #################################

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if images_mfcd_high is not None:
            (_, _, _, _, inputs_embeds_mfcd_high, _) = self.prepare_inputs_labels_for_multimodal(
                inputs_mfcd_high, position_ids_mfcd, attention_mask_mfcd,
                None, None, images_mfcd_high, image_sizes=image_sizes,
            )
            (_, _, _, _, inputs_embeds_mfcd_low, _) = self.prepare_inputs_labels_for_multimodal(
                inputs_mfcd_low, position_ids_mfcd, attention_mask_mfcd,
                None, None, images_mfcd_low, image_sizes=image_sizes,
            )
            kwargs["inputs_embeds_mfcd_high"] = inputs_embeds_mfcd_high
            kwargs["inputs_embeds_mfcd_low"] = inputs_embeds_mfcd_low

        if images_inter_random is not None:
            inter_empty_ids = input_ids_inter_empty.clone()
            (_, _, _, _, embeds_inter_random_image, _) = self.prepare_inputs_labels_for_multimodal(
                inputs_inter_question, None, None, None, None,
                images_inter_random, image_sizes=image_sizes,
            )
            (_, _, _, _, embeds_inter_empty_text, _) = self.prepare_inputs_labels_for_multimodal(
                inter_empty_ids, None, None, None, None,
                images, image_sizes=image_sizes,
            )
            (_, _, _, _, embeds_inter_random_empty, _) = self.prepare_inputs_labels_for_multimodal(
                inter_empty_ids.clone(), None, None, None, None,
                images_inter_random, image_sizes=image_sizes,
            )
            kwargs["inputs_embeds_inter_random_image"] = embeds_inter_random_image
            kwargs["inputs_embeds_inter_empty_text"] = embeds_inter_empty_text
            kwargs["inputs_embeds_inter_random_empty"] = embeds_inter_random_empty

        if images_fuzzycd is not None:
            for branch_index in range(4):
                (_, _, _, _, branch_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                    inputs_fuzzycd.clone(), None, None, None, None,
                    images_fuzzycd[branch_index:branch_index + 1], image_sizes=image_sizes,
                )
                kwargs[f"inputs_embeds_fuzzycd_{branch_index}"] = branch_embeds

        if input_ids_crops_language is not None:
            kwargs["inputs_embeds_crops_language"] = self.get_model().embed_tokens(
                input_ids_crops_language
            )
        if input_ids_vista_null is not None:
            kwargs["inputs_embeds_vista_null"] = self.get_model().embed_tokens(
                input_ids_vista_null
            )

        ############ for vcd ############
        if "images_cd" in kwargs and kwargs["images_cd"] is not None:
            (
                inputs_vcd,
                position_ids_vcd,
                attention_mask_vcd,
                _,
                inputs_embeds_vcd,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs_vcd,
                position_ids_vcd,
                attention_mask_vcd,
                None,
                None,
                kwargs["images_cd"],
                image_sizes=image_sizes
            )
            kwargs['inputs_embeds_vcd'] = inputs_embeds_vcd
        #################################

        ############ for icd ############
        if "input_ids_icd" in kwargs and kwargs["input_ids_icd"] is not None:
            if images is not None:
                (
                    inputs_icd,
                    position_ids_icd,
                    attention_mask_icd,
                    _,
                    inputs_embeds_icd,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    kwargs["input_ids_icd"],
                    position_ids_icd,
                    attention_mask_icd,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes
                )
            else:
                inputs_embeds_icd = self.get_model().embed_tokens(kwargs["input_ids_icd"])
            kwargs['inputs_embeds_icd'] = inputs_embeds_icd
        #################################

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    @torch.no_grad()
    def generate_denoise(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        mask_ratio_mean = kwargs.pop("mask_ratio_mean", 0.0)
        mask_ratio_zero = kwargs.pop("mask_ratio_zero", 0.0)
        mask_ratio_pad = kwargs.pop("mask_ratio_pad", 0.0)
        sigma_gauss_noise = kwargs.pop("sigma_gauss_noise", 0.0)
        diff_noise_step = kwargs.pop("diff_noise_step", 0)
        num_noised_embed = kwargs.pop("num_noised_embed", 1)

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        inputs_clone = inputs.clone()

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds_,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds_ = self.get_model().embed_tokens(inputs)

        inputs_embeds_noise_all = []
        for i in range(num_noised_embed):
            if images is not None:
                (
                    inputs_noise,
                    position_ids_noise,
                    attention_mask_noise,
                    _,
                    inputs_embeds_noise,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs_clone,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes,
                    mask_ratio_mean=mask_ratio_mean,
                    mask_ratio_zero=mask_ratio_zero,
                    mask_ratio_pad=mask_ratio_pad,
                    sigma_gauss_noise=sigma_gauss_noise,
                    diff_noise_step=diff_noise_step
                )
            else:
                inputs_embeds_noise = self.get_model().embed_tokens(inputs)
            inputs_embeds_noise_all.append(inputs_embeds_noise)
        inputs_embeds_noise_all = torch.cat(inputs_embeds_noise_all, dim=0)
        inputs_embeds = torch.cat((inputs_embeds_, inputs_embeds_noise_all), 0)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

def prepare_inputs_for_generation_cd(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values:
        input_ids = input_ids[:, -1:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images_cd", None),
        }
    )
    return model_inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
