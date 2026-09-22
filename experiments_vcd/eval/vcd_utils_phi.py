from typing import List, Tuple, Optional, Union

import torch
import torch.utils.checkpoint


from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput


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
    key_position=None,
    vad=None
) -> Union[Tuple, CausalLMOutputWithPast]:
    use_cache = use_cache if use_cache is not None else self.config.use_cache
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
        return self.language_model.forward(
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
        return self.language_model.forward(
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
        inputs_embeds = self.language_model.get_input_embeddings()(inputs)

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
        kwargs['inputs_embeds_icd'] = inputs_embeds_icd
    #################################

    return self.language_model.generate(
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        **kwargs
    )

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    get_torch_version,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.phi.modeling_phi import (PHI_INPUTS_DOCSTRING,
                                                  _CONFIG_FOR_DOC,
                                                  )
from torch.nn import CrossEntropyLoss


@add_start_docstrings_to_model_forward(PHI_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward_phiforcausallm(
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
    return_dict: Optional[bool] = None,
    cache_position=None,
    images_cd=None,   # useless here, but for the debugfree of _validate_model_kwargs()
    inputs_embeds_vcd=None,   # useless here, but for the debugfree of _validate_model_kwargs()
    inputs_embeds_icd=None,   # useless here, but for the debugfree of _validate_model_kwargs()
    input_ids_icd=None,   # useless here, but for the debugfree of _validate_model_kwargs()
    key_position=None,
    vad=None
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, PhiForCausalLM

    >>> model = PhiForCausalLM.from_pretrained("microsoft/phi-1")
    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")

    >>> prompt = "This is an example script ."
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    'This is an example script .\n\n\n\nfrom typing import List\n\ndef find_most_common_letter(words: List[str'
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
