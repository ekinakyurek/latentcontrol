import torch
import torch.utils.checkpoint
from transformers.models.gptj import modeling_gptj
from transformers.models.gptj.modeling_gptj import BaseModelOutputWithPast
from transformers.models.gptj.modeling_gptj import logger


def fixed_pos_embedding(x, position_ids, seq_dim=1):
    dim = x.shape[-1]
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, dim, 2, device=position_ids.device) / dim)
    )
    sinusoid_inp = (
        torch.einsum("b i , j -> b i j", position_ids, inv_freq)
        .to(x.device)
        .float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


modeling_gptj.fixed_pos_embedding = fixed_pos_embedding


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)

    return x.flatten(
        -2
    )  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


modeling_gptj.rotate_every_two = rotate_every_two


def apply_rotary_pos_emb(x, sincos, offset=0):

    sin, cos = map(
        lambda t: t[:, :, None, :].repeat_interleave(2, 3),
        sincos,
    )

    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


modeling_gptj.apply_rotary_pos_emb = apply_rotary_pos_emb


def gptj_attention_forward(
    self,
    hidden_states,
    position_ids,
    attention_mask=None,
    layer_past=None,
    head_mask=None,
    use_cache=False,
    output_attentions=False,
):

    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(
        query, self.num_attention_heads, self.head_dim, True
    )
    key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
    value = self._split_heads(
        value, self.num_attention_heads, self.head_dim, False
    )

    if self.rotary_dim is not None:
        k_rot = key[:, :, :, : self.rotary_dim]
        k_pass = key[:, :, :, self.rotary_dim :]

        q_rot = query[:, :, :, : self.rotary_dim]
        q_pass = query[:, :, :, self.rotary_dim :]
        sincos = fixed_pos_embedding(k_rot, position_ids, seq_dim=1)
        k_rot = apply_rotary_pos_emb(k_rot, sincos)
        q_rot = apply_rotary_pos_emb(q_rot, sincos)

        key = torch.cat([k_rot, k_pass], dim=-1)
        query = torch.cat([q_rot, q_pass], dim=-1)
    else:
        sincos = fixed_pos_embedding(key, position_ids, seq_dim=1)
        key = apply_rotary_pos_emb(key, sincos)
        query = apply_rotary_pos_emb(query, sincos)

    key = key.permute(0, 2, 1, 3)
    query = query.permute(0, 2, 1, 3)

    if layer_past is not None:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    if use_cache is True:
        present = (key, value)
    else:
        present = None

    # compute self-attention: V x Softmax(QK^T)
    attn_output, attn_weights = self._attn(
        query, key, value, attention_mask, head_mask
    )

    attn_output = self._merge_heads(
        attn_output, self.num_attention_heads, self.head_dim
    )
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)


modeling_gptj.GPTJAttention.forward = gptj_attention_forward


def gptj_block_forward(
    self,
    hidden_states,
    layer_past=None,
    attention_mask=None,
    head_mask=None,
    use_cache=False,
    output_attentions=False,
    position_ids=None,
):
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        position_ids,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]

    feed_forward_hidden_states = self.mlp(hidden_states)
    hidden_states = attn_output + feed_forward_hidden_states + residual

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions)


modeling_gptj.GPTJBlock.forward = gptj_block_forward


def model_forward(
    self,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same"
            " time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError(
            "You have to specify either input_ids or inputs_embeds"
        )

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            input_shape[-1] + past_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # Attention mask.
    if attention_mask is not None:
        assert batch_size > 0, "batch_size has to be defined and > 0"
        attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x num_attention_heads x N x N
    # head_mask has shape n_layer x batch x num_attention_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    hidden_states = inputs_embeds

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(
                    past_state.to(hidden_states.device)
                    for past_state in layer_past
                )
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient"
                    " checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                None,
                attention_mask,
                head_mask[i],
                position_ids=position_ids,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_ids=position_ids,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1],
            )

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                presents,
                all_hidden_states,
                all_self_attentions,
            ]
            if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


modeling_gptj.GPTJModel.forward = model_forward
