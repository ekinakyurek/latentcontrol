import os
from pathlib import Path
import torch
import torch.nn as nn
from absl import logging
from transformers import GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM


class GPTPromptCoderMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        coder_path: str = None,
        n_steps: int = None,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
        padding_idx: int = None,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        if not hasattr(model.config, "n_embd"):
            model.config.n_embd = model.config.hidden_size
        # Make sure to freeze Tranformers model
        if coder_path is not None:
            for param in model.parameters():
                param.requires_grad = False

            model.set_coder_layers(coder_path)

        elif n_steps is not None:
            for param in model.parameters():
                param.requires_grad = False

            model.initialize_coder(
                n_steps=n_steps,
                random_range=random_range,
                initialize_from_vocab=initialize_from_vocab,
            )

            model.disable = False

        elif n_steps is None:
            model.disable = True
            logging.warning("n_steps is None, no prompt paramaters")

        model.loss_fn = nn.CrossEntropyLoss()

        if padding_idx is not None:
            model.transformer.wte.padding_idx = padding_idx

        return model

    def set_coder_layers(
        self,
        coder_path: str,
    ) -> None:
        """
        Args:
            coder_path: torch soft prompt file path
        """
        self.coder = torch.load(coder_path, map_location=torch.device("cpu"))
        self.n_steps = len(self.coder)
        print(f"Set soft prompt! (n_steps: {self.n_steps})")

    def initialize_coder(
        self,
        n_steps: int = 20,
        random_range: float = 0.01,
        initialize_from_vocab: bool = True,
    ) -> None:

        self.n_steps = n_steps
        coder = []
        for step in range(self.n_steps + 1):
            init_weight_value = (
                torch.randn(self.config.n_embd, self.config.n_embd) * random_range
            )

            if initialize_from_vocab:
                init_bias_value = self.transformer.wte.weight.mean(dim=0, keepdims=True)
            else:
                init_bias_value = torch.FloatTensor(
                    n_steps, self.config.n_embd
                ).uniform_(-random_range, random_range)

            layer = nn.Linear(self.config.n_embd, self.config.n_embd)
            layer.weight.data = init_weight_value
            layer.weight.requires_grad_(True)
            layer.bias.data = init_bias_value
            layer.bias.requires_grad_(True)
            coder.append(layer)

        self.coder = nn.ModuleList(coder)

    def _cat_latent_embeddings_to_input(
        self, input_embeds, latent_embeds
    ) -> torch.Tensor:
        inputs_embeds = torch.cat([input_embeds, latent_embeds], dim=1)
        return inputs_embeds

    def _extend_attention_mask(self, input_attention_mask, n_steps=1):

        if len(list(input_attention_mask.shape)) == 1:
            input_attention_mask = input_attention_mask.unsqueeze(0)

        n_batches = input_attention_mask.shape[0]

        return torch.cat(
            [input_attention_mask, torch.full((n_batches, n_steps), 1).to(self.device)],
            dim=1,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "input_lengths": kwargs.get("input_lengths"),
        }

    def save_coder(self, path: str, filename: str = "coder.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.coder, os.path.join(path, filename))
        logging.info(f"Saved coder to : {os.path.join(path, filename)}")

    def _divide_inputs(self, input_ids, input_lengths):
        if len(input_ids.shape) == 2:
            return input_ids[:, :input_lengths], input_ids[:, input_lengths:]
        else:
            return input_ids[:, :input_lengths, :], input_ids[:, input_lengths:, :]

    def forward(
        self,
        input_ids=None,
        input_lengths=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        **kwargs,
    ):
        if self.disable or past_key_values is not None:

            output = super().forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=True,
                past_key_values=past_key_values,
                **kwargs,
            )
            if attention_mask is not None and not hasattr(output, "attention_mask"):
                output.attention_mask = attention_mask

            return output

        if input_lengths is not None:
            if input_ids is not None:
                input_ids, output_ids = self._divide_inputs(input_ids, input_lengths)
                inputs_embeds = self.transformer.wte(input_ids)
                output_embeds = self.transformer.wte(output_ids)
            elif inputs_embeds is not None:
                inputs_embeds, output_embeds = self._divide_inputs(
                    inputs_embeds, input_lengths
                )

            if labels is not None:
                input_labels, output_labels = self._divide_inputs(labels, input_lengths)

            if attention_mask is not None:
                attention_mask, output_attention_mask = self._divide_inputs(
                    attention_mask, input_lengths
                )
        else:
            logging.error("input lengths empty or both")

        transformer = self.transformer

        kwargs["output_hidden_states"] = True

        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        elif position_ids is not None and input_lengths is not None:
            position_ids, output_position_ids = self._divide_inputs(
                position_ids, input_lengths
            )

        kwargs["position_ids"] = position_ids

        input_position_ids = position_ids

        output = transformer.forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            **kwargs,
        )

        kwargs.pop("position_ids")

        for step in range(self.n_steps):
            W = self.coder[step].weight
            b = self.coder[step].bias
            latent_embed = output[0][:, -1:, :] @ W + b
            # latent_embed = b.expand(output[0].shape[0], -1).unsqueeze(1)

            attention_mask = self._extend_attention_mask(attention_mask, n_steps=1)

            if position_ids is not None:
                position_ids = position_ids.max(dim=-1, keepdims=True).values + 1

                input_position_ids = torch.cat(
                    [input_position_ids, position_ids], dim=-1
                )

            output = transformer.forward(
                inputs_embeds=latent_embed,
                attention_mask=attention_mask,
                past_key_values=output.past_key_values,
                use_cache=True,
                position_ids=position_ids,
                **kwargs,
            )

        W = self.coder[self.n_steps].weight
        b = self.coder[self.n_steps].bias
        latent_embed = output[0][:, -1:, :] @ W + b

        attention_mask = self._extend_attention_mask(attention_mask, n_steps=1)

        attention_mask = torch.cat([attention_mask, output_attention_mask], dim=1)

        output_embeds = torch.cat([latent_embed, output_embeds], dim=1)

        output_position_ids = output_attention_mask.long().cumsum(-1) + 1

        output_position_ids = torch.cat(
            [torch.ones_like(position_ids), output_position_ids], dim=-1
        )

        output_position_ids += position_ids

        output = transformer.forward(
            inputs_embeds=output_embeds,
            attention_mask=attention_mask,
            past_key_values=output.past_key_values,
            use_cache=True,
            position_ids=output_position_ids,
            **kwargs,
        )

        lm_logits = self.lm_head(output[0]).contiguous()

        # eos_id = transformer.wte.padding_idx
        # lm_logits[:, :, eos_id] -= 9999

        output.logits = lm_logits

        if not hasattr(output, "attention_mask"):
            output.attention_mask = attention_mask

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            output_labels = output_labels.contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), output_labels.view(-1)
            )

            output.loss = loss

        return output


class GPT2PromptCoderLM(GPTPromptCoderMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class GPTNeoPromptCoderLM(GPTPromptCoderMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)


class GPTJPromptCoderLM(GPTPromptCoderMixin, GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
