import math
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from transformers import GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM


class GPTPostfixMixin:
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

        if initialize_from_vocab:
            init_prompt_value = (
                self.transformer.wte.weight[:n_steps].clone().detach()
            )
        else:
            init_prompt_value = torch.FloatTensor(
                n_steps, self.config.n_embd
            ).uniform_(-random_range, random_range)

        self.coder = nn.Embedding(n_steps, self.config.n_embd)
        self.coder_query = nn.Embedding(n_steps, self.config.n_embd)
        self.coder_gate = nn.Embedding(n_steps, self.config.n_embd)
        # Initialize weight
        self.coder.weight.data = init_prompt_value
        self.coder.weight.requires_grad_(True)

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
            [
                input_attention_mask,
                torch.ones(
                    n_batches,
                    n_steps,
                    device=self.device,
                    dtype=input_attention_mask.dtype,
                ),
            ],
            dim=1,
        )

    def _extend_attention_mask_for_prompts(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [
                attention_mask,
                torch.ones(
                    n_batches,
                    self.n_steps,
                    device=self.device,
                    dtype=attention_mask.dtype,
                ),
            ],
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
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        input = {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "input_lengths": kwargs.get("input_lengths"),
        }

        return input

    def save_coder(self, path: str, filename: str = "coder.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.coder, os.path.join(path, filename))
        logging.info(f"Saved coder to : {os.path.join(path, filename)}")

    def _divide_inputs(self, input_ids, input_lengths):
        if len(input_ids.shape) == 2:
            return input_ids[:, :input_lengths], input_ids[:, input_lengths:]
        else:
            return (
                input_ids[:, :input_lengths, :],
                input_ids[:, input_lengths:, :],
            )

    def _add_prompt_tokens(self, input_ids, output_ids, value=-100):
        prompt_ids = (
            input_ids.new_ones((input_ids.shape[0], self.n_steps)) * value
        )
        return torch.cat([input_ids, prompt_ids, output_ids], dim=1)

    def _attention(self, prompt_embeds, input_embeds, mask=None):
        # prompt_embeds: T', H
        # input_embeds: B, T, H
        query = prompt_embeds.repeat(input_embeds.shape[0], 1, 1)
        n_dim = input_embeds.shape[-1]

        scores = (query * input_embeds.transpose(-1, -2)) * (
            1.0 / math.sqrt(n_dim)
        )
        if mask is not None:
            scores -= (1 - mask[..., None]) * 1e10
        probs = F.softmax(scores, dim=1)  # B, T', T
        output = probs @ input_embeds  # B, T', H
        return output

    def _template_gates(self, prompt_embeds, input_embeds, mask=None):
        # prompt_embeds: T', H
        # input_embeds: B, T, H
        query = prompt_embeds.repeat(input_embeds.shape[0], 1, 1)
        n_dim = input_embeds.shape[-1]
        scores = (query * input_embeds.transpose(-1, -2)) * (
            1.0 / math.sqrt(n_dim)
        )
        if mask is not None:
            scores -= (1 - mask[..., None]) * 1e10
        gate = scores.sum(dim=1, keepdim=True)
        gate = F.sigmoid(gate)
        return gate

    def _add_prompt_embeds(self, input_embeds, output_embeds, mask=None):

        attention_output = self._attention(
            self.coder_query.weight, input_embeds, mask=mask
        )
        template_gate = self._template_gates(
            self.coder_gate.weight, input_embeds, mask=mask
        )
        prompt_embeds = self.coder.weight[None, ...]

        prompt_embeds = (
            template_gate * attention_output
            + (1 - template_gate) * self.coder.weight
        )

        return torch.cat([input_embeds, prompt_embeds, output_embeds], dim=1)

    def forward(
        self,
        input_ids=None,
        input_lengths=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        **kwargs,
    ):
        if self.disable or past_key_values is not None:

            output = super().forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                **kwargs,
            )

            if attention_mask is not None and not hasattr(
                output, "attention_mask"
            ):
                output.attention_mask = attention_mask

            return output

        if input_lengths is not None:

            if attention_mask is not None:
                (
                    input_attention_mask,
                    output_attention_mask,
                ) = self._divide_inputs(attention_mask, input_lengths)

                attention_mask = self._add_prompt_tokens(
                    input_attention_mask, output_attention_mask, 1
                )
                #  update position ids
                # position_ids = kwargs.get('position_ids')

                # if position_ids is None:

                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values is not None:
                    position_ids = position_ids[:, -1].unsqueeze(-1)
                # else:
                #     position_ids_offset = position_ids[:, :input_lengths].max(-1).values
                #     position_ids_output = position_ids_offset.unsqueeze(-1) + attention_mask[:, input_lengths:].long().cumsum(-1)
                #     position_ids = torch.cat([position_ids[:, :input_lengths], position_ids_output], dim=-1)

                #     if past_key_values is not None:
                #         position_ids = position_ids[:, -1].unsqueeze(-1)

                kwargs["position_ids"] = position_ids

            if labels is not None:
                input_labels, output_labels = self._divide_inputs(
                    labels, input_lengths
                )
                labels = self._add_prompt_tokens(
                    input_labels, output_labels, -100
                )

            if input_ids is not None:
                input_ids, output_ids = self._divide_inputs(
                    input_ids, input_lengths
                )
                inputs_embeds = self.transformer.wte(input_ids)
                output_embeds = self.transformer.wte(output_ids)
            elif inputs_embeds is not None:
                inputs_embeds, output_embeds = self._divide_inputs(
                    inputs_embeds, input_lengths
                )

            inputs_embeds = self._add_prompt_embeds(
                inputs_embeds, output_embeds, mask=input_attention_mask
            )
            input_ids = None

        else:
            logging.error("input lengths empty or both")

        output = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        if not hasattr(output, "attention_mask"):
            output.attention_mask = attention_mask

        return output


class GPT2PostfixLM(GPTPostfixMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class GPTNeoPostfixLM(GPTPostfixMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)


class GPTJPostfixLM(GPTPostfixMixin, GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
