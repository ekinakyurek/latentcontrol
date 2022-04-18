import os
from pathlib import Path
import torch
import torch.nn as nn
from absl import logging
from transformers import GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM
from .postfix_tuner import GPT2PostfixLM, GPTJPostfixLM, GPTNeoPostfixLM
from .prompt_coder import (
    GPT2PromptCoderLM,
    GPTJPromptCoderLM,
    GPTNeoPromptCoderLM,
)


class GPTPromptTuningMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        soft_prompt_path: str = None,
        n_tokens: int = None,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        padding_idx: int = None,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        if not hasattr(model.config, "n_embd"):
            model.config.n_embd = model.config.hidden_size

        # # Make sure to freeze Tranformers model
        for name, param in model.named_parameters():
            if "coder" not in name:
                param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

        model.disable = False

        if padding_idx is not None:
            model.transformer.wte.padding_idx = padding_idx

        return model

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.n_tokens = self.soft_prompt.num_embeddings
        logging.info(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
    ) -> None:

        self.n_tokens = n_tokens

        if initialize_from_vocab:
            init_prompt_value = (
                self.transformer.wte.weight[:n_tokens].clone().detach()
            )
        else:
            init_prompt_value = torch.FloatTensor(
                n_tokens, self.config.n_embd
            ).uniform_(-random_range, random_range)

        self.soft_prompt = nn.Embedding(n_tokens, self.config.n_embd)
        # Initialize weight
        self.soft_prompt.weight.data = init_prompt_value

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(
            inputs_embeds.size(0), 1, 1
        )

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full(
                    (n_batches, self.n_tokens),
                    ignore_index,
                    device=labels.device,
                    dtype=labels.dtype,
                ),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask_for_prompts(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [
                torch.ones(
                    n_batches,
                    self.n_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
                attention_mask,
            ],
            dim=1,
        )

    def _extend_attention_mask_to_right(self, input_attention_mask, n_steps=1):

        if len(list(input_attention_mask.shape)) == 1:
            input_attention_mask = input_attention_mask.unsqueeze(0)

        n_batches = input_attention_mask.shape[0]

        return torch.cat(
            [
                input_attention_mask,
                torch.ones(
                    n_batches,
                    n_steps,
                    device=input_attention_mask.device,
                    dtype=input_attention_mask.dtype,
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
            if attention_mask.shape[-1] == input_ids.shape[-1]:
                attention_mask = self._extend_attention_mask_for_prompts(
                    attention_mask
                ).to(self.device)
                input_lengths = kwargs.get("input_lengths")
                if input_lengths is not None:
                    kwargs["input_lengths"] = input_lengths + self.n_tokens

            # position_ids = attention_mask[:, self.n_tokens:].long().cumsum(-1) - 1
            # position_ids = torch.cat(
            #         [position_ids.new_ones((position_ids.shape[0], self.n_tokens)), position_ids], dim=-1
            #     )
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
        }

        if kwargs.get("input_lengths") is not None:
            input["input_lengths"] = kwargs.get("input_lengths")

        return input

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        logging.info(f"Saved soft prompt: {os.path.join(path, filename)}")

    def forward(
        self,
        input_ids=None,
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

        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                self.device
            )

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            if (
                attention_mask.shape[-1]
                == inputs_embeds.shape[1] - self.n_tokens
            ):
                attention_mask = self._extend_attention_mask_for_prompts(
                    attention_mask
                ).to(self.device)

        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # position_ids = attention_mask[:, self.n_tokens:].long().cumsum(-1) - 1
            # position_ids = torch.cat(
            #         [position_ids.new_ones((position_ids.shape[0], self.n_tokens)), position_ids], dim=-1
            #     )
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            input_lengths = kwargs.get("input_lengths")
            if input_lengths is not None:
                kwargs["input_lengths"] = input_lengths + self.n_tokens
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        kwargs["position_ids"] = position_ids

        output = super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            **kwargs,
        )
        if not hasattr(output, "attention_mask"):
            output.attention_mask = attention_mask
        return output


class GPT2PromptTuningLM(GPTPromptTuningMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class GPTNeoPromptTuningLM(GPTPromptTuningMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)


class GPTJPromptTuningLM(GPTPromptTuningMixin, GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)


class GPT2PromptTuningCoderLM(GPTPromptTuningMixin, GPT2PromptCoderLM):
    def __init__(self, config):
        super().__init__(config)


class GPTNeoPromptTuningCoderLM(GPTPromptTuningMixin, GPTNeoPromptCoderLM):
    def __init__(self, config):
        super().__init__(config)


class GPTJPromptTuningCoderLM(GPTPromptTuningMixin, GPTJPromptCoderLM):
    def __init__(self, config):
        super().__init__(config)


class GPT2PromptTuningPostfixLM(GPTPromptTuningMixin, GPT2PostfixLM):
    def __init__(self, config):
        super().__init__(config)


class GPTNeoPromptTuningPostfixLM(GPTPromptTuningMixin, GPTNeoPostfixLM):
    def __init__(self, config):
        super().__init__(config)


class GPTJPromptTuningPostfixLM(GPTPromptTuningMixin, GPTJPostfixLM):
    def __init__(self, config):
        super().__init__(config)
