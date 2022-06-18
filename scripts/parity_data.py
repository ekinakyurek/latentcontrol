import json
import logging
import math
import pdb
import pickle
from typing import Callable, List, Mapping, NamedTuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from src import utils


class Example(NamedTuple):
    x1: Union[int, str]
    y: Union[int, str]


class ParityDataset(Dataset):

    split_ratios: List = [("train", 0.8), ("dev", 0.1), ("test", 0.1)]

    def __init__(self, split_ratios=None, **kwargs):

        for (k, v) in kwargs.items():
            self.__setattr__(k, v)

        if split_ratios is not None:
            self.split_ratios = split_ratios

        self.data = self.get_data(**kwargs)

    def random_with_n_digits(rng, n: int):
        return rng.random(n) < 0.5

    def save_to_file(self, path: str):
        str_data = [d for d in self]
        with open(path, "wb") as handle:
            if path.endswith("json"):
                json.dump(handle, str_data)
            elif path.endswith("pickle"):
                pickle.dump(str_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                raise ValueError("Unknown data file format")

    def generate_expressions(
        self,
        max_digits: int = 10,
        min_digits: int = 0,
        N_per_digit: int = 200,
        split: str = "train",
        seed: int = 0,
        reversed_outputs=False,
    ) -> List[Example]:

        rng = np.random.default_rng(seed)
        examples = set()
        for e1 in range(max_digits):
            for _ in range(N_per_digit * e1):
                x1 = ParityDataset.random_with_n_digits(rng, e1 + 1)
                y = np.sum(x1) % 2
                y = str(x1.sum())
                x1 = " ".join(x1.astype(int).astype(str))
                examples.add((x1, y))

        examples = list(map(lambda x: Example(*x), examples))

        seed, new_seed = utils.split_seed(seed)

        rng = np.random.default_rng(new_seed)
        rng.shuffle(examples)
        return examples

    def get_split(self, examples, split: str = "train") -> List[Example]:
        L = len(examples)
        data = {}
        index = 0

        for (i, (split_name, ratio)) in enumerate(self.split_ratios):
            length = math.floor(L * ratio)
            if i != len(self.split_ratios) - 1:
                end_index = min(index + length, L)
            else:
                end_index = L
            data[split_name] = examples[index:end_index]
            index = end_index

        return data[split]

    def get_data(self, split: str = "train", seed=0, **kwargs) -> List[Example]:
        seed, new_seed = utils.split_seed(seed)
        examples = self.generate_expressions(seed=new_seed, **kwargs)
        examples = self.get_split(examples, split=split)
        if split != "train":
            kwargs["min_digits"] = kwargs.get("max_digits", 8)
            kwargs["max_digits"] = kwargs["min_digits"] + 2
            seed, new_seed = utils.split_seed(seed)
            hard_examples = self.generate_expressions(seed=new_seed, **kwargs)
            if split == "dev":
                pass
                logging.info("Dev set is easy")
                # examples += hard_examples[: len(hard_examples) // 2]
            elif split == "test":
                examples += hard_examples[len(hard_examples) // 2 :]
                # pdb.set_trace()
                # pass
            else:
                logging.warn("unknown split")
        seed, new_seed = utils.split_seed(seed)
        rng = np.random.default_rng(new_seed)
        rng.shuffle(examples)
        return examples

    def get_collate(tokenizer) -> Callable:
        def collate(data) -> Mapping[str, torch.Tensor]:
            inputs = [d[0] for d in data]
            targets = [d[1] for d in data]

            tokenizer.padding_side = "left"

            inputs = tokenizer.batch_encode_plus(
                inputs, padding="longest", return_tensors="pt"
            )

            tokenizer.padding_side = "right"

            targets = tokenizer.batch_encode_plus(
                targets, padding="longest", return_tensors="pt"
            )

            input_ids = torch.cat([inputs.input_ids, targets.input_ids], dim=1)

            labels = torch.cat(
                [
                    torch.zeros_like(inputs.input_ids) + tokenizer.pad_token_id,
                    targets.input_ids,
                ],
                dim=1,
            )

            attention_mask = torch.cat(
                [inputs.attention_mask, targets.attention_mask], dim=1
            )

            labels = labels.masked_fill(
                labels == tokenizer.pad_token_id, -100
            )  # GPT-2 specific

            input_lengths = torch.tensor([inputs.input_ids.shape[1]])

            data = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "input_lengths": input_lengths,
            }

            return data

        return collate

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        input = f"Q: How many 1s in {ex.x1}?\n"
        output = f"A: {ex.y}."
        return input, output


if __name__ == "__main__":
    datasets = [ParityDataset(split=s) for s in ("train", "dev", "test")]
    # pdb.set_trace()
    # assert len(set(datasets[0].data).intersection(datasets[2].data)) == 0

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    collate_fn = ParityDataset.get_collate(tokenizer)
    train_loader = DataLoader(
        datasets[0], batch_size=2, shuffle=True, collate_fn=collate_fn
    )

    logging.info(f"{[len(d) for d in datasets]}")

    pdb.set_trace()
    for data in train_loader:
        pdb.set_trace()
