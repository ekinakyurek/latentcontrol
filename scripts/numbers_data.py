import json
import math
import pdb
import pickle
from operator import add, mul, sub  # noqa: F401, E501
from typing import Callable, List, Mapping, NamedTuple, Union
import numpy as np
import torch
from absl import logging
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from src import utils


def _op_to_str(op: str) -> str:
    if op == "add":
        return "plus"
    elif op == "sub":
        return "minus"
    elif op == "mul":
        return "times"


def _digit_to_str(digit: int) -> str:
    return " ".join(list(str(digit)))


class Example(NamedTuple):
    x1: Union[int, str]
    x2: Union[int, str]
    op: Union[Callable, str]
    y: Union[int, str]

    def __eq__(self, other):
        if isinstance(other, Example):
            return str(self) == str(other)
        return False

    def __str__(self):
        op_str = _op_to_str(self.op)
        return f"{self.x1} {op_str} {self.x2} = {self.y}"


class ArithmethicDataset(Dataset):

    split_ratios: List = [("train", 0.8), ("dev", 0.1), ("test", 0.1)]

    def __init__(
        self,
        split_ratios: List = None,
        reversed_outputs: bool = False,
        **kwargs,
    ):

        for (k, v) in kwargs.items():
            self.__setattr__(k, v)

        if split_ratios is not None:
            self.split_ratios = split_ratios

        self.reversed_outputs = reversed_outputs

        self.data = self.get_data(**kwargs)

    def random_with_n_digits(rng, n: int) -> int:
        range_start = 10 ** (n - 1)
        range_end = (10**n) - 1
        return rng.integers(range_start, range_end)

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
        max_digits: int = 8,
        min_digits: int = 0,
        operations: List["str"] = ["add"],
        negatives: bool = False,
        N_per_digit: int = 90,
        seed=0,
    ) -> List[Example]:
        rng = np.random.default_rng(seed)
        examples = set()
        for operation in operations:
            op_fn = eval(operation)
            for e1 in range(min_digits, max_digits):
                for _ in range(N_per_digit * e1 * e1):
                    x1 = ArithmethicDataset.random_with_n_digits(rng, e1 + 1)
                    x2 = ArithmethicDataset.random_with_n_digits(rng, e1 + 1)
                    if not negatives and operation == "sub":
                        if op_fn(x1, x2) < 0:
                            x2, x1 = x1, x2
                    examples.add((x1, x2, operation, op_fn(x1, x2)))

        examples = list(map(lambda x: Example(*x), examples))
        seed, new_seed = utils.split_seed(seed)
        rng = np.random.default_rng(new_seed)
        rng.shuffle(examples)
        return examples

    def get_split(
        self,
        examples,
        split: str = "train",
    ) -> List[Example]:
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

    @staticmethod
    def get_collate(tokenizer: AutoTokenizer, interleave=True) -> Callable:
        def collate(data) -> Mapping[str, torch.Tensor]:
            if interleave:
                inputs = [d[0] for d in data]
                targets = [d[1] for d in data]

                tokenizer.padding_side = "left"

                inputs = tokenizer.batch_encode_plus(
                    inputs,
                    padding="longest",
                    return_tensors="pt",
                )

                tokenizer.padding_side = "right"

                targets = tokenizer.batch_encode_plus(
                    targets, padding="longest", return_tensors="pt"
                )

                input_ids = torch.cat(
                    [inputs.input_ids, targets.input_ids], dim=1
                )

                labels = torch.cat(
                    [
                        torch.zeros_like(inputs.input_ids)
                        + tokenizer.pad_token_id,
                        targets.input_ids,
                    ],
                    dim=1,
                )

                attention_mask = torch.cat(
                    [inputs.attention_mask, targets.attention_mask], dim=1
                )
            else:
                raise ValueError("Normal batching is not implemente yet!")

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

    def stringify(self, ex: Example) -> Example:
        op_str = _op_to_str(ex.op)
        x1_str = _digit_to_str(ex.x1)
        x2_str = _digit_to_str(ex.x2)
        y_str = _digit_to_str(ex.y)
        if self.reversed_outputs:
            y_str = y_str[::-1]
        return Example(x1_str, x2_str, op_str, y_str)

    def __getitem__(self, idx):
        ex = self.stringify(self.data[idx])
        input = f"Q: What is {ex.x1} {ex.op} {ex.x2} ?\n"
        output = f"{ex.y} ."
        return input, output


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    datasets = [ArithmethicDataset(split=s) for s in ("train", "dev", "test")]

    assert len(set(datasets[0].data).intersection(datasets[2].data)) == 0

    collate_fn = ArithmethicDataset.get_collate(tokenizer)
    train_loader = DataLoader(
        datasets[0], batch_size=2, shuffle=True, collate_fn=collate_fn
    )

    logging.info(f"{[len(d) for d in datasets]}")

    for data in train_loader:
        pdb.set_trace()
