import json
import logging
import math
import pdb
import pickle
from collections import namedtuple
from operator import add, mul, sub
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


Example = namedtuple("Example", "x1 x2 op y")


class ArithmethicDataset(Dataset):

    split_ratios: List = [("train", 0.8), ("dev", 0.1), ("test", 0.1)]

    def __init__(self, split_ratios=None, **kwargs):

        for (k, v) in kwargs.items():
            self.__setattr__(k, v)

        if split_ratios is not None:
            self.split_ratios = split_ratios

        self.data = self.generate_data(**kwargs)

    def random_with_n_digits(rng, n):
        range_start = 10 ** (n - 1)
        range_end = (10**n) - 1
        return rng.integers(range_start, range_end)

    def save_to_file(self, path):
        str_data = [d for d in self]
        with open(path, "wb") as handle:
            if path.endswith("json"):
                json.dump(handle, str_data)
            elif path.endswith("pickle"):
                pickle.dump(str_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                raise ValueError("Unknown data file format")

    def generate_data(
        self,
        max_digits: int = 5,
        operations: List["str"] = ["add"],
        negatives: bool = False,
        N_per_digit: int = 200,
        split: str = "train",
        seed: int = 0,
    ) -> List:

        rng = np.random.default_rng(seed)
        examples = set()
        for operation in operations:
            op_fn = eval(operation)
            for e1 in range(max_digits):
                for e2 in range(max_digits):
                    for _ in range(N_per_digit * e1 * e2):
                        x1 = ArithmethicDataset.random_with_n_digits(rng, e1 + 1)
                        x2 = ArithmethicDataset.random_with_n_digits(rng, e2 + 1)
                        if not negatives and operation == "sub":
                            if op_fn(x1, x2) < 0:
                                x2, x1 = x1, x2
                        examples.add((x1, x2, operation, op_fn(x1, x2)))

        examples = list(map(lambda x: Example(*x), examples))

        rng = np.random.default_rng(seed)
        rng.shuffle(examples)

        L = len(examples)
        data = {}
        index = 0

        for (i, (sname, ratio)) in enumerate(self.split_ratios):
            length = math.floor(L * ratio)
            if i != len(self.split_ratios) - 1:
                end_index = min(index + length, L)
            else:
                end_index = L
            data[sname] = examples[index:end_index]
            index = end_index

        return data[split]

    def get_collate(tokenizer):
        def collate(data):
            inputs = [d[0] for d in data]
            targets = [d[1] for d in data]

            inputs = tokenizer.batch_encode_plus(
                inputs, padding="longest", return_tensors="pt"
            )

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

    def __len__(self):
        return len(self.data)

    def op_to_str(self, op):
        if op == "add":
            return "plus"
        elif op == "sub":
            return "minus"
        elif op == "mul":
            return "times"

    def digit_to_str(self, digit: int):
        return " ".join(list(str(digit)))

    def stringify(self, ex: Example):
        op_str = self.op_to_str(ex.op)
        x1_str = self.digit_to_str(ex.x1)
        x2_str = self.digit_to_str(ex.x2)
        y_str = self.digit_to_str(ex.y)
        return Example(x1_str, x2_str, op_str, y_str)

    def __getitem__(self, idx):
        ex = self.stringify(self.data[idx])
        input = f"{ex.x1} {ex.op} {ex.x2} ="
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