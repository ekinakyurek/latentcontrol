import json
import os
import pickle
from typing import Callable, List, Mapping, NamedTuple, Union

from absl import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from datasets import load_dataset


class Example(NamedTuple):
    x1: Union[int, str]
    x2: Union[int, str]
    op: Union[Callable, str]
    y: Union[int, str]


class NumGLUEDataset(Dataset):

    split_ratios: List = [("train", 0.8), ("dev", 0.1), ("test", 0.1)]
    label_map: Mapping[int, str] = {
        0: "entailment",
        1: "neutral",
        2: "contradiction",
    }

    def __init__(self, split_ratios: List = None, seed: int = 0, **kwargs):
        self.seed = seed
        for (k, v) in kwargs.items():
            self.__setattr__(k, v)

        split = kwargs.get("split")
        if split == "dev":
            length = 5000
        elif split == "test":
            length = 5000
        elif split == "train":
            length = 500
        data = self.load_dataset(split=split)
        length = min(len(data), length)
        data = data.shuffle(self.seed).select(range(length))
        self.data = data

    def load_dataset(self, split="train", root="datasets/numglue/data"):
        data_files = {f"{split}": f"NumGLUE_{split}.str.json"}
        # features = Features(
        #     {
        #         "question": Value("string"),
        #         "answer": Value("string"),
        #         "type": Value("string"),
        #     }
        # )

        if not os.path.isfile(os.path.join(root, data_files[split])):
            with open(
                os.path.join(root, data_files[split].replace(".str", "")), "r"
            ) as f:
                data = []
                for line in f:
                    d = json.loads(line)
                    if d["type"] == "Type_7":
                        for k, v in d.items():
                            d[k] = str(v)
                        data.append(d)

            with open(os.path.join(root, data_files[split]), "w") as f:
                lines = "\n".join([json.dumps(d) for d in data])
                print(lines, file=f)

        logging.info(f"loading {split} data")
        data = load_dataset(
            root,
            data_files=data_files,
            split=split,
        )
        return data

    def save_to_file(self, path: str):
        str_data = [d for d in self]
        with open(path, "wb") as handle:
            if path.endswith("json"):
                json.dump(handle, str_data)
            elif path.endswith("pickle"):
                pickle.dump(str_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                raise ValueError("Unknown data file format")

    @staticmethod
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
        premise = ex["statement1"]
        hypothesis = ex["statement2"]
        input = f"Premise: {premise}\nHypothesis:\n{hypothesis}\nRelation:"
        label = ex["answer"].lower()
        output = f"{label} ."
        return input, output


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    datasets = [NumGLUEDataset(split=s) for s in ("train", "dev", "test")]

    # assert len(set(datasets[0].data).intersection(datasets[2].data)) == 0

    collate_fn = NumGLUEDataset.get_collate(tokenizer)
    train_loader = DataLoader(
        datasets[0], batch_size=2, shuffle=True, collate_fn=collate_fn
    )

    logging.info(f"{[len(d) for d in datasets]}")

    for data in train_loader:
        break
