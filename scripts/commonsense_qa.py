import json
import pdb
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


class CommonSenseQADataset(Dataset):

    split_ratios: List = [("train", 0.8), ("dev", 0.1), ("test", 0.1)]

    def __init__(self, split_ratios: List = None, **kwargs):

        for (k, v) in kwargs.items():
            self.__setattr__(k, v)

        split = kwargs.get("split")
        if split == "dev" or split == "test":
            split = "validation"
        self.data = load_dataset("commonsense_qa", split=split)

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
        question = ex["question"]
        choices = ex["choices"]
        choicez = zip(choices["label"], choices["text"])
        choice_text = [
            f"({label.lower()}) {text.lower()}" for label, text in choicez
        ]
        choice_text = "\n".join(choice_text)
        input = f"Q: {question}\nAnswer Choices:\n{choice_text}"
        answer_index = choices["label"].index(ex["answerKey"])
        answer_text = choices["text"][answer_index]
        answer_label = f"({ex['answerKey'].lower()})"
        output = f"{answer_text} {answer_label} ."
        return input, output


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    datasets = [CommonSenseQADataset(split=s) for s in ("train", "dev", "test")]

    # assert len(set(datasets[0].data).intersection(datasets[2].data)) == 0

    collate_fn = CommonSenseQADataset.get_collate(tokenizer)
    train_loader = DataLoader(
        datasets[0], batch_size=2, shuffle=True, collate_fn=collate_fn
    )

    logging.info(f"{[len(d) for d in datasets]}")

    for data in train_loader:
        pdb.set_trace()
