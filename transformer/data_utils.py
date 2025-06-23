import os
import json
import argparse
from typing import Literal, TYPE_CHECKING, TypedDict

import torch
from datasets import load_dataset

if TYPE_CHECKING:
    from tokenizers import Tokenizer


class DataSample(TypedDict):
    src: str
    tgt: str


def _load_data(split: Literal["raw", "simplified"]) -> list[DataSample]:
    dataset = load_dataset("seara/ru_go_emotions", split)
    data = []
    if split == "simplified":
        text = (
            dataset["train"]["text"]
            + dataset["validation"]["text"]
            + dataset["test"]["text"]
        )
        ru_text = (
            dataset["train"]["ru_text"]
            + dataset["validation"]["ru_text"]
            + dataset["test"]["ru_text"]
        )
    else:
        text = dataset["train"]["text"]
        ru_text = dataset["train"]["ru_text"]

    for text, ru_text in zip(text, ru_text):
        data.append({"src": text, "tgt": ru_text})
    return data


def download_data(save_path: str, data_size: Literal["small", "full"], train_fraction: float = 0.8) -> None:
    data = _load_data("raw" if data_size == "full" else "simplified")
    train_size = int(len(data) * train_fraction)
    train_data = data[:train_size]
    val_data = data[train_size:]

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "train.json"), "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_path, "val.json"), "w") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)


def get_data_batch_iterator(
    data: list[DataSample], tokenizer: "Tokenizer", batch_size: int = 16
):
    for i in range(0, len(data), batch_size):
        batch_data = data[i : i + batch_size]
        src_data = [item["src"] for item in batch_data]
        tgt_data = [item["tgt"] for item in batch_data]
        src_encoding = tokenizer.encode_batch(src_data)
        tgt_encoding = tokenizer.encode_batch(tgt_data)

        yield (
            torch.tensor([b.ids for b in src_encoding]),
            torch.tensor([b.ids for b in tgt_encoding]),
            torch.tensor([b.attention_mask for b in src_encoding]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare dataset for training or evaluation"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data",
        help="Path to save the dataset files",
    )
    parser.add_argument(
        "--data_size",
        type=str,
        choices=["small", "full"],
        default="full",
        help="Size of the dataset to download (default: 'full')",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8)",
    )
    args = parser.parse_args()
    download_data(args.save_path, args.data_size, args.train_fraction)
