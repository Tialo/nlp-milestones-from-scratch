from typing import Literal

import torch
from datasets import load_dataset


def load_data(split: Literal["simplified", "raw"] = "simplified"):
    dataset = load_dataset('seara/ru_go_emotions', split)
    data = []
    if split == "simplified":
        text = dataset['train']['text'] + dataset['validation']['text'] + dataset['test']['text']
        ru_text = dataset['train']['ru_text'] + dataset['validation']['ru_text'] + dataset['test']['ru_text']
    else:
        text = dataset['train']['text']
        ru_text = dataset['train']['ru_text']

    for text, ru_text in zip(text, ru_text):
        data.append((text, ru_text))
    return data


def get_data_batch_iterator(data, src_tokenizer, tgt_tokenizer, batch_size: int = 16):
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        src_data, tgt_data = zip(*batch_data)
        src_encoding = src_tokenizer.encode_batch(src_data)
        tgt_encoding = tgt_tokenizer.encode_batch(tgt_data)

        yield (
            torch.tensor([b.ids for b in src_encoding]),
            torch.tensor([b.ids for b in tgt_encoding]),
            torch.tensor([b.attention_mask for b in src_encoding]),
        )
