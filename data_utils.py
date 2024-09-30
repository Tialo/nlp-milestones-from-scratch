import os

import torch
import requests


def download_data(rewrite=False, save_path="train_data.txt"):
    if os.path.isfile(save_path) and not rewrite:
        return print("data is already downloaded")

    base_url = "https://raw.githubusercontent.com/mashashma/WMT2022-data/main/en-ru/en-ru.1m_{}.tsv"

    # there are 10 files in this repo
    n_files = 10
    with open(save_path, "w", encoding="utf-8") as f:
        for file_idx in range(1, n_files + 1):
            r = requests.get(base_url.format(file_idx))
            f.write(r.text)
            print(f"{file_idx}/{n_files} saved")


def load_data(data_path="train_data.txt"):
    with open(data_path, encoding="utf-8") as f:
        data = f.read().splitlines()
    return [example.split("\t") for example in data if "\t" in example]


def get_data_batch_iterator(data, tokenizer, batch_size=16):
    # TODO: generate batches with strategy to minimize number of pad tokens (pack texts with similar length together)
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        src_data, tgt_data = zip(*batch_data)
        src_encoding = tokenizer.encode_batch(src_data)
        tgt_encoding = tokenizer.encode_batch(tgt_data)

        yield (
            torch.tensor([b.ids for b in src_encoding]),
            torch.tensor([b.ids for b in tgt_encoding]),
            torch.tensor([b.attention_mask for b in src_encoding]),
        )


if __name__ == '__main__':
    download_data()
