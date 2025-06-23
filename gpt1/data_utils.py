import os
import re
import gc
import argparse
import json
import multiprocessing

import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, load_dataset_builder

from tokenizer_utils import Tokenizer


def clean_text(text: str) -> str | None:
    """
    Remove all non-ascii characters.
    Remove all samples where 20%+ of characters were non-ascii.
    """
    ascii_only = ''.join(char for char in text if ord(char) < 128)
    single_spaced = re.sub(r'\s+', ' ', ascii_only)
    cleaned_text = single_spaced.strip()
    if len(cleaned_text) / len(text) > 0.8:
        return cleaned_text
    return None


def get_dataset_n_samples() -> int:
    builder = load_dataset_builder("HuggingFaceFW/fineweb-edu", name="sample-10BT")
    return builder.info.splits["train"].num_examples


def download_dataset_in_shards(data_samples: int | None = None, remove_non_ascii: bool = True, shard_size: int = int(5e5)):
    os.makedirs("raw_shards", exist_ok=True)
    data_samples = data_samples or get_dataset_n_samples()
    start_index = 0
    shard_index = 0
    clean_func = clean_text if remove_non_ascii else lambda x: x
    while start_index < data_samples:
        if not os.path.isfile(f"raw_shards/{shard_index}.json"):
            end_index = min(start_index + shard_size, data_samples)
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=f"train[{start_index}:{end_index}]")
            data = []
            for sample in tqdm(dataset, desc=f"Saving shard {shard_index}"):
                text = clean_func(sample["text"])
                if text:
                    data.append(text)
            with open(f"raw_shards/{shard_index}.json", "w") as f:
                json.dump(data, f)
        shard_index += 1
        start_index += shard_size
        gc.collect()


def iterate_raw_shards(raw_shards_list: list[str], yield_is_last: bool = False):
    for raw_shard_path in raw_shards_list:
        with open("raw_shards/" + raw_shard_path) as f:
            raw_shard = json.load(f)
        if yield_is_last:
            shard_len = len(raw_shard)
            for i, shard_sample in enumerate(raw_shard):
                yield shard_sample, i + 1 == shard_len
        else:
            yield from raw_shard


class ShardsBuilder:
    def __init__(self, tokenizer: Tokenizer, shard_size: int = int(1e8)):
        self.tokenizer = tokenizer
        self.shard_size = shard_size

    def _encode(self, iter_res):
        text, is_last = iter_res
        return self.tokenizer.encode(text), is_last

    def _build_shards(self, shard_name: str, raw_shards_list: list[str]):
        vocab_size = self.tokenizer.vocab_size
        if vocab_size < 65536:
            # enough to save token ids from 0 up to 65535
            # for shard_size = 1e8 each shard ~190MB
            np_dtype = np.uint16
        else:
            # from 0 up to ~4e9
            # ~380MB
            np_dtype = np.uint32

        curr_shard = np.full(self.shard_size, fill_value=vocab_size, dtype=np_dtype)
        buffer_offset = 0
        shards_saved = 0
        n_raw_shards = len(raw_shards_list)
        data_iterator = iterate_raw_shards(raw_shards_list, yield_is_last=True)
        print(f"Processing {n_raw_shards} {shard_name} raw shards")
        nproc = max(1, os.cpu_count() // 2)

        with multiprocessing.Pool(nproc) as pool:
            for tokens, is_last in tqdm(pool.imap(self._encode, data_iterator, chunksize=64), desc="Processing document"):
                if len(tokens) > self.shard_size - buffer_offset or is_last:
                    truncated_size = min(len(tokens), self.shard_size - buffer_offset)
                    leftover_size = max(0, len(tokens) - truncated_size)
                    curr_shard[buffer_offset:buffer_offset + truncated_size] = tokens[:truncated_size]
                    curr_shard = curr_shard[:buffer_offset + truncated_size]  # truncate shard if it is last

                    assert sum(curr_shard == vocab_size) == 0
                    np.save(f"shards/{shard_name}_{shards_saved}.npy", curr_shard)
                    shards_saved += 1
                    curr_shard = np.full(self.shard_size, fill_value=vocab_size, dtype=np_dtype)
                    curr_shard[:leftover_size] = tokens[truncated_size: truncated_size + leftover_size]
                    buffer_offset = leftover_size
                else:
                    curr_shard[buffer_offset: buffer_offset + len(tokens)] = tokens
                    buffer_offset += len(tokens)

    def build_shards(self, train_raw_shards: list[str], val_raw_shards: list[str]):
        os.makedirs("shards", exist_ok=True)
        self._build_shards(shard_name="train", raw_shards_list=train_raw_shards)
        self._build_shards(shard_name="val", raw_shards_list=val_raw_shards)


class DataLoader:
    def __init__(self, shards_name: str, batch_size: int, max_len: int):
        assert shards_name in ("train", "val")
        shards = [shard_name for shard_name in os.listdir("shards") if shards_name in shard_name]
        self.shards = sorted(shards, key=self._get_shard_number)
        assert self.shards
        self.shard_idx = self.current_offset = self.n_batches = 0
        self.batch_size = batch_size
        self.max_len = max_len
        for shard_idx in range(len(self.shards)):
            shard = self._get_shard(shard_idx)
            self.n_batches += len(shard) // (self.batch_size * self.max_len + 1)
        self.shard = self._get_shard(self.shard_idx)
    
    @staticmethod
    def _get_shard_number(shard_name: str) -> int:
        return int(''.join(x for x in shard_name if x.isdigit()))

    def _iter_shard_idx(self):
        # cyclic iteration over [0, 1, ..., len(self.shards) - 1]
        self.shard_idx = (self.shard_idx + 1) % len(self.shards)

    def _get_shard(self, shard_idx: int):
        shard = np.load(f"shards/{self.shards[shard_idx]}").astype(np.int32)
        return torch.tensor(shard, dtype=torch.long)

    def reset(self):
        if self.shard_idx != 0:
            self.shard = self._get_shard(0)
        self.shard_idx = 0
        self.current_offset = 0

    def get_batch(self):
        b, l = self.batch_size, self.max_len
        # drop all tokens from shard that don't fit into the batch
        if self.current_offset + b * l + 1 > len(self.shard):
            self._iter_shard_idx()
            self.shard = self._get_shard(self.shard_idx)
            self.current_offset = 0
        batch = self.shard[self.current_offset: self.current_offset + b * l + 1]
        self.current_offset += b * l + 1
        inputs = batch[:-1].view(b, l)
        outputs = batch[1:].view(b, l)
        return inputs, outputs


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare dataset for training or evaluation"
    )
    parser.add_argument(
        "--data_samples",
        type=int,
        default=None,
        help="Number of samples to download from the dataset (default: all)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32768,
        help="Size of BPE tokenizer vocabulary (default: 32768)",
    )
    parser.add_argument(
        "--n_val_shards",
        type=int,
        default=1,
        help="How many shards to use for validation (default: 1)",
    )
    parser.add_argument(
        "--n_tokenizer_shards",
        type=int,
        default=4,
        help="How many train shards to use for tokenizer train (default: 4)",
    )
    parser.add_argument(
        "--use_non_ascii",
        action="store_true",
        help="Use non ascii characters in dataset (removed by default)",
    )
    args = parser.parse_args()

    download_dataset_in_shards(
        data_samples=args.data_samples,
        remove_non_ascii=not args.use_non_ascii,
    )
    gc.collect()

    raw_shards = sorted(
        os.listdir("raw_shards"),
        key=lambda x: int(x.replace(".json", ""))
    )
    val_raw_shards = raw_shards[:args.n_val_shards]
    train_raw_shards = raw_shards[args.n_val_shards:]
    tokenizer_raw_shards = train_raw_shards[:args.n_tokenizer_shards]

    tokenizer_path = "tokenizer.json"
    if os.path.isfile(tokenizer_path):
        tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_data(iterate_raw_shards(tokenizer_raw_shards), vocab_size=args.vocab_size)
        tokenizer.save_pretrained(tokenizer_path)
    gc.collect()

    shard_builder = ShardsBuilder(tokenizer)
    shard_builder.build_shards(train_raw_shards, val_raw_shards)


if __name__ == "__main__":
    main()
