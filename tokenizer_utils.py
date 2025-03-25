import torch

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from data_utils import download_data


def _build_tokenizer(data: list[str], save_path: str):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[START]", "[END]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(data, trainer)
    tokenizer.save(save_path)


def build_tokenizers(data_path: str):
    download_data(save_path=data_path)
    with open(data_path) as f:
        data = f.read().splitlines()
    
    splitted_data = (row.split("\t") for row in data)
    # Some rows have 1 or 0 sequences
    language_src, language_tgt = zip(*[row for row in splitted_data if len(row) == 2])
    _build_tokenizer(language_src, "tokenizer_src.json")
    _build_tokenizer(language_tgt, "tokenizer_tgt.json")


def get_tokenizer(tokenizer_path: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.post_processor = TemplateProcessing(
        single="[START] $A [END]",
        special_tokens=[
            ("[START]", tokenizer.token_to_id("[START]")),
            ("[END]", tokenizer.token_to_id("[END]")),
        ],
    )
    tokenizer.enable_padding()
    return tokenizer


def decode(tokenizer, sequence):
    if isinstance(sequence, torch.Tensor):
        if sequence.ndim == 2:
            if sequence.shape[0] != 1:
                raise ValueError("Can't handle 2D tensor in decode with 1st dimension > 1")
            sequence = sequence[0]
        sequence = sequence.tolist()
    return tokenizer.decode(sequence)


if __name__ == '__main__':
    build_tokenizers("data.txt")
