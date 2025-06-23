from collections.abc import Iterator

import torch

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKD, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder


class Tokenizer:
    def __init__(self, hf_tokenizer: HFTokenizer):
        self.tokenizer = hf_tokenizer
        self.vocab_size = hf_tokenizer.get_vocab_size()

    @classmethod
    def from_data(cls, train_data: Iterator[str], vocab_size: int, length: int | None = None):
        tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = Sequence([NFKD(), StripAccents()])
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            # [SEP] used for Task-specific input transformations (3.3 in original paper)
            special_tokens=["[UNK]", "[PAD]", "[START]", "[END]", "[SEP]", "[EXT]"],
            end_of_word_suffix="</w>",
        )
        tokenizer.train_from_iterator(train_data, trainer, length=length)
        tokenizer.post_processor = TemplateProcessing(
            # don't add [END] here, so we could encode
            # text and generate completions
            single="[START] $A",
            pair="[START] $A [SEP] $B [EXT]",
            special_tokens=[
                ("[START]", tokenizer.token_to_id("[START]")),
                ("[EXT]", tokenizer.token_to_id("[EXT]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"))
        tokenizer.decoder = BPEDecoder()
        return cls(tokenizer)

    def encode(self, text: list[str] | str, add_end_token: bool = True, add_ext_token: bool = False) -> torch.Tensor:
        assert not (add_end_token and add_ext_token)
        add_token = " [END]" if add_end_token else ""
        add_token = " [EXT]" if add_ext_token else add_token

        if isinstance(text, str):
            text += add_token
            return torch.tensor(self.tokenizer.encode(text).ids)

        text = [sample + add_token for sample in text]
        return torch.tensor([t.ids for t in self.tokenizer.encode_batch(text)])

    def encode_pair(self, pair: list[tuple[str, str]] | tuple[str, str]) -> torch.Tensor:
        if isinstance(pair[0], str):
            return torch.tensor(self.tokenizer.encode(pair[0], pair[1]).ids)
        return torch.tensor([t.ids for t in self.tokenizer.encode_batch(pair)])

    def decode(self, sequence: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(sequence, torch.Tensor):
            if sequence.ndim == 2:
                if sequence.shape[0] != 1:
                    raise ValueError(
                        "Can't handle 2D tensor in decode with 1st dimension > 1"
                    )
                sequence = sequence[0]
            sequence = sequence.tolist()
        return self.tokenizer.decode(sequence, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def change_max_len(self, max_len: int) -> None:
        self.tokenizer.enable_truncation(max_len)

    def save_pretrained(self, save_path: str) -> None:
        self.tokenizer.save(save_path)

    @classmethod
    def from_pretrained(cls, pretrained_path):
        return cls(HFTokenizer.from_file(pretrained_path))
