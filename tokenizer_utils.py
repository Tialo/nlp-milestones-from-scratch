import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKD, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder

def _build_tokenizer(data: list[str], save_path: str):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Sequence([NFKD(), StripAccents()])
    trainer = BpeTrainer(
        vocab_size=4_096,
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[START]", "[END]"],
        end_of_word_suffix="</w>",
    )
    tokenizer.train_from_iterator(data, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[START] $A [END]",
        special_tokens=[
            ("[START]", tokenizer.token_to_id("[START]")),
            ("[END]", tokenizer.token_to_id("[END]")),
        ],
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"))
    tokenizer.enable_truncation(72)
    tokenizer.decoder = BPEDecoder()
    tokenizer.save(save_path)
    return tokenizer


def build_tokenizers(splitted_data):
    language_src, language_tgt = zip(*splitted_data)
    src_tokenizer = _build_tokenizer(language_src, "tokenizer_src.json")
    tgt_tokenizer = _build_tokenizer(language_tgt, "tokenizer_tgt.json")
    return src_tokenizer, tgt_tokenizer


def get_tokenizer(tokenizer_path: str):
    return Tokenizer.from_file(tokenizer_path)


def decode(tokenizer, sequence) -> str:
    if isinstance(sequence, torch.Tensor):
        if sequence.ndim == 2:
            if sequence.shape[0] != 1:
                raise ValueError("Can't handle 2D tensor in decode with 1st dimension > 1")
            sequence = sequence[0]
        sequence = sequence.tolist()
    return tokenizer.decode(sequence, skip_special_tokens=True)
