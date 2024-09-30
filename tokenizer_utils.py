from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from data_utils import download_data


def build_tokenizer(data_path, save_path="tokenizer.json"):
    download_data(save_path=data_path)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[START]", "[END]"])
    tokenizer.pre_tokenizer = Whitespace()

    files = [data_path]
    tokenizer.train(files, trainer)
    tokenizer.save(save_path)


def get_tokenizer(tokenizer_path):
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


if __name__ == '__main__':
    build_tokenizer("data.txt")
