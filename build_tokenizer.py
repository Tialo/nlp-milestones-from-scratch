from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from get_data import download_data


if __name__ == '__main__':
    data_path = "data.txt"
    download_data(save_path=data_path)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[START]", "[END]"])
    tokenizer.pre_tokenizer = Whitespace()

    files = [data_path]
    tokenizer.train(files, trainer)
    tokenizer.save("tokenizer.json")
