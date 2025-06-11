import os
import json
import argparse

import torch
import evaluate

from transformer import Transformer
from generator import Generator
from tokenizer_utils import get_tokenizer, decode


def evaluate_model(model_path, tokenizer_path, val_path, bleu_samples=1000, verbose: bool = True):
    with open(val_path, encoding="utf-8") as f:
        val_data = json.load(f)

    tokenizer = get_tokenizer(tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer.from_pretrained(model_path).to(device)
    model.eval()

    generator = Generator(
        model,
        tokenizer.token_to_id("[START]"),
        tokenizer.token_to_id("[END]"),
    )
    bleu = evaluate.load("bleu")

    bleu_values = []
    num_samples = min(bleu_samples, len(val_data))

    print(f"Evaluating model on {num_samples} samples...")
    for i, sample in enumerate(val_data[:num_samples]):
        src_language, tgt_language = sample
        input_tokens = tokenizer.encode(src_language).ids
        output_tokens = generator.generate(torch.tensor(input_tokens).to(device))
        decoded = decode(tokenizer, output_tokens)
        try:
            bleu_value = bleu.compute(
                references=[tgt_language], predictions=[decoded], smooth=True
            )["bleu"]
        except ZeroDivisionError:
            bleu_value = 0.0
        bleu_values.append(bleu_value)
        percent = 100 * (i + 1) / num_samples
        prev_percent = 100 * i / num_samples
        if verbose and (percent // 10 > prev_percent // 10):
            print(f"Evaluated on {i + 1} samples")

    return 100 * sum(bleu_values) / len(bleu_values)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Transformer model using BLEU score"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the saved model directory",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, help="Path to the tokenizer file (.json)"
    )
    parser.add_argument(
        "--val_path", type=str, help="Path to the validation data file (.json)"
    )
    parser.add_argument(
        "--bleu_samples",
        type=int,
        default=1000,
        help="Number of samples to use for BLEU evaluation (default: 1000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose progress of evaluation (default: True)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise ValueError(f"Error: Model path does not exist: {args.model_path}")

    if not os.path.isfile(args.tokenizer_path):
        raise ValueError(f"Error: Tokenizer path does not exist: {args.tokenizer_path}")

    if not os.path.isfile(args.val_path):
        raise ValueError(f"Error: Validation data path does not exist: {args.val_path}")

    avg_bleu = evaluate_model(args.model_path, args.tokenizer_path, args.val_path, args.bleu_samples, args.verbose)
    print(f"\nAverage BLEU: {avg_bleu:.2f}%")


if __name__ == "__main__":
    main()
