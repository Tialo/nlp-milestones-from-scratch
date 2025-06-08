import os
import random
import json
import argparse
from typing import TYPE_CHECKING
from dataclasses import dataclass, asdict

import torch
import numpy as np

from generator import Generator
from transformer import Transformer
from loss import LabelSmoothingLoss
from tokenizer_utils import get_tokenizer, decode, build_tokenizer
from data_utils import get_data_batch_iterator, load_data


if TYPE_CHECKING:
    from transformer import TransformerConfig


@dataclass
class TrainConfig:
    tokenizer_path: str = "tokenizer.json"
    batch_size: int = 128
    epochs: int = 8
    base_lr: float = 0.8
    train_fraction: float = 0.8
    warmup_fraction: float = 0.3  # original paper used 4% of data for a warmup
    accumulation_steps: int = 10
    label_smoothing: float = 0.1
    use_cross_entropy: bool = True
    seed: int = 42


def set_seed(seed: int | None = 42):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_training(config: TrainConfig, transformer_config: "TransformerConfig"):
    set_seed(config.seed)
    data = load_data("raw")
    train_size = int(len(data) * config.train_fraction)
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_epoch_batches = (train_size + config.batch_size - 1) // config.batch_size
    train_epoch_steps = (train_epoch_batches + config.accumulation_steps - 1) // config.accumulation_steps
    train_steps = train_epoch_steps * config.epochs
    warmup_steps = int(train_steps * config.warmup_fraction)

    if not os.path.isfile(config.tokenizer_path):
        tokenizer = build_tokenizer(train_data, save_path=config.tokenizer_path)
    else:
        tokenizer = get_tokenizer(config.tokenizer_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(transformer_config).to(device)
    generator = Generator(
        model,
        tokenizer.token_to_id("[START]"),
        tokenizer.token_to_id("[END]"),
    )
    criterion = LabelSmoothingLoss(
        ignore_index=tokenizer.token_to_id("[PAD]"),
        smoothing=config.label_smoothing,
        use_cross_entropy=config.use_cross_entropy,
    )
    return {
        "data": data,
        "train_data": train_data,
        "val_data": val_data,
        "train_epoch_batches": train_epoch_batches,
        "warmup_steps": warmup_steps,
        "tokenizer": tokenizer,
        "device": device,
        "model": model,
        "generator": generator,
        "criterion": criterion,
    }


def rate(step: int, d_model: int = 512, warmup: int = 4000):
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


def train_one_epoch(model, data_iterator, criterion, opt, scheduler, device, train_epoch_batches, accumulation_steps, global_step, epoch_index, n_epochs):
    model.train()
    epoch_loss_history = []
    accumulated_loss = 0
    backwards_since_last_step = 0
    step_indices = []
    step_losses = []

    batch_digits = len(str(train_epoch_batches))
    print(f"Epoch: [{epoch_index+1}/{n_epochs}]")
    for batch_index, (src_tokens, tgt_tokens, src_mask) in enumerate(data_iterator):
        src_tokens = src_tokens.to(device)
        tgt_inputs = tgt_tokens[:, :-1].to(device)
        tgt_labels = tgt_tokens[:, 1:].to(device)
        src_mask = src_mask.to(device)

        logits = model(src_tokens, tgt_inputs, src_mask=src_mask)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            tgt_labels.view(-1),
        )
        epoch_loss_history.append(loss.item())
        accumulated_loss += loss.item()
        loss.backward()
        backwards_since_last_step += 1

        # Print every 5% of progress, but only batch/total, no percent
        prev_percent = 100 * batch_index / train_epoch_batches
        percent = 100 * (batch_index + 1) / train_epoch_batches
        prev_threshold = int(prev_percent // 5)
        current_threshold = int(percent // 5)
        if current_threshold > prev_threshold:
            current_lr = scheduler.get_last_lr()[0]
            current_loss = accumulated_loss / backwards_since_last_step
            print(
                f"Step: [{str(batch_index+1).rjust(batch_digits)}/{train_epoch_batches}]",
                f"Step loss: {current_loss:.4f}",
                f"LR: {current_lr:.6f}",
                sep=' | '
            )

        if (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == train_epoch_batches
        ):
            mean_loss = accumulated_loss / backwards_since_last_step
            opt.step()
            opt.zero_grad()
            scheduler.step()
            step_indices.append(global_step)
            step_losses.append(mean_loss)
            accumulated_loss = 0
            backwards_since_last_step = 0
            global_step += 1

    return sum(epoch_loss_history) / len(epoch_loss_history), global_step, step_indices, step_losses


@torch.no_grad
def validate_one_epoch(model, data_iterator, criterion, device):
    model.eval()
    epoch_val_loss_history = []
    for src_tokens, tgt_tokens, src_mask in data_iterator:
        src_tokens = src_tokens.to(device)
        tgt_inputs = tgt_tokens[:, :-1].to(device)
        tgt_labels = tgt_tokens[:, 1:].to(device)
        src_mask = src_mask.to(device)

        logits = model(src_tokens, tgt_inputs, src_mask=src_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_labels.view(-1))
        epoch_val_loss_history.append(loss.item())
    return sum(epoch_val_loss_history) / len(epoch_val_loss_history)


def cherry_pick_generation(val_data, tokenizer, generator, n_picks, device):
    val_batch_iterator = get_data_batch_iterator(
        val_data,
        tokenizer,
        batch_size=1,
    )
    print("Cherry picked generations:")
    for _ in range(n_picks):
        src_tokens, tgt_tokens, _ = next(val_batch_iterator)
        generated = generator.generate(
            src_tokens.to(device),
            # 6.1 We set the maximum output length during inference to input length + 50, but terminate early when possible
            max_tokens=len(src_tokens) + 50,
        )
        print(
            f"Source:    {decode(tokenizer, src_tokens)}",
            f"Target:    {decode(tokenizer, tgt_tokens)}",
            f"Generated: {decode(tokenizer, generated)}\n",
            sep='\n'
        )


def train_main(config: TrainConfig, transformer_config: "TransformerConfig", save_path: str):
    if os.path.isdir(save_path):
        raise RuntimeError(f"Directory {save_path} already exists, can't train model and save it there.")
    os.mkdir(save_path)
    prep = prepare_training(config, transformer_config)
    train_data = prep["train_data"]
    val_data = prep["val_data"]
    train_epoch_batches = prep["train_epoch_batches"]
    warmup_steps = prep["warmup_steps"]
    tokenizer = prep["tokenizer"]
    device = prep["device"]
    model = prep["model"]
    generator = prep["generator"]
    criterion = prep["criterion"]

    def lr_schedule(step):
        return rate(step, d_model=512, warmup=warmup_steps)

    opt = torch.optim.Adam(model.parameters(), lr=config.base_lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_schedule)

    global_step = 0
    epoch_indices = []
    epoch_train_losses = []
    epoch_val_losses = []
    step_indices = []
    step_train_losses = []

    for e in range(config.epochs):
        train_iterator = get_data_batch_iterator(
            train_data,
            tokenizer,
            batch_size=config.batch_size,
        )
        epoch_train_loss_avg, global_step, step_x, step_losses = train_one_epoch(
            model, train_iterator, criterion, opt, scheduler, device,
            train_epoch_batches, config.accumulation_steps, global_step, e, config.epochs
        )
        epoch_indices.append(e)
        epoch_train_losses.append(epoch_train_loss_avg)
        step_indices.extend(step_x)
        step_train_losses.extend(step_losses)

        cherry_pick_generation(
            val_data, tokenizer, generator, 4, device
        )
        print(f"\nTrain loss: {epoch_train_loss_avg:.4f}", end=' | ', flush=True)
        val_iterator = get_data_batch_iterator(
            val_data,
            tokenizer,
            batch_size=2 * config.batch_size,
        )
        epoch_val_loss_avg = validate_one_epoch(
            model, val_iterator, criterion, device,
        )
        epoch_val_losses.append(epoch_val_loss_avg)
        print(f"Valid loss: {epoch_val_loss_avg:.4f}\n")
        print()

        torch.cuda.empty_cache()

    model.save_pretrained(save_path)
    print(f"Model successfully saved at {save_path}!")

    return {
        "epoch_train_loss": (epoch_indices, epoch_train_losses),
        "epoch_val_loss": (epoch_indices, epoch_val_losses),
        "step_train_loss": (step_indices, step_train_losses),
    }


def create_train_config_from_args(args) -> TrainConfig:
    return TrainConfig(
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        base_lr=args.base_lr,
        train_fraction=args.train_fraction,
        warmup_fraction=args.warmup_fraction,
        accumulation_steps=args.accumulation_steps,
        label_smoothing=args.label_smoothing,
        use_cross_entropy=args.use_cross_entropy,
        seed=args.seed,
    )


def create_transformer_config_from_args(args) -> "TransformerConfig":
    from transformer import TransformerConfig
    
    return TransformerConfig(
        vocab_size=args.vocab_size,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_encoder_heads=args.n_encoder_heads,
        n_decoder_heads=args.n_decoder_heads,
        embed_size=args.embed_size,
        d_ff=args.d_ff,
        max_len=args.max_len,
        tie_embeddings=args.tie_embeddings,
        post_ln=args.post_ln,
        add_two_layer_norms=args.add_two_layer_norms,
        use_additional_dropout=args.use_additional_dropout,
        xavier_initialization=args.xavier_initialization,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer model')
    
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--tokenizer_path', type=str, default='tokenizer.json',
                           help='Path to tokenizer file (default: tokenizer.json)')
    train_group.add_argument('--batch_size', type=int, default=128,
                           help='Batch size for training (default: 128)')
    train_group.add_argument('--epochs', type=int, default=8,
                           help='Number of training epochs (default: 8)')
    train_group.add_argument('--base_lr', type=float, default=0.8,
                           help='Base learning rate (default: 0.8)')
    train_group.add_argument('--train_fraction', type=float, default=0.8,
                           help='Fraction of data to use for training (default: 0.8)')
    train_group.add_argument('--warmup_fraction', type=float, default=0.3,
                           help='Fraction of training steps for warmup (default: 0.3)')
    train_group.add_argument('--accumulation_steps', type=int, default=10,
                           help='Gradient accumulation steps (default: 10)')
    train_group.add_argument('--label_smoothing', type=float, default=0.1,
                           help='Label smoothing factor (default: 0.1)')
    train_group.add_argument('--use_cross_entropy', action='store_true', default=True,
                           help='Use cross entropy loss (default: True)')
    train_group.add_argument('--use_kl_divergence', dest='use_cross_entropy', action='store_false',
                           help='Use KL divergence loss instead of cross entropy')
    train_group.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    
    model_group = parser.add_argument_group('Transformer Model Configuration')
    model_group.add_argument('--vocab_size', type=int, default=8192,
                           help='Vocabulary size (default: 8192)')
    model_group.add_argument('--n_encoder_layers', type=int, default=6,
                           help='Number of encoder layers (default: 6)')
    model_group.add_argument('--n_decoder_layers', type=int, default=6,
                           help='Number of decoder layers (default: 6)')
    model_group.add_argument('--n_encoder_heads', type=int, default=8,
                           help='Number of encoder attention heads (default: 8)')
    model_group.add_argument('--n_decoder_heads', type=int, default=8,
                           help='Number of decoder attention heads (default: 8)')
    model_group.add_argument('--embed_size', type=int, default=512,
                           help='Embedding dimension (default: 512)')
    model_group.add_argument('--d_ff', type=int, default=2048,
                           help='Feed-forward network dimension (default: 2048)')
    model_group.add_argument('--max_len', type=int, default=4096,
                           help='Maximum sequence length (default: 4096)')
    model_group.add_argument('--tie_embeddings', action='store_true', default=True,
                           help='Tie input and output embeddings (default: True)')
    model_group.add_argument('--no_tie_embeddings', dest='tie_embeddings', action='store_false',
                           help='Do not tie input and output embeddings')
    model_group.add_argument('--post_ln', action='store_true', default=True,
                           help='Use post-layer normalization (default: True)')
    model_group.add_argument('--pre_ln', dest='post_ln', action='store_false',
                           help='Use pre-layer normalization instead of post-layer normalization')
    model_group.add_argument('--add_two_layer_norms', action='store_true', default=False,
                           help='Add additional layer normalization layers (default: False)')
    model_group.add_argument('--use_additional_dropout', action='store_true', default=False,
                           help='Use additional dropout layers (default: False)')
    model_group.add_argument('--xavier_initialization', action='store_true', default=False,
                           help='Use Xavier initialization (default: False)')

    parser.add_argument('--save_path', type=str, default='model',
                       help='Path to save the trained model (default: model)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    train_config = create_train_config_from_args(args)
    transformer_config = create_transformer_config_from_args(args)
    
    train_main(
        train_config,
        transformer_config,
        args.save_path,
    )
