import os
import random
from dataclasses import dataclass

import torch
import numpy as np

from generator import Generator
from transformer import Transformer
from loss import LabelSmoothingLoss
from tokenizer_utils import get_tokenizer, decode, build_tokenizer
from data_utils import get_data_batch_iterator, load_data



@dataclass
class TrainConfig:
    # Misc
    model_save_path: str
    seed: int = 42
    tokenizer_path: str = "tokenizer.json"

    # Model architecture
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    n_encoder_heads: int = 8
    n_decoder_heads: int = 8
    embed_size: int = 512
    d_ff: int = 2048
    max_len: int = 4096
    tie_embeddings: bool = True
    post_ln: bool = True
    add_two_layer_norms: bool = False
    use_additional_dropout: bool = False
    xavier_initialization: bool = False

    # Training process
    batch_size: int = 128
    epochs: int = 8
    base_lr: float = 0.8
    train_fraction: float = 0.8
    warmup_fraction: float = 0.3  # original paper used 4% of data for a warmup
    accumulation_steps: int = 10
    label_smoothing: float = 0.1
    use_cross_entropy: bool = True


def set_seed(seed: int | None = 42):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_training(config: TrainConfig):
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
    model = Transformer(
        vocab_size=tokenizer.get_vocab_size(),
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        n_encoder_heads=config.n_encoder_heads,
        n_decoder_heads=config.n_decoder_heads,
        embed_size=config.embed_size,
        d_ff=config.d_ff,
        max_len=config.max_len,
        tie_embeddings=config.tie_embeddings,
        post_ln=config.post_ln,
        add_two_layer_norms=config.add_two_layer_norms,
        use_additional_dropout=config.use_additional_dropout,
        xavier_initialization=config.xavier_initialization,
    ).to(device)
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


def train_one_epoch(model, data_iterator, criterion, opt, scheduler, device, train_epoch_batches, accumulation_steps, global_step, epoch_index):
    model.train()
    epoch_loss_history = []
    accumulated_loss = 0
    backwards_since_last_step = 0
    step_indices = []
    step_losses = []

    for batch_index, (src_tokens, tgt_tokens, src_mask) in enumerate(data_iterator):
        src_tokens = src_tokens.to(device)  # (batch_size, seq_len_src)
        tgt_inputs = tgt_tokens[:, :-1].to(device)  # (batch_size, seq_len_tgt - 1)
        tgt_labels = tgt_tokens[:, 1:].to(device)  # (batch_size, seq_len_tgt - 1)
        src_mask = src_mask.to(device)

        logits = model(src_tokens, tgt_inputs, src_mask=src_mask)  # (batch_size, seq_len_tgt, vocab_size)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            tgt_labels.view(-1),
        )
        epoch_loss_history.append(loss.item())
        accumulated_loss += loss.item()
        loss.backward()
        backwards_since_last_step += 1

        if (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == train_epoch_batches
        ):
            mean_loss = accumulated_loss / backwards_since_last_step
            percent = (batch_index + 1) / train_epoch_batches * 100
            current_lr = scheduler.get_last_lr()[0]
            if percent % 5 < 100 / train_epoch_batches:
                print(f"Epoch {epoch_index+1} [{batch_index+1}/{train_epoch_batches}] ({percent:.1f}%): Step loss: {mean_loss:.4f}, LR: {current_lr:.6f}")
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
def validate_one_epoch(model, data_iterator, criterion, device, val_data, tokenizer, generator, batch_size, epoch_index):
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

    val_batch_iterator = get_data_batch_iterator(
        val_data,
        tokenizer,
        batch_size=1,
    )
    for _ in range(4):
        src_tokens, tgt_tokens, _ = next(val_batch_iterator)
        generated = generator.generate(
            src_tokens.to(device),
            # 6.1 We set the maximum output length during inference to input length + 50, but terminate early when possible
            max_tokens=len(src_tokens) + 50,
        )
        print("Validation example\n"
              f"Source: {decode(tokenizer, src_tokens)}\n"
              f"Target: {decode(tokenizer, tgt_tokens)}\n"
              f"Generated: {decode(tokenizer, generated)}\n\n")

    return sum(epoch_val_loss_history) / len(epoch_val_loss_history)


def train_main(config: TrainConfig):
    prep = prepare_training(config)
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
            train_epoch_batches, config.accumulation_steps, global_step, e
        )
        epoch_indices.append(e)
        epoch_train_losses.append(epoch_train_loss_avg)
        step_indices.extend(step_x)
        step_train_losses.extend(step_losses)
        print(f"Epoch {e+1}: Train loss: {epoch_train_loss_avg:.4f}")

        val_iterator = get_data_batch_iterator(
            val_data,
            tokenizer,
            batch_size=2 * config.batch_size,
        )
        epoch_val_loss_avg = validate_one_epoch(
            model, val_iterator, criterion, device, val_data, tokenizer, generator, config.batch_size, e
        )
        epoch_val_losses.append(epoch_val_loss_avg)
        print(f"Epoch {e+1}: Valid loss: {epoch_val_loss_avg:.4f}")

        torch.cuda.empty_cache()

    torch.save(model.state_dict(), config.model_save_path)

    return {
        "epoch_train_loss": (epoch_indices, epoch_train_losses),
        "epoch_val_loss": (epoch_indices, epoch_val_losses),
        "step_train_loss": (step_indices, step_train_losses),
    }


if __name__ == "__main__":
    train_main(TrainConfig(
        model_save_path="model.pth",
    ))
