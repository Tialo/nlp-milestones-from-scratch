import os
from itertools import islice

import torch
import torch.nn as nn
from clearml import Task
from tqdm.auto import tqdm

from transformer import Transformer
from tokenizer_utils import get_tokenizer, decode
from data_utils import get_data_batch_iterator, load_data

src_tokenizer = get_tokenizer("tokenizer_src.json")
tgt_tokenizer = get_tokenizer("tokenizer_tgt.json")

pad_index = tgt_tokenizer.token_to_id("[PAD]")
start_index = tgt_tokenizer.token_to_id("[START]")
end_index = tgt_tokenizer.token_to_id("[END]")

data = load_data("data.txt")

SAVE_BEST_MODEL = False
TRAIN_SIZE = 100_000
VAL_SIZE = 20_000
EPOCHS = 20
BASE_LR = 1.0
BATCH_SIZE = 24
WARMUP_STEPS = 3000
ACCUMULATION_STEPS = 10

assert TRAIN_SIZE + VAL_SIZE < len(data), f"Val dataset is truncated. Total samples {len(data)}, trying to use {TRAIN_SIZE + VAL_SIZE} samples"

task = Task.init(
    project_name="vanilla-transformer",
    task_name="transfromer-training",
)
task.connect({
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "warmup_steps": WARMUP_STEPS,
    "train_size": TRAIN_SIZE,
    "val_size": VAL_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS,
    "base_lr": BASE_LR,
})

train_data = data[:TRAIN_SIZE]
val_data = data[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    src_tokenizer.get_vocab_size(),
    tgt_tokenizer.get_vocab_size(),
).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

def rate(step: int, d_model: int = 512, warmup: int = WARMUP_STEPS):
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

opt = torch.optim.Adam(model.parameters(), lr=BASE_LR, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lr_lambda=rate,
)

best_val_loss = float('inf')

def train_one_epoch(model, data_iterator, criterion, opt, scheduler, device):
    model.train()
    epoch_loss_history = []
    accumulated_loss = 0

    for i, (src_tokens, tgt_tokens, src_mask) in enumerate(tqdm(
        data_iterator,
        total=len(train_data) // BATCH_SIZE,
        desc=f"Train epoch {e}",
    )):
        # TODO: add label smoothing
        src_tokens = src_tokens.to(device)
        tgt_inputs = tgt_tokens[:, :-1].to(device)
        tgt_labels = tgt_tokens[:, 1:].to(device)
        src_mask = src_mask.to(device)

        logits = model(src_tokens, tgt_inputs, src_mask=src_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_labels.view(-1))
        # Normalize loss to account for accumulation
        loss /= ACCUMULATION_STEPS
        epoch_loss_history.append(loss.item() * ACCUMULATION_STEPS)
        accumulated_loss += loss.item()

        loss.backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            global_step = e * (len(train_data) // BATCH_SIZE // ACCUMULATION_STEPS) + i // ACCUMULATION_STEPS
            task.logger.report_scalar("train_loss", "train_batch", accumulated_loss, global_step)
            current_lr = scheduler.get_last_lr()[0]
            task.logger.report_scalar("learning_rate", "lr", current_lr, global_step)

            opt.step()
            opt.zero_grad()
            scheduler.step()
            accumulated_loss = 0
    
    # Handle remaining gradients
    if (i + 1) % ACCUMULATION_STEPS != 0:
        opt.step()
        opt.zero_grad()
        scheduler.step()

    return sum(epoch_loss_history) / len(epoch_loss_history)

@torch.no_grad
def validate_one_epoch(model, data_iterator, criterion, device):
    model.eval()

    epoch_val_loss_history = []
    for src_tokens, tgt_tokens, src_mask in tqdm(
        data_iterator,
        total=len(val_data) // (2 * BATCH_SIZE),
        desc=f"Val epoch {e}",
    ):
        src_tokens = src_tokens.to(device)
        tgt_inputs = tgt_tokens[:, :-1].to(device)
        tgt_labels = tgt_tokens[:, 1:].to(device)
        src_mask = src_mask.to(device)

        logits = model(src_tokens, tgt_inputs, src_mask=src_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_labels.view(-1))
        epoch_val_loss_history.append(loss.item())

    val_batch_iterator = get_data_batch_iterator(
        val_data,
        src_tokenizer,
        tgt_tokenizer,
        batch_size=1,
    )
    train_batch_iterator = get_data_batch_iterator(
        train_data,
        src_tokenizer,
        tgt_tokenizer,
        batch_size=1,
    )
    for _ in range(2):
        src_tokens, tgt_tokens, _ = next(val_batch_iterator)
        generated = model.generate(
            src_tokens.to(device),
            start_index,
            end_index,
            # "We set the maximum output length during inference to input length + 50, but terminate early when possible"
            max_tokens=len(src_tokens) + 50,
        )

        task.logger.report_text(
            "Validation example\n"
            f"Source: {decode(src_tokenizer, src_tokens)}\n"
            f"Target: {decode(tgt_tokenizer, tgt_tokens)}\n"
            f"Generated: {decode(tgt_tokenizer, generated)}\n",
            print_console=False,
        )

        src_tokens, tgt_tokens, _ = next(train_batch_iterator)
        generated = model.generate(
            src_tokens.to(device),
            start_index,
            end_index,
            max_tokens=len(src_tokens) + 50,
        )
        task.logger.report_text(
            "Train example\n"
            f"Source: {decode(src_tokenizer, src_tokens)}\n"
            f"Target: {decode(tgt_tokenizer, tgt_tokens)}\n"
            f"Generated: {decode(tgt_tokenizer, generated)}\n",
            print_console=False,
        )

    return sum(epoch_val_loss_history) / len(epoch_val_loss_history)


for e in range(EPOCHS):
    train_iterator = get_data_batch_iterator(
        train_data,
        src_tokenizer,
        tgt_tokenizer,
        batch_size=BATCH_SIZE,
    )
    epoch_train_loss_avg = train_one_epoch(model, train_iterator, criterion, opt, scheduler, device)
    task.logger.report_scalar("train_loss", "train_epoch", epoch_train_loss_avg, e)

    val_iterator = get_data_batch_iterator(
        val_data,
        src_tokenizer,
        tgt_tokenizer,
        batch_size=2 * BATCH_SIZE,
    )

    epoch_val_loss_avg = validate_one_epoch(model, val_iterator, criterion, device)
    task.logger.report_scalar("val_loss", "val_epoch", epoch_val_loss_avg, e)

    if SAVE_BEST_MODEL and epoch_val_loss_avg < best_val_loss:
        best_val_loss = epoch_val_loss_avg
        torch.save(model.state_dict(), "best_model.pth")
        task.upload_artifact("best_model", artifact_object="best_model.pth")
    
    torch.cuda.empty_cache()
