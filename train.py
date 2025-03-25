import os
import json

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

TRAIN_SIZE = 50_000
VAL_SIZE = 5_000
EPOCHS = 30
BATCH_SIZE = 64
WARMUP_STEPS = 3000

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

opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lr_lambda=rate,
)

if os.path.isfile("best_val_loss.txt"):
    with open("best_val_loss.txt") as f:
        best_val_loss = float(f.read())
else:
    best_val_loss = float('inf')

train_loss_history = []
val_loss_history = []

for e in range(EPOCHS):
    data_iterator = get_data_batch_iterator(
        train_data,
        src_tokenizer,
        tgt_tokenizer,
        batch_size=BATCH_SIZE,
    )
    model.train()

    epoch_train_loss_history = []
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
        epoch_train_loss_history.append(loss.item())

        global_step = e * (len(train_data) // BATCH_SIZE) + i
        task.logger.report_scalar("loss", "train_batch", loss.item(), global_step)
        current_lr = scheduler.get_last_lr()[0]
        task.logger.report_scalar("learning_rate", "lr", current_lr, global_step)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    epoch_train_loss_avg = sum(epoch_train_loss_history) / len(epoch_train_loss_history)
    task.logger.report_scalar("loss", "train_epoch", epoch_train_loss_avg, e)
    train_loss_history.append(epoch_train_loss_history)

    data_iterator = get_data_batch_iterator(
        val_data,
        src_tokenizer,
        tgt_tokenizer,
        batch_size=2 * BATCH_SIZE,
    )
    model.eval()

    epoch_val_loss_history = []
    with torch.no_grad():
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

        src_tokens, tgt_tokens, src_mask = next(get_data_batch_iterator(
            val_data,
            src_tokenizer,
            tgt_tokenizer,
            batch_size=1,
        ))
        generated = model.generate(
            src_tokens.to(device),
            start_index,
            end_index,
            # "We set the maximum output length during inference to input length + 50, but terminate early when possible"
            max_tokens=len(src_tokens) + 50,
        )
        source_text = decode(src_tokenizer, src_tokens)
        target_text = decode(tgt_tokenizer, tgt_tokens)
        generated_text = decode(tgt_tokenizer, generated)
        
        print("Source:", source_text)
        print("Target:", target_text)
        print("Generated:", generated_text)
        print()

        task.logger.report_text(f"Source: {source_text}\nTarget: {target_text}\nGenerated: {generated_text}", "translation_examples")


    val_loss = torch.tensor(epoch_val_loss_history).mean().item()
    task.logger.report_scalar("loss", "validation", val_loss, e)

    print(f"{val_loss=}, {best_val_loss=}", flush=True)
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "best_model.pth")
        with open("best_val_loss.txt", "w") as f:
            f.write(str(val_loss))
        best_val_loss = val_loss
        task.upload_artifact("best_model", artifact_object="best_model.pth")

    val_loss_history.append(epoch_val_loss_history)


with open("train_loss.json", "w") as f:
    json.dump(train_loss_history, f)

with open("val_loss.json", "w") as f:
    json.dump(val_loss_history, f)
