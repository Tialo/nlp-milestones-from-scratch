import os
import json

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from transformer import Transformer
from tokenizer_utils import get_tokenizer, decode
from data_utils import get_data_batch_iterator, load_data

tokenizer = get_tokenizer("tokenizer.json")
pad_index = tokenizer.token_to_id("[PAD]")
start_index = tokenizer.token_to_id("[START]")
end_index = tokenizer.token_to_id("[END]")
data = load_data("data.txt")

TRAIN_SIZE = int(2e4)
VAL_SIZE = int(5e3)
train_data = data[:TRAIN_SIZE]
val_data = data[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(tokenizer.get_vocab_size()).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

def rate(step: int, d_model: int = 512, warmup: int = 4000):
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lr_lambda=rate,
)

epochs = 15
batch_size = 64
if os.path.isfile("best_val_loss.txt"):
    with open("best_val_loss.txt") as f:
        best_val_loss = float(f.read())
else:
    best_val_loss = float('inf')

train_loss_history = []
val_loss_history = []

for e in range(epochs):
    data_iterator = get_data_batch_iterator(train_data, tokenizer, batch_size=batch_size)
    model.train()

    epoch_train_loss_history = []
    for src_tokens, tgt_tokens, src_mask in tqdm(data_iterator, total=len(train_data) // batch_size, desc=f"Train epoch {e}"):
        # TODO: add label smoothing
        src_tokens = src_tokens.to(device)
        tgt_inputs = tgt_tokens[:, :-1].to(device)
        tgt_labels = tgt_tokens[:, 1:].to(device)
        src_mask = src_mask.to(device)

        logits = model(src_tokens, tgt_inputs, src_mask=src_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_labels.view(-1))
        epoch_train_loss_history.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
    train_loss_history.append(epoch_train_loss_history)

    data_iterator = get_data_batch_iterator(val_data, tokenizer, batch_size=2 * batch_size)
    model.eval()

    epoch_val_loss_history = []
    with torch.no_grad():
        for src_tokens, tgt_tokens, src_mask in tqdm(data_iterator, total=len(val_data) // (2 * batch_size), desc=f"Val epoch {e}"):
            src_tokens = src_tokens.to(device)
            tgt_inputs = tgt_tokens[:, :-1].to(device)
            tgt_labels = tgt_tokens[:, 1:].to(device)
            src_mask = src_mask.to(device)

            logits = model(src_tokens, tgt_inputs, src_mask=src_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_labels.view(-1))
            epoch_val_loss_history.append(loss.item())

        src_tokens, tgt_tokens, src_mask = next(get_data_batch_iterator(val_data, tokenizer, batch_size=1))
        generated = model.generate(
            src_tokens.to(device),
            start_index,
            end_index,
            # We set the maximum output length during inference to input length + 50, but terminate early when possible
            max_tokens=len(src_tokens) + 50,
        )
        print("Source:", decode(tokenizer, src_tokens))
        print("Target:", decode(tokenizer, tgt_tokens))
        print("Generated:", decode(tokenizer, generated))
        print()

    val_loss = torch.tensor(epoch_val_loss_history).mean().item()
    print(f"{val_loss=}, {best_val_loss=}", flush=True)
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "best_model.pth")
        with open("best_val_loss.txt", "w") as f:
            f.write(str(val_loss))
        best_val_loss = val_loss
    val_loss_history.append(epoch_val_loss_history)


with open("train_loss.json", "w") as f:
    json.dump(train_loss_history, f)

with open("val_loss.json", "w") as f:
    json.dump(val_loss_history, f)
