import json

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from transformer import Transformer
from tokenizer_utils import get_tokenizer
from data_utils import get_data_batch_iterator, load_data

tokenizer = get_tokenizer("tokenizer.json")
pad_index = tokenizer.token_to_id("[PAD]")
data = load_data("data.txt")

train_data = data[:10_000]
val_data = data[10_000:12_500]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(tokenizer.get_vocab_size()).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
# TODO: Implement learning rate change from paper
# Otherwise can't achieve sufficient quality
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 5
batch_size = 64
best_val_loss = 1e3

train_loss_history = []
val_loss_history = []

for e in range(epochs):
    data_iterator = get_data_batch_iterator(train_data, tokenizer, batch_size=batch_size)
    model.train()

    epoch_train_loss_history = []
    for src_tokens, tgt_tokens, src_mask in tqdm(data_iterator, total=len(train_data) // batch_size):
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
    train_loss_history.append(epoch_train_loss_history)

    data_iterator = get_data_batch_iterator(val_data, tokenizer, batch_size=2 * batch_size)
    model.eval()

    epoch_val_loss_history = []
    with torch.no_grad():
        for src_tokens, tgt_tokens, src_mask in tqdm(data_iterator, total=len(val_data) // (2 * batch_size)):
            src_tokens = src_tokens.to(device)
            tgt_inputs = tgt_tokens[:, :-1].to(device)
            tgt_labels = tgt_tokens[:, 1:].to(device)
            src_mask = src_mask.to(device)

            logits = model(src_tokens, tgt_inputs, src_mask=src_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_labels.view(-1))
            epoch_val_loss_history.append(loss.item())

    val_loss = torch.tensor(epoch_val_loss_history).mean().item()
    # TODO: probably increase of batch_size -> seq_length causes loss to be less, not sure it's representative
    print(f"{val_loss=}, {best_val_loss=}", flush=True)
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "best_model.pth")
        best_val_loss = val_loss
    val_loss_history.append(epoch_val_loss_history)


with open("train_loss.json", "w") as f:
    json.dump(train_loss_history, f)

with open("val_loss.json", "w") as f:
    json.dump(val_loss_history, f)
