import random

import torch
import numpy as np
from clearml import Task
from tqdm.auto import tqdm

from transformer import Transformer
from loss import LabelSmoothingLoss
from tokenizer_utils import get_tokenizer, decode
from data_utils import get_data_batch_iterator, load_data


def set_seed(seed: int | None = 42):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

src_tokenizer = get_tokenizer("tokenizer_src.json")
tgt_tokenizer = get_tokenizer("tokenizer_tgt.json")

pad_index = tgt_tokenizer.token_to_id("[PAD]")
start_index = tgt_tokenizer.token_to_id("[START]")
end_index = tgt_tokenizer.token_to_id("[END]")

data = load_data(split="raw")

SAVE_BEST_MODEL = False
TRAIN_FRACTION = 0.8
EPOCHS = 8
BASE_LR = 1
BATCH_SIZE = 96
# original paper used 4% of data for a warmup
WARMUP_FRACTION = 0.3
ACCUMULATION_STEPS = 10
LABEL_SMOOTHING = 0.1

task = Task.init(
    project_name="vanilla-transformer",
    task_name="transfromer-training",
)
task.connect({
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "warmup_fraction": WARMUP_FRACTION,
    "train_fraction": TRAIN_FRACTION,
    "accumulation_steps": ACCUMULATION_STEPS,
    "base_lr": BASE_LR,
})

train_size = int(len(data) * TRAIN_FRACTION)
train_data = data[:train_size]
val_data = data[train_size:]
all_steps = len(train_data) * EPOCHS // (BATCH_SIZE * ACCUMULATION_STEPS)
warmup_steps = int(all_steps * WARMUP_FRACTION)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    src_tokenizer.get_vocab_size(),
    tgt_tokenizer.get_vocab_size(),
).to(device)


criterion = LabelSmoothingLoss(
    ignore_index=pad_index,
    smoothing=LABEL_SMOOTHING,
)


def rate(step: int, d_model: int = 512, warmup: int = warmup_steps):
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

opt = torch.optim.Adam(model.parameters(), lr=BASE_LR, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lr_lambda=rate,
)


def train_one_epoch(model, data_iterator, criterion, opt, scheduler, device):
    model.train()
    epoch_loss_history = []
    accumulated_loss = 0

    for i, (src_tokens, tgt_tokens, src_mask) in enumerate(tqdm(
        data_iterator,
        total=len(train_data) // BATCH_SIZE,
        desc=f"Train epoch {e}",
    )):
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

        if (i + 1) % ACCUMULATION_STEPS == 0:
            global_step = e * (len(train_data) // BATCH_SIZE // ACCUMULATION_STEPS) + i // ACCUMULATION_STEPS
            # log mean of accumulated loss
            task.logger.report_scalar("step_loss", "train", accumulated_loss / ACCUMULATION_STEPS, global_step)
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
    for _ in range(4):
        src_tokens, tgt_tokens, _ = next(val_batch_iterator)
        generated = model.generate(
            src_tokens.to(device),
            start_index,
            end_index,
            # 6.1 We set the maximum output length during inference to input length + 50, but terminate early when possible
            max_tokens=len(src_tokens) + 50,
        )

        task.logger.report_text(
            "Validation example\n"
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
    task.logger.report_scalar("epoch_loss", "train", epoch_train_loss_avg, e)

    val_iterator = get_data_batch_iterator(
        val_data,
        src_tokenizer,
        tgt_tokenizer,
        batch_size=2 * BATCH_SIZE,
    )

    epoch_val_loss_avg = validate_one_epoch(model, val_iterator, criterion, device)
    task.logger.report_scalar("epoch_loss", "val", epoch_val_loss_avg, e)
    
    torch.cuda.empty_cache()


torch.save(model.state_dict(), "model.pth")