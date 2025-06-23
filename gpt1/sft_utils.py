from typing import TYPE_CHECKING
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm, trange
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from tokenizer_utils import Tokenizer
    from gpt import GPTClassificator


# ================ COMMON ================

# https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/optimization.py#L99
def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


class GPTDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenized = []
        self.labels = []

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, index):
        return self.tokenized[index], self.labels[index]


class BaseTrainer:
    def __init__(
        self,
        gpt: "GPTClassificator",
        train_dataset: GPTDataset,
        val_dataset: GPTDataset,
        epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        lr: float,
        criterion,
        warmup_fraction: float,
        pad_token_id: int,
        base_collate_fn,
        metric_fn,
        device: torch.device,
    ):
        self.gpt = gpt
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.criterion = criterion
        self.warmup_fraction = warmup_fraction
        self.collate_fn = partial(base_collate_fn, pad_token_id=pad_token_id)
        self.metric_fn = metric_fn
        self.device = device

    def get_opt(self):
        training_steps = len(self.train_dataset) * self.epochs // self.batch_size
        lr_lambda = partial(
            _get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=int(training_steps * self.warmup_fraction),
            num_training_steps=training_steps,
        )
        no_decay_parameters, decay_parameters = self.gpt.get_splitted_params_for_opt()
        opt = torch.optim.AdamW(
            [
                {"params": no_decay_parameters, "weight_decay": 0.0},
                {"params": decay_parameters, "weight_decay": 0.01},
            ],
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return opt, scheduler

    def train_one_epoch(self, loader: torch.utils.data.DataLoader, opt, scheduler):
        torch.cuda.empty_cache()
        self.gpt.train()
        loss_history = []
        accumulated_loss = 0.0
        for step_index, (inputs, labels) in enumerate(tqdm(loader, position=1, leave=False), start=1):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            logits = self.gpt(inputs)
            if logits.size(1) == 1:
                logits = logits.squeeze(1)
                labels = labels.type(torch.float)
            else:  # CrossEntropyLoss is used
                labels = labels.type(torch.long)
            loss = self.criterion(logits, labels) / self.gradient_accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()

            if step_index % self.gradient_accumulation_steps == 0:
                opt.step()
                scheduler.step()
                opt.zero_grad()
                loss_history.append(accumulated_loss)
                accumulated_loss = 0.0
        return loss_history

    def evaluate_one_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        raise NotImplementedError

    def train(self):
        opt, scheduler = self.get_opt()
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // self.gradient_accumulation_steps,
            collate_fn=self.collate_fn,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

        loss_history = []
        metric_history = []
        for _ in trange(self.epochs, position=0):
            loss_history.extend(self.train_one_epoch(train_loader, opt, scheduler))
            metric_history.append(self.evaluate_one_epoch(val_loader))

        return loss_history, metric_history

# ======== CLASSIFICATION UTILS ========

class ClassificationDataset(GPTDataset):
    def __init__(self, texts, labels, tokenizer: "Tokenizer"):
        super().__init__()
        for text, label in zip(texts, labels):
            self.tokenized.append(tokenizer.encode(text, add_end_token=False, add_ext_token=True))
            self.labels.append(label)


def classification_collate_fn(batch, pad_token_id):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
    labels = torch.tensor(labels)  # (batch_size,)
    return inputs, labels


class ClassificationTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        criterion = kwargs.pop("criterion", nn.BCEWithLogitsLoss())
        base_collate_fn = kwargs.pop("base_collate_fn", classification_collate_fn)
        super().__init__(*args, **kwargs, criterion=criterion, base_collate_fn=base_collate_fn)

    @torch.no_grad
    def evaluate_one_epoch(self, loader: torch.utils.data.DataLoader):
        torch.cuda.empty_cache()
        self.gpt.eval()
        all_labels = []
        predicted_labels = []
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            logits = self.gpt(inputs)  # (batch_size, n_targets)
            if logits.size(1) == 1:
                logits = logits.squeeze(1)
                predicted = logits > 0
            else:
                predicted = logits.argmax(dim=1)
            predicted_labels.extend(predicted.cpu().type(torch.int).tolist())
            all_labels.extend(labels.cpu().type(torch.int).tolist())
        return self.metric_fn(y_true=all_labels, y_pred=predicted_labels)


# ======== NLI UTILS ========

class NLIDataset(GPTDataset):
    def __init__(self, premises, hypotheses, labels, tokenizer: "Tokenizer", label_encoder: LabelEncoder):
        super().__init__()
        raw_labels = []
        for premise, hypothesis, label in zip(premises, hypotheses, labels):
            self.tokenized.append(tokenizer.encode_pair((premise, hypothesis)))
            raw_labels.append(label)

        if hasattr(label_encoder, 'classes_'):
            self.labels = label_encoder.transform(raw_labels)
        else:
            self.labels = label_encoder.fit_transform(raw_labels)


class NLITrainer(ClassificationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, metric_fn=accuracy_score, base_collate_fn=classification_collate_fn)


# ======== QA UTILS ========

class QADataset(GPTDataset):
    def __init__(self, articles, questions, options, answers, tokenizer: "Tokenizer"):
        super().__init__()
        answer_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}

        for article, question, option, answer in zip(articles, questions, options, answers):
            assert len(option) == 4
            context = article + " " + question
            sample_tokenized = []
            for current_option in option:
                sample_tokenized.append(tokenizer.encode_pair((context, current_option)))
            self.tokenized.append(sample_tokenized)
            self.labels.append(answer_mapping[answer])


def qa_collate_fn(batch, pad_token_id):
    inputs = [inner_item for item in batch for inner_item in item[0]]
    labels = [item[1] for item in batch]
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)   # (batch_size * 4, seq_len)
    labels = torch.tensor(labels)  # (batch_size,)
    return inputs, labels


class QATrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, criterion=nn.CrossEntropyLoss(), metric_fn=accuracy_score, base_collate_fn=qa_collate_fn)

    @torch.no_grad
    def evaluate_one_epoch(self, loader: DataLoader) -> float:
        torch.cuda.empty_cache()
        self.gpt.eval()
        all_labels = []
        predicted_labels = []
        for inputs, labels in loader:
            inputs = inputs.to(self.device)  # (batch_size * 4, seq_len)
            logits = self.gpt(inputs)  # (batch_size * 4, 1)
            logits = logits.view(-1, 4)  # (batch_size, 4)
            predicted = logits.argmax(dim=1)  # (batch_size,)
            predicted_labels.extend(predicted.cpu().type(torch.int).tolist())
            all_labels.extend(labels.cpu().type(torch.int).tolist())
        return self.metric_fn(y_true=all_labels, y_pred=predicted_labels)

    def train_one_epoch(self, loader: DataLoader, opt, scheduler):
        torch.cuda.empty_cache()
        self.gpt.train()
        loss_history = []
        accumulated_loss = 0.0
        for step_index, (inputs, labels) in enumerate(tqdm(loader, position=1, leave=True), start=1):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)  # (batch_size, 1)
            logits = self.gpt(inputs)  # (batch_size * 4, 1)
            logits = logits.view(-1, 4)  # (batch_size, 4)
            loss = self.criterion(logits, labels) / self.gradient_accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()

            if step_index % self.gradient_accumulation_steps == 0:
                opt.zero_grad()
                opt.step()
                scheduler.step()
                loss_history.append(accumulated_loss)
                accumulated_loss = 0.0
        return loss_history


# ======= SIMILARITY UTILS =======

class SimilarityDataset(GPTDataset):
    def __init__(self, texts1, texts2, labels, tokenizer: "Tokenizer", label_scale: float = 1):
        super().__init__()
        for text1, text2, label in zip(texts1, texts2, labels):
            self.tokenized.append([
                tokenizer.encode_pair((text1, text2)),
                tokenizer.encode_pair((text2, text1)),
            ])
            self.labels.append(label / label_scale)


def sim_collate_fn(batch, pad_token_id):
    inputs = [inner_item for item in batch for inner_item in item[0]]
    labels = [item[1] for item in batch]
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)   # (batch_size * 2, seq_len)
    labels = torch.tensor(labels)  # (batch_size,)
    return inputs, labels


class SimilarityTrainer(ClassificationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, criterion=nn.BCEWithLogitsLoss(), base_collate_fn=sim_collate_fn)
