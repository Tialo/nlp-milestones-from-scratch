import os

import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from datasets import load_dataset
import matplotlib.pyplot as plt

from sft_utils import (
    ClassificationDataset,
    ClassificationTrainer,
    SimilarityDataset,
    SimilarityTrainer,
    NLIDataset,
    NLITrainer,
    QADataset,
    QATrainer,
)
from tokenizer_utils import Tokenizer
from gpt import GPTClassificator, GPTSimilarity, GPT

torch.set_float32_matmul_precision('high')

PRETRAINED_PATH = "model"
RESULTS_PATH = "sft_results"
MAX_LEN = GPT.from_pretrained(PRETRAINED_PATH).config.max_len

DATASETS_INFO = [
    # ===== CLASSIFICATION =====
    {
        "type": "classification",
        "name": "CoLa",
        "metric": "mc",
        "subset": "cola",
    },
    {
        "type": "classification",
        "name": "SST2",
        "metric": "acc",
        "subset": "sst2",
    },
    # ===== SIMILARITY =====
    {
        "type": "similarity",
        "name": "MRPC",
        "metric": "f1",
        "subset": "mrpc",
        "use_test": True,
    },
    {
        "type": "similarity",
        "name": "STSB",
        "metric": "pc",
        "subset": "stsb",
        "label_scale": 5.0,
    },
    {
        "type": "similarity",
        "name": "QQP",
        "metric": "f1",
        "subset": "qqp",
        "texts1_column": "question1",
        "texts2_column": "question2",
    },
    # ===== NLI =====
    {
        "type": "nli",
        "name": "MNLI-m",
        "subset": "mnli_matched",
        "train_split": "validation[:8000]",
        "val_split": "validation[8000:]",
        "n_targets": 3,
    },
    {
        "type": "nli",
        "name": "MNLI-mm",
        "subset": "mnli_mismatched",
        "train_split": "validation[:8000]",
        "val_split": "validation[8000:]",
        "n_targets": 3,
    },
    {
        "type": "nli",
        "name": "SciTail",
        "dataset": "allenai/scitail",
        "subset": "dgem_format",
        "use_test": True,
    },
    {
        "type": "nli",
        "name": "QNLI",
        "subset": "qnli",
        "premise_column": "question",
        "hypothesis_column": "sentence",
    },
    {
        "type": "nli",
        "name": "RTE",
        "subset": "rte",
        "premise_column": "sentence1",
        "hypothesis_column": "sentence2",
    },
    # ===== QA =====
    {
        "type": "qa",
        "name": "RACE-m",
        "subset": "middle",
    },
    {
        "type": "qa",
        "name": "RACE-h",
        "subset": "high",
    },
    {
        "type": "qa",
        "name": "RACE",
        "subset": "all",
    },
]

tokenizer = Tokenizer.from_pretrained("tokenizer.json")
tokenizer.change_max_len(MAX_LEN)
pad_token_id = tokenizer.token_to_id("[PAD]")
os.makedirs(RESULTS_PATH, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for dataset in DATASETS_INFO:
    dataset_name = dataset["name"]
    dataset_path_name = dataset_name.lower().replace("-", "_")
    if os.path.isfile(os.path.join(RESULTS_PATH, dataset_path_name,'loss.png')):
        continue

    dataset_type = dataset["type"]
    if dataset_type == "classification":
        if dataset["metric"] == "acc":
            metric_fn = accuracy_score
        elif dataset["metric"] == "mc":
            metric_fn = matthews_corrcoef
        else:
            assert 0

        train_hf_dataset = load_dataset("nyu-mll/glue", name=dataset["subset"], split="train")
        val_hf_dataset = load_dataset("nyu-mll/glue", name=dataset["subset"], split="validation")

        train_dataset = ClassificationDataset(train_hf_dataset["sentence"], train_hf_dataset["label"], tokenizer)
        val_dataset = ClassificationDataset(val_hf_dataset["sentence"], val_hf_dataset["label"], tokenizer)

        gpt = GPTClassificator(GPT.from_pretrained(PRETRAINED_PATH), pad_token_id=pad_token_id).to(device)
        gpt = torch.compile(gpt)
        trainer = ClassificationTrainer(
            gpt=gpt,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=5,
            batch_size=32,
            gradient_accumulation_steps=1,
            lr=6.25e-5,
            warmup_fraction=0.04,
            pad_token_id=pad_token_id,
            metric_fn=metric_fn,
            device=device,
        )
    elif dataset_type == "similarity":
        if dataset["metric"] == "f1":
            metric_fn = f1_score
        elif dataset["metric"] == "pc":
            def custom_pc(y_true, y_pred):
                return float(pearsonr(y_true, y_pred)[0])  # symmetric
            metric_fn = custom_pc
        else:
            assert 0

        train_hf_dataset = load_dataset("nyu-mll/glue", name=dataset["subset"], split="train")
        val_split = "validation+test" if dataset.get("use_test") else "validation"
        val_hf_dataset = load_dataset("nyu-mll/glue", name=dataset["subset"], split=val_split)

        texts1_column = dataset.get("texts1_column", "sentence1")
        texts2_column = dataset.get("texts2_column", "sentence2")
        label_scale = dataset.get("label_scale", 1.0)
        train_dataset = SimilarityDataset(train_hf_dataset[texts1_column], train_hf_dataset[texts2_column], train_hf_dataset["label"], tokenizer, label_scale)
        val_dataset = SimilarityDataset(val_hf_dataset[texts1_column], val_hf_dataset[texts2_column], val_hf_dataset["label"], tokenizer, label_scale)

        gpt = GPTSimilarity(GPT.from_pretrained(PRETRAINED_PATH), pad_token_id=pad_token_id).to(device)
        gpt = torch.compile(gpt)
        trainer = SimilarityTrainer(
            gpt=gpt,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=5,
            batch_size=32,
            gradient_accumulation_steps=4,
            lr=6.25e-5,
            warmup_fraction=0.04,
            pad_token_id=pad_token_id,
            metric_fn=metric_fn,
            device=device,
        )
    elif dataset_type == "nli":
        dataset_path = dataset.get("dataset", "nyu-mll/glue")

        train_split = "train"
        val_split = "validation"
        if "train_split" in dataset:
            train_split = dataset['train_split']
            val_split = dataset['val_split']
        elif "use_test" in dataset:
            val_split += "+test"

        train_hf_dataset = load_dataset(dataset_path, name=dataset["subset"], split=train_split)
        val_hf_dataset = load_dataset(dataset_path, name=dataset["subset"], split=val_split)

        premise_column = dataset.get("premise_column", "premise")
        hypothesis_column = dataset.get("hypothesis_column", "hypothesis")
        le = LabelEncoder()
        train_dataset = NLIDataset(train_hf_dataset[premise_column], train_hf_dataset[hypothesis_column], train_hf_dataset["label"], tokenizer, le)
        val_dataset = NLIDataset(val_hf_dataset[premise_column], val_hf_dataset[hypothesis_column], val_hf_dataset["label"], tokenizer, le)

        n_targets = dataset.get("n_targets", 1)
        criterion = nn.BCEWithLogitsLoss() if n_targets == 1 else nn.CrossEntropyLoss()
        gpt = GPTClassificator(GPT.from_pretrained(PRETRAINED_PATH), pad_token_id=pad_token_id, n_targets=n_targets).to(device)
        gpt = torch.compile(gpt)
        trainer = NLITrainer(
            gpt=gpt,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=5,
            batch_size=32,
            gradient_accumulation_steps=4,
            lr=6.25e-5,
            criterion=criterion,
            warmup_fraction=0.04,
            pad_token_id=pad_token_id,
            device=device,
        )
    elif dataset_type == "qa":
        train_hf_dataset = load_dataset('ehovy/race', name=dataset["subset"], split="train")
        val_hf_dataset = load_dataset('ehovy/race', name=dataset["subset"], split="validation+test")

        train_dataset = QADataset(
            articles=train_hf_dataset["article"],
            questions=train_hf_dataset["question"],
            options=train_hf_dataset["options"],
            answers=train_hf_dataset["answer"],
            tokenizer=tokenizer,
        )
        val_dataset = QADataset(
            articles=val_hf_dataset["article"],
            questions=val_hf_dataset["question"],
            options=val_hf_dataset["options"],
            answers=val_hf_dataset["answer"],
            tokenizer=tokenizer,
        )

        gpt = GPTClassificator(GPT.from_pretrained(PRETRAINED_PATH), pad_token_id=pad_token_id).to(device)
        gpt = torch.compile(gpt)
        trainer = QATrainer(
            gpt=gpt,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=5,
            batch_size=32,
            gradient_accumulation_steps=16,
            lr=6.25e-5,
            warmup_fraction=0.04,
            pad_token_id=pad_token_id,
            device=device,
        )
    else:
        assert 0

    os.makedirs(os.path.join(RESULTS_PATH, dataset_path_name), exist_ok=True)

    print(f"Train and evaluation of {dataset_name} dataset")
    loss_history, metric_history = trainer.train()

    plt.figure(figsize=(11, 7))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.title(f'{dataset_name} loss history')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_PATH, dataset_path_name,'loss.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(11, 7))
    plt.plot(range(1, len(metric_history) + 1), metric_history)
    plt.title(f'{dataset_name} metric history')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_PATH, dataset_path_name,'metric.png'), dpi=300)
    plt.close()
