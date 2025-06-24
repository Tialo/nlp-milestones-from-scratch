# GPT-1 From Scratch

## Project Content

This project implements several components from the paper *Improving Language Understanding by Generative Pre-Training*:

* Implementation of the GPT model
* Downloading data for pretraining
* Tokenization
* Pretraining stage
* Supervised fine-tuning (SFT) stage and evaluation

## Comparison with Original Paper

For pretraining, I used 2.5B tokens from the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset.

This dataset is of much higher quality than the one used in the original paper, so using it for pretraining can help the model achieve better results on downstream tasks.

### NLI Results

| Model | MNLI-m | MNLI-mm | SNLI | SciTail | QNLI | RTE |
|-------|--------|---------|------|---------|------|-----|
| GPT-1 (original) | <u>82.1</u> | <u>81.4</u> | <u>89.9</u> | 88.3 | <u>88.1</u> | 56.0 |
| GPT-1 from scratch | 63.5 | 67.1 | - | <u>89.5</u> | 85.8 | <u>64.5</u> |

### QA Results

| Model | Story Cloze | RACE-m | RACE-h | RACE |
|-------|-------------|--------|--------|------|
| GPT-1 (original) | <u>86.5</u> | <u>62.9</u> | <u>57.4</u> | <u>59.0</u> |
| GPT-1 from scratch | - | 25 | 25 | 25 |

### Classification Results

| Model | CoLA | SST2 | MRPC | STSB | QQP |
|-------|------|------|------|------|-----|
| GPT-1 (original) | <u>45.4</u> | <u>91.3</u> | 82.3 | <u>82.0</u> | 70.3 |
| GPT-1 from scratch | 35.0 | 89.2 | <u>85.5</u> | 20.2 | <u>86.5</u> |


## Interpretation of Results

The GPT model from scratch failed on every QA dataset. This is because the content of almost all samples in the dataset is too large for the maximum model length (512 tokens). The input gets truncated, so valuable information needed for the correct answer is lost, causing the model to make random choices.

The same issue applies to other datasets for NLI and Classification tasks. Many samples are too large and get truncated.

However, on other datasets where the input length is appropriate, the model can achieve reasonable metric values.

## Quickstart

1. Clone the project and install requirements:

```bash
git clone git@github.com:Tialo/nlp-milestones-from-scratch.git
cd nlp-milestones-from-scratch/gpt1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. To download data, split it into shards, and train the tokenizer, run:

```bash
python data_utils.py
```
**Note:** This could take significant time. It took 2 hours on my machine.

3. To pretrain the model, run:

```bash
python train.py
```

4. To train and evaluate the model on downstream tasks, run:

```bash
python sft_train.py
```