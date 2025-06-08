### Pure Vanilla Transformer from scratch

I started this project because I wanted a clearer understanding of the Transformer architecture. While reading *Attention Is All You Need*, I set out to implement it as closely as possible to the original paper. However, most available implementations included later tweaks and improvements. That made me wonder if I fully understood the original design.

So I built this version from scratch. It follows the paper closely by default, but also allows you to enable more modern practices to see how they affect performance and compare results.

### Quickstart

* Clone the project and install requirements:
```bash
git clone git@github.com:Tialo/pure-vanilla-transformer-from-scratch.git
cd vanilla-transformer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
* To **train** a model on the en/ru NMT dataset, simply run:
```bash
python train.py
```
* To **evaluate** a model and compute the BLEU score, run:
```bash
python evaluate_model.py model tokenizer.json
```
* To change architecture or training parameters, see:
```bash
python train.py --help
```

### Training Tweaks Explained

This project is heavily inspired by [Annotated Transformer](https://github.com/harvardnlp/annotated-transformer/), where most advanced techniques were used.

* `--use_cross_entropy` and `--use_kl_divergence`

    Uses custom cross-entropy loss or KL divergence loss between the label-smoothed distribution and the predicted distribution. Since KLDiv = CE - entropy, and the gradient of entropy with respect to the model's parameters is 0, the gradient of KLDiv equals that of CE. Implemented this parameter just to verify this relation. This parameter changes optimization function.

* `--tie_embeddings` and `--no_tie_embeddings`

    Some implementations neglected tying embeddings with the pre-softmax layer. This not only reduces the number of model parameters (by 4 millions for a vocab_size of 8,192), but in my case, it also sped up convergence and improved BLEU results by ~10%. These parameters control embedding tying.

* `--post_ln` and `--pre_ln`

    These parameters control the order in which Layer Normalization is applied in Encoder and Decoder Layers. [See also](https://github.com/harvardnlp/annotated-transformer/issues/92#issuecomment-1132966376) for more historical context.

* `--add_two_layer_norms`

    This parameter applies Layer Normalization to Encoder and Decoder outputs (not individual layers). 

* `--use_additional_dropout`

    It's common to use dropout after softmax in scaled dot-product attention and in the Feed Forward layer inside Multi-Head Attention layer, but neither of these were proposed in the original paper. This parameter sets dropout rates in these parts to 0.1.

* `--xavier_initialization`

    This parameter applies `xavier_uniform_` parameters initialization to the model.
