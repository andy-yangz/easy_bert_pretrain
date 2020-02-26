# easy_bert_pretrain
The very easy BERT pretrain process by  using tokenizers and transformers repos.

### Requirements

So we need following repos:

```
tokenizers==0.5.0
transformers==2.5.0
```

### How to use

#### Tokenize corpus

First, you need data, lots of data. If not, you still can train anyway.

So I put an sample of text, lunyun(论语), the bible of confucianist, in the repo. How nice am I.

Then you just need set the `train_tokenizer.sh` script and run it.

```bash
python train_tokenizer.py \
    --files ./data/lunyun.txt \
    --out model \
    --vocab_size 10000 \
    --min_frequency 2 \
    --limit_alphabet 10000
```

#### Pretrain BERT

Second, just set `pretrain_bert.sh` and run it.

```bash
python run_language_modeling.py \
    --train_data_file ./data/lunyu.txt \
    --output_dir ./model/ \
    --model_type bert \
    --mlm \
    --config_name ./model/ \
    --tokenizer_name ./model/ \
    --do_train \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_gpu_train_batch_size 16 \
    --evaluate_during_training \
    --line_by_line \
    --block_size 256
```

If you need specify your dataset processing, you can edit the Dataset class in `run_language_modeling.py` file.

