python train_tokenizer.py \
    --files ./data/lunyu.txt \
    --out model \
    --vocab_size 10000 \
    --min_frequency 2 \
    --limit_alphabet 10000
