import argparse
import glob
import time
from pathlib import Path

from tokenizers import (ByteLevelBPETokenizer,
                        CharBPETokenizer,
                        SentencePieceBPETokenizer,
                        BertWordPieceTokenizer)

TOKENIZERS = {"byte": ByteLevelBPETokenizer, 
              "char": CharBPETokenizer, 
              "sentencepiece": SentencePieceBPETokenizer, 
              "bert": BertWordPieceTokenizer}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="The files to use as training; accept '**/*.txt' type of patterns \
          if enclosed in quotes",
) #训练文件，允许 **/*.txt 这样多个文件，但需要注意文件打开上限否则会报错.

parser.add_argument(
    "--out",
    default="./",
    type=str,
    help="Path to the output directory, where the files will be saved",
)
parser.add_argument(
    "--name", default="bert-wordpiece", type=str, help="The name of the output vocab files"
)

# Training Parameters
parser.add_argument(
    "--vocab_size", default=10000, type=int, help="The vocab size set for trainer"
)
parser.add_argument(
    "--min_frequency", default=2, type=int, help="The minimal freqency to keep "
)
parser.add_argument(
    "--special_tokens", nargs='+', default=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    help="The special tokens"
)
parser.add_argument(
    "--limit_alphabet", default=1000, type=int, 
    help="The alphabet number. \
          Every chinese character will be taken as a new alphabet, so you need set it large."
)
args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    print(f"File does not exist: {args.files}")
    exit(1)


# Initialize an empty tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True,
)

# And then train
trainer = tokenizer.train(
    files,
    vocab_size=args.vocab_size,
    min_frequency=args.min_frequency,
    show_progress=True,
    special_tokens=args.special_tokens,
    limit_alphabet=args.limit_alphabet,
    wordpieces_prefix="##",
)

# Save the files
Path(args.out).mkdir(parents=True, exist_ok=True)
tokenizer.save(args.out, args.name)