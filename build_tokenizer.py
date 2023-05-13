import argparse
import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from transformers import PreTrainedTokenizerFast

from util import VocabConfig, VocabUtils


def build_tokenizer(cfg: VocabConfig):
    if cfg.velocity_events % cfg.velocity_bins != 0:
        raise ValueError("velocity_max must be exactly divisible by velocity_bins")
    if cfg.max_wait_time % cfg.wait_events != 0:
        raise ValueError("max_wait_time must be exactly divisible by wait_events")

    utils = VocabUtils(cfg)

    added_tokens = [
        "<pad>",
        "<start>",
        "<end>",
    ]
    vocab = []
    vocab.extend([utils.format_wait_token(i) for i in range(cfg.wait_events)])
    for i in range(len(cfg.short_instrument_names)):
        vocab.extend([utils.format_note_token(i, v, n) for n in range(cfg.note_events) for v in range(cfg.velocity_bins)])

    tokenizer = Tokenizer(WordLevel(vocab={x: i for i, x in enumerate(vocab)}, unk_token="<pad>"))
    tokenizer.add_tokens(added_tokens)
    tokenizer.save("tmp_tokenizer.json")
    
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tmp_tokenizer.json",
        model_input_names=["input_ids"],
        bos_token="<start>",
        eos_token="<end>",
        pad_token="<pad>",
        unk_token="<pad>",
    )
    fast_tokenizer.save_pretrained("tokenizer-midi")
    os.remove("tmp_tokenizer.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vocab_config",
        type=str,
        default="./vocab_config.json",
        help="Path to vocab config file",
    )

    args = p.parse_args()

    build_tokenizer(VocabConfig.from_json(args.vocab_config))
