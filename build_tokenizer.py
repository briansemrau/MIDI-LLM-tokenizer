import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.normalizers import Lowercase
from transformers import PreTrainedTokenizerFast

from midiutil import VocabConfig, VocabUtils


def build_tokenizer(cfg: VocabConfig):
    utils = VocabUtils(cfg)

    special_tokens = [
        "<pad>",
        "<start>",
        "<end>",
    ]
    vocab = [*special_tokens]
    vocab.extend([utils.format_wait_token(i) for i in range(cfg.wait_events)])
    for i in range(len(cfg.short_instr_bin_names)):
        vocab.extend([utils.format_note_token(i, n, v) for n in range(cfg.note_events) for v in range(cfg.velocity_bins)])

    tokenizer = Tokenizer(WordLevel(vocab={x: i for i, x in enumerate(vocab)}, unk_token="<pad>"))
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.normalizer = Lowercase()
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
