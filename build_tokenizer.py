import argparse
import json
import os
from typing import List, Dict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from transformers import PreTrainedTokenizerFast


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_SCALE_LEN = len(NOTE_NAMES)


def build_midi_converter(
    program_to_bin: Dict[int, int],
    bin_to_program: Dict[int, int],
    note_events: int,
    wait_events: int,
    max_wait_time: int,
    velocity_bins: int,
    velocity_max: int,
    start_note: int = 0,
    start_octave: int = -1,
):
    pass


def build_tokenizer(
    instrument_names: List[str],
    note_events: int = 128,
    start_note: int = 0,
    start_octave: int = -1,
    wait_events: int = 125,
    max_wait_time: int = 1000,
    velocity_events: int = 128,
    velocity_bins: int = 16,
):
    if velocity_events % velocity_bins != 0:
        raise ValueError("velocity_max must be exactly divisible by velocity_bins")
    if max_wait_time % wait_events != 0:
        raise ValueError("max_wait_time must be exactly divisible by wait_events")

    short_instrument_names = []
    for instr in instrument_names:
        i = min(3, len(instr))
        while instr[:i] in short_instrument_names:
            i += 1
        short_instrument_names.append(instr[:i])

    added_tokens = [
        "<pad>",
        "<start>",
        "<end>",
    ]
    vocab = []
    vocab.extend([f"T{i} " for i in range(wait_events)])
    for instr in short_instrument_names:
        vocab.extend([f"{instr}{v}{NOTE_NAMES[(n+start_note)%NOTE_SCALE_LEN]}{n//NOTE_SCALE_LEN+start_octave} " for v in range(velocity_bins) for n in range(note_events)])

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

    vocab_config = json.load(open(args.vocab_config, "r"))

    build_tokenizer(
        instrument_names=vocab_config["bin_instruments"],
        note_events=vocab_config["note_events"],
        start_note=vocab_config["start_note"],
        start_octave=vocab_config["start_octave"],
        wait_events=vocab_config["wait_events"],
        max_wait_time=vocab_config["max_wait_time"],
        velocity_events=vocab_config["velocity_events"],
        velocity_bins=vocab_config["velocity_bins"],
    )
