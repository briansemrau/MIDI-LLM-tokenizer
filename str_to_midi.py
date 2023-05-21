import argparse
import os

import midiutil
from midiutil import VocabConfig

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "text",
        type=str,
        help="The text to convert to MIDI",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output MIDI file",
    )
    p.add_argument(
        "--vocab_config",
        type=str,
        default="./vocab_config.json",
        help="Path to vocab config file",
    )

    args = p.parse_args()

    cfg = VocabConfig.from_json(args.vocab_config)

    mid = midiutil.convert_str_to_midi(cfg, args.text)
    mid.save(os.path.abspath(args.output))
