import argparse
import io
import mido
import multiprocessing

from util import VocabConfig, VocabUtils


def convert_midi_to_str(cfg: VocabConfig, data: bytes) -> str:
    try:
        mid = mido.MidiFile(file=io.BytesIO(data))
    except:
        return None

    pass


def midi_to_jsonl(cfg: VocabConfig, path: str, output: str, workers: int = 1):
    pool = multiprocessing.Pool(workers)
    if path.endswith(".tar.gz"):
        pass
    if path.endswith(".zip"):
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a folder or archive containing MIDI files",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file",
    )
    p.add_argument(
        "--vocab_config",
        type=str,
        default="./vocab_config.json",
        help="Path to vocab config file",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use for parallel processing",
    )

    args = p.parse_args()

    cfg = VocabConfig.from_json(args.vocab_config)

    midi_to_jsonl(cfg, args.path, args.output, args.workers)
