import argparse
import functools
import io
import json
import multiprocessing
import os
import tarfile
import zipfile
from typing import Callable, Iterable, Tuple

import mido

import midiutil
from midiutil import VocabConfig, VocabUtils


def convert_midi_bytes_to_str(cfg: VocabConfig, data: Tuple[str, bytes]) -> Tuple[str, str]:
    filename, filedata = data
    try:
        mid = mido.MidiFile(file=io.BytesIO(filedata))
    except:
        return None
    return filename, midiutil.convert_midi_to_str(cfg, mid)


def midi_to_jsonl(cfg: VocabConfig, path: str, output: str, workers: int = 1):
    pool = multiprocessing.Pool(workers)
    file_generator: Callable[[], Iterable[Tuple[str, bytes]]] = None
    if path.endswith(".tar.gz"):
        def file_generator():
            with tarfile.open(path, "r:gz") as tar:
                for member in tar:
                    if member.isfile():
                        yield (member.name, tar.extractfile(member).read())
    elif path.endswith(".zip"):
        def file_generator():
            with zipfile.ZipFile(path, "r") as zip:
                for member in zip.infolist():
                    if not member.is_dir():
                        yield (member.filename, zip.read(member.filename))
    elif path.endswith((".mid", ".midi")):
        def file_generator():
            with open(path, "rb") as f:
                yield (os.path.basename(path), f.read())
    else:
        raise ValueError(f"Invalid file type: {path}")

    # write results to jsonl file
    with open(output, "w") as f:
        for (filename, result) in pool.map(functools.partial(convert_midi_bytes_to_str, cfg), file_generator(), chunksize=32):
            if result is not None:
                json.dump({"file": filename, "text": result}, f)
                f.write("\n")


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
