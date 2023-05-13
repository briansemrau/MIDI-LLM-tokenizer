import argparse
import functools
import io
import json
import mido
import multiprocessing
import os
from typing import Iterable, Callable, Tuple

import tarfile
import zipfile

from util import VocabConfig, VocabUtils


def convert_midi_to_str(cfg: VocabConfig, data: Tuple[str, bytes]) -> Tuple[str, str]:
    filename, filedata = data
    try:
        mid = mido.MidiFile(file=io.BytesIO(filedata))
    except:
        return None
    utils = VocabUtils(cfg)

    if len(mid.tracks) > 1:
        mid.tracks = [mido.merge_tracks(mid.tracks)]

    channel_programs = {0: 0 for _ in range(16)}
    delta_time = 0
    pedal_on = False
    pedal_events = {}

    output = ""

    for msg in mid.tracks[0]:
        delta_time += msg.time

        if msg.is_meta:
            continue

        t = msg.type
        if t == "program_change":
            channel_programs[msg.channel] = msg.program
        elif t == "note_on":
            output += utils.data_to_wait_tokens(delta_time)
            output += utils.data_to_note_token(
                channel_programs[msg.channel],
                msg.velocity,
                msg.note,
            )
        elif t == "note_off":
            if pedal_on:
                pedal_events[(msg.channel, msg.note)] = True
            else:
                output += utils.data_to_wait_tokens(delta_time)
                output += utils.data_to_note_token(
                    channel_programs[msg.channel],
                    0,
                    msg.note,
                )
        elif t == "control_change":
            if msg.control == 64:
                pedal_on = msg.value >= 64
                if not pedal_on:
                    output += utils.data_to_wait_tokens(delta_time)
                    for (channel, note) in pedal_events:
                        output += utils.data_to_note_token(
                            channel_programs[channel],
                            0,
                            note,
                        )
                    pedal_events = {}
        else:
            pass
    return filename, output


def midi_to_jsonl(cfg: VocabConfig, path: str, output: str, workers: int = 1):
    pool = multiprocessing.Pool(workers)
    file_generator: Callable[[], Iterable[Tuple[str, bytes]]] = None
    if path.endswith(".tar.gz"):
        def file_generator():
            with tarfile.open(path, "r:gz") as tar:
                for member in tar.getmembers():
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
        for (result, filename) in pool.map(functools.partial(convert_midi_to_str, cfg), file_generator(), chunksize=32):
            if result is not None:
                json.dump({"text": result, "file": filename}, f)
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
