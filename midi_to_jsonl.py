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


def mix_volume(velocity: int, volume: int, expression: int) -> float:
    return min(velocity * (volume / 127.0) * (expression / 127.0), 127)


def convert_midi_to_str(cfg: VocabConfig, data: Tuple[str, bytes]) -> Tuple[str, str]:
    filename, filedata = data
    try:
        mid = mido.MidiFile(file=io.BytesIO(filedata))
    except:
        return None
    utils = VocabUtils(cfg)

    if len(mid.tracks) > 1:
        mid.tracks = [mido.merge_tracks(mid.tracks)]

    delta_time_ms = 0.0
    tempo = 500000
    channel_program = {i: 0 for i in range(16)}
    channel_volume = {i: 127 for i in range(16)}
    channel_expression = {i: 127 for i in range(16)}  # unlikely to be useful
    channel_notes = {i: {} for i in range(16)}
    pedal_on = False
    pedal_events = {}

    output = "<start>"

    for msg in mid.tracks[0]:
        time_ms = mido.tick2second(msg.time, mid.ticks_per_beat, tempo) * 1000.0
        delta_time_ms += time_ms

        if msg.is_meta:
            if msg.type == "set_tempo":
                tempo = msg.tempo
            continue

        t = msg.type
        if t == "program_change":
            channel_program[msg.channel] = msg.program
        elif t == "note_on":
            output += utils.data_to_wait_tokens(delta_time_ms)
            delta_time_ms = 0.0
            output += utils.data_to_note_token(
                channel_program[msg.channel],
                mix_volume(msg.velocity, channel_volume[msg.channel], channel_expression[msg.channel]),
                msg.note,
            )
            channel_notes[msg.channel][msg.note] = True
        elif t == "note_off":
            if pedal_on:
                pedal_events[(msg.channel, msg.note)] = True
            else:
                output += utils.data_to_wait_tokens(delta_time_ms)
                delta_time_ms = 0.0
                output += utils.data_to_note_token(
                    channel_program[msg.channel],
                    0,
                    msg.note,
                )
                del channel_notes[msg.channel][msg.note]
        elif t == "control_change":
            if msg.control == 7 or msg.control == 39:  # volume
                channel_volume[msg.channel] = msg.value
            elif msg.control == 11:  # expression
                channel_expression[msg.channel] = msg.value
            elif msg.control == 64:  # sustain pedal
                pedal_on = msg.value >= 64
                if not pedal_on:
                    output += utils.data_to_wait_tokens(delta_time_ms)
                    delta_time_ms = 0.0
                    for (channel, note) in pedal_events:
                        output += utils.data_to_note_token(
                            channel_program[channel],
                            0,
                            note,
                        )
                        del channel_notes[channel][note]
                    pedal_events = {}
            elif msg.control == 123:  # all notes off
                for channel in channel_notes.keys():
                    for note in channel_notes[channel]:
                        output += utils.data_to_wait_tokens(delta_time_ms)
                        delta_time_ms = 0.0
                        output += utils.data_to_note_token(
                            channel_program[channel],
                            0,
                            note,
                        )
                    channel_notes[channel] = {}
        else:
            pass
    output += "<end>"
    return filename, output


# def convert_str_to_midi(cfg: VocabConfig, data: str) -> mido.MidiFile:
#     # placeholder AI generated trash
#     utils = VocabUtils(cfg)
#     mid = mido.MidiFile()
#     track = mido.MidiTrack()
#     mid.tracks.append(track)

#     track.append(mido.Message("program_change", program=0))

#     for token in data.split():
#         if token.startswith("w"):
#             track.append(mido.Message("note_on", time=utils.wait_tokens_to_data(token)))
#         else:
#             (program, velocity, note) = utils.note_token_to_data(token)
#             track.append(mido.Message("program_change", program=program))
#             if velocity > 0:
#                 track.append(mido.Message("note_on", velocity=velocity, note=note))
#             else:
#                 if pedal_on:
#                     pedal_events[(0, note)] = True
#                 else:
#                     track.append(mido.Message("note_on", velocity=0, note=note))
#     return mid


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
        for (filename, result) in pool.map(functools.partial(convert_midi_to_str, cfg), file_generator(), chunksize=32):
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
