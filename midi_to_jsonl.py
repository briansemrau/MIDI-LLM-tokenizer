import argparse
import functools
import io
import json
import multiprocessing
import os
import tarfile
import zipfile
from typing import Iterable, List, Optional, Tuple, Union

import mido
from tqdm import tqdm

import midi_util
from midi_util import AugmentConfig, VocabConfig


def convert_midi_bytes_to_str(cfg: VocabConfig, aug_cfg: AugmentConfig, data: Tuple[str, bytes]) -> Tuple[str, Union[str, List, None]]:
    filename, filedata = data
    try:
        mid = mido.MidiFile(file=io.BytesIO(filedata))
    except:
        return filename, None
    if mid.type not in (0, 1, 2):
        return filename, None
    if len(mid.tracks) == 0:
        return filename, None
    
    if aug_cfg is not None:
        return filename, [midi_util.convert_midi_to_str(cfg, mid, augment) for augment in aug_cfg.get_augment_values(filename)]

    return filename, midi_util.convert_midi_to_str(cfg, mid)


def midi_to_jsonl(cfg: VocabConfig, path: str, output: str, augment_config: Optional[AugmentConfig] = None, workers: int = 1):
    pool = multiprocessing.Pool(workers)
    if path.endswith(".tar.gz"):
        def file_generator() -> Iterable[Tuple[str, bytes]]:
            with tarfile.open(path, "r:gz") as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith((".mid", ".midi")):
                        yield (member.name, tar.extractfile(member).read())
    elif path.endswith(".zip"):
        def file_generator() -> Iterable[Tuple[str, bytes]]:
            with zipfile.ZipFile(path, "r") as zip:
                for member in zip.infolist():
                    if not member.is_dir() and member.filename.endswith((".mid", ".midi")):
                        yield (member.filename, zip.read(member.filename))
    elif path.endswith((".mid", ".midi")):
        def file_generator() -> Iterable[Tuple[str, bytes]]:
            with open(path, "rb") as f:
                yield (os.path.basename(path), f.read())
    else:
        raise ValueError(f"Invalid file type: {path}")

    failed_file_count = 0
    total_file_count = 0

    # write results to jsonl file
    with open(output, "w") as f, open(output + ".failed", "w") as f_failed:
        for (filename, result) in tqdm(pool.imap(functools.partial(convert_midi_bytes_to_str, cfg, augment_config), file_generator(), chunksize=48)):
            total_file_count += 1
            if result is not None:
                if type(result) is list:
                    for r in result:
                        f.write(json.dumps({"file": filename, "text": r}) + "\n")
                else:
                    f.write(json.dumps({"file": filename, "text": result}) + "\n")
            else:
                f_failed.write(filename + "\n")
                failed_file_count += 1
    
    print(f"Failed to convert {failed_file_count} files ({failed_file_count / total_file_count * 100:.2f}%)")
    if failed_file_count == 0:
        os.remove(output + ".failed")


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
        "--augment_config",
        type=str,
        help="Path to augment config file",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use for parallel processing",
    )
    args = p.parse_args()

    cfg = VocabConfig.from_json(args.vocab_config)

    augment_config = None
    if args.augment_config is not None:
        augment_config = AugmentConfig.from_json(args.augment_config, cfg)

    midi_to_jsonl(cfg, args.path, args.output, augment_config, args.workers)
