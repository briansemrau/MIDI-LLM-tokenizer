import argparse
import functools
import hashlib
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


def midi_to_jsonl(cfg: VocabConfig, path: str, output: str, augment_config: Optional[AugmentConfig] = None, workers: int = 1, deduplicate: bool = False):
    file_md5s = set()
    duplicate_file_count = 0

    def check_dedup(filebytes: bytes):
        nonlocal duplicate_file_count
        if deduplicate:
            file_md5 = hashlib.md5(filebytes[:512]).update(filebytes[-256:])  # no need to hash whole file, and this is a main-thread hot path
            if file_md5 in file_md5s:
                duplicate_file_count += 1
                return True
            file_md5s.add(file_md5)
        return False
    
    pool = multiprocessing.Pool(workers)
    if path.endswith(".tar.gz"):
        def file_generator() -> Iterable[Tuple[str, bytes]]:
            with tarfile.open(path, "r:gz") as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith((".mid", ".midi")):
                        filebytes = tar.extractfile(member).read()
                        if check_dedup(filebytes):
                            continue
                        yield (member.name, filebytes)
    elif path.endswith(".zip"):
        def file_generator() -> Iterable[Tuple[str, bytes]]:
            with zipfile.ZipFile(path, "r") as zip:
                for member in zip.infolist():
                    if not member.is_dir() and member.filename.endswith((".mid", ".midi")):
                        filebytes = zip.read(member.filename)
                        if check_dedup(filebytes):
                            continue
                        yield (member.filename, filebytes)
    elif path.endswith((".mid", ".midi")):
        def file_generator() -> Iterable[Tuple[str, bytes]]:
            with open(path, "rb") as f:
                filebytes = f.read()
                yield (os.path.basename(path), filebytes)
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
    
    total_file_count_dup = total_file_count + duplicate_file_count

    if deduplicate:
        print(f"Skipped {duplicate_file_count} duplicate files ({duplicate_file_count / (total_file_count_dup) * 100:.2f}%)")
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
    p.add_argument(
        "--deduplicate",
        type=bool,
        default=False,
        help="Deduplicate MIDI files using their MD5 hash. (It's likely better practice to dedup the data before preprocessing it here.)",
    )
    args = p.parse_args()

    cfg = VocabConfig.from_json(args.vocab_config)

    augment_config = None
    if args.augment_config is not None:
        augment_config = AugmentConfig.from_json(args.augment_config, cfg)

    midi_to_jsonl(cfg, args.path, args.output, augment_config, args.workers, args.deduplicate)
