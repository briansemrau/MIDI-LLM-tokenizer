# Author: Brian Semrau
# Visualizations for MIDI datasets
# 
# Good reference material:
# https://www.music.mcgill.ca/~ich/classes/mumt306/StandardMIDIfileformat.html


#%%
# Setup
import functools
import io
import json
import mido
import multiprocessing
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm

nproc = multiprocessing.cpu_count()

import tarfile
import zipfile
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# dataset_path = "/mnt/e/datasets/music/lakh_midi_v0.1/lmd_full.tar.gz"
# dataset_name = "Lakh MIDI v0.1"
# dataset_short = "lmd_full"
# dataset_size = 176581
dataset_path = "/mnt/e/datasets/music/midishrine-game-may-2023/files.zip"
dataset_name = "MidiShrine (May 2023)"
dataset_short = "msg_may2023"
dataset_size = 3321
sample_size = dataset_size#10000

if dataset_path.endswith(".tar.gz"):
    def file_generator(max: int):
        count = 0
        with tarfile.open(dataset_path, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith((".mid", ".midi")):
                    yield tar.extractfile(member).read()
                    count += 1
                    if count >= max:
                        break
elif dataset_path.endswith(".zip"):
    def file_generator(max: int):
        count = 0
        with zipfile.ZipFile(dataset_path, "r") as zip:
            for member in zip.infolist():
                if not member.is_dir() and member.filename.endswith((".mid", ".midi")):
                    yield zip.read(member.filename)
                    count += 1
                    if count >= max:
                        break
    
def load_file(f):
    try:
        mid = mido.MidiFile(file=io.BytesIO(f))
        # Filter bad files
        if mid.type not in (0, 1, 2):
            return None
        if len(mid.tracks) == 0:
            return None
        # for track in mid.tracks:
        #     for msg in track:
        #         if msg.type == "note_on" and msg.channel > 15:
        #             return None
        return mid
    except:
        # Filter corrupt
        return None

def load_and_exec(func, f):
    x = load_file(f)
    if x is not None:
        return func(x), True
    return None, False

def map_dataset(func, count: int):
    with multiprocessing.Pool(nproc) as pool:
        for (x, valid) in tqdm(pool.imap_unordered(functools.partial(load_and_exec, func), file_generator(count), chunksize=48), total=count):
            if valid:
                yield x


#%%
# Count which types of midi files
#sample_size = dataset_size

types: Dict[int, int] = {}
def get_file_type(mid):
    return mid.type
for file_type in map_dataset(get_file_type, sample_size):
    types[file_type] = types.get(file_type, 0) + 1

# plot
plt.figure(figsize=(8, 6))
labeled_types = {{0: "Single Track", 1: "Multi Track", 2: "Multi Song",}[k]: v for (k, v) in types.items() if k in [0, 1, 2]}
plt.bar(labeled_types.keys(), labeled_types.values(), color=[mpl.cm.tab10(i) for i in range(len(labeled_types))])
plt.title(f"MIDI File type counts\n{dataset_name}, n={sample_size:,}")
plt.xlabel("MIDI File type")
plt.ylabel("Count")
# label bars with values
for i, v in enumerate(labeled_types.values()):
    plt.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.savefig(f"{dataset_short}_midi_file_types.png", dpi=300)
plt.show()


#%%
# Count occurrences of each instrument
#sample_size = dataset_size

instr_occ_sum: Dict[int, int] = {}
instr_occ_tuples = []
instr_c10_occ_sum: Dict[int, int] = {}
instr_c10_occ_tuples = []
channel_occ_sum: Dict[int, int] = {}
for i in range(128):
    instr_occ_sum[i] = 0
    instr_c10_occ_sum[i] = 0
def count_file_instruments(mid):
    file_instruments: set[int] = set()
    file_instruments_c10: set[int] = set()
    file_channels: set[int] = set()
    for track in mid.tracks:
        for msg in track:
            if hasattr(msg, "channel") and msg.channel <= 15:
                file_channels.add(msg.channel)
            if msg.type == "note_on" and msg.channel == 9:
                file_instruments_c10.add(msg.note)
            if msg.type == "program_change":
                if msg.channel != 9:
                    file_instruments.add(msg.program)
    return file_instruments, file_instruments_c10, file_channels
for file_instruments, file_instruments_c10, file_channels in map_dataset(count_file_instruments, sample_size):
    instr_occ_tuples.append(file_instruments)
    for k in file_instruments:
        instr_occ_sum[k] += 1
    instr_c10_occ_tuples.append(file_instruments_c10)
    for k in file_instruments_c10:
        instr_c10_occ_sum[k] += 1
    for k in file_channels:
        channel_occ_sum[k] = channel_occ_sum.get(k, 0) + 1

instrument_labels = json.load(open("instrument_names.json"))
drum_labels = json.load(open("drum_names.json"))

# plot channel occurrences
plt.figure(figsize=(10, 5))
plt.title(f"Channel Occurrences\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Channel")
plt.ylabel("File Count")
plt.xticks(np.arange(0, 16, 1))
plt.bar(channel_occ_sum.keys(), channel_occ_sum.values())
plt.savefig(f"{dataset_short}_channel_occurrences.png", dpi=300)
plt.show()

# sort by most common, plot top with labels
sorted_instruments_top = dict(sorted(instr_occ_sum.items(), key=lambda item: item[1], reverse=True)[:20])
instruments_labeled = {instrument_labels[str(k+1)]: v for k, v in sorted_instruments_top.items()}
plt.figure(figsize=(10, 5))
plt.title(f"Most Common Instruments\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Instrument")
plt.ylabel("File Count")
plt.xticks(rotation=45, ha="right")
plt.bar(instruments_labeled.keys(), instruments_labeled.values())
plt.savefig(f"{dataset_short}_most_common_instruments.png", dpi=300)
plt.show()

# plot least common
sorted_instruments_bot = dict(sorted(instr_occ_sum.items(), key=lambda item: item[1])[:20])
instruments_labeled = {instrument_labels[str(k+1)]: v for k, v in sorted_instruments_bot.items()}
plt.figure(figsize=(10, 5))
plt.title(f"Least Common Instruments\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Instrument")
plt.ylabel("File Count")
plt.xticks(rotation=45, ha="right")
plt.bar(instruments_labeled.keys(), instruments_labeled.values())
plt.savefig(f"{dataset_short}_least_common_instruments.png", dpi=300)
plt.show()

# plot all, unlabeled, sorted by instrument number
plt.figure(figsize=(10, 5))
plt.title(f"Instrument Occurrences\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Instrument")
plt.ylabel("File Count")
plt.xticks(np.arange(0, 128, 8))
plt.bar(instr_occ_sum.keys(), instr_occ_sum.values())
plt.savefig(f"{dataset_short}_instrument_occurrences.png", dpi=300)
plt.show()

# plot co-occurrences. ignore lower triangle and diagonal
co_occurrences = np.zeros((128, 128))
for file_instruments in instr_occ_tuples:
    for i in file_instruments:
        for j in file_instruments:
            co_occurrences[i, j] += 1
for i in range(128):
    for j in range(i+1):
        co_occurrences[i, j] = 0
plt.figure(figsize=(10, 10))
plt.title(f"Instrument Co-occurrences\n{dataset_name}, n={sample_size:,}")
plt.xlabel("MIDI Instrument")
plt.ylabel("MIDI Instrument")
plt.xticks(ticks=np.arange(0-0.5, 128-0.5, 8), labels=np.arange(0, 128, 8))
plt.yticks(ticks=np.arange(0-0.5, 128-0.5, 8), labels=np.arange(0, 128, 8))
plt.grid(which='major', axis='both', color='blue', linestyle='-', linewidth=0.5, alpha=0.2)
cmap = mpl.cm.Blues.copy()
cmap.set_under("white")
cmap.set_over("purple")
plt.imshow(co_occurrences, cmap=cmap, vmin=1, vmax=np.quantile(co_occurrences, 0.999))
plt.subplots_adjust(hspace=0.5)
plt.savefig(f"{dataset_short}_instrument_co_occurrences.png", dpi=300)
plt.show()

# plot most common drums
sorted_drums_top = dict(sorted(instr_c10_occ_sum.items(), key=lambda item: item[1], reverse=True)[:20])
drums_labeled = {drum_labels.get(str(k+1), "Undefined"): v for k, v in sorted_drums_top.items()}
plt.figure(figsize=(10, 5))
plt.title(f"Most Common Drums\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Drum")
plt.ylabel("File Count")
plt.xticks(rotation=45, ha="right")
plt.bar(drums_labeled.keys(), drums_labeled.values())
plt.savefig(f"{dataset_short}_most_common_drums.png", dpi=300)
plt.show()

# plot least common drums
filtered_drums = {k: v for k, v in instr_c10_occ_sum.items() if str(k+1) in drum_labels}
sorted_drums_bot = dict(sorted(filtered_drums.items(), key=lambda item: item[1])[:20])
drums_labeled = {drum_labels.get(str(k+1), f"{k} (Undefined)"): v for k, v in sorted_drums_bot.items()}
plt.figure(figsize=(10, 5))
plt.title(f"Least Common Drums\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Drum")
plt.ylabel("File Count")
plt.xticks(rotation=45, ha="right")
plt.bar(drums_labeled.keys(), drums_labeled.values())
plt.savefig(f"{dataset_short}_least_common_drums.png", dpi=300)
plt.show()

# plot all drums, sorted by instrument number
plt.figure(figsize=(10, 5))
plt.title(f"Drum Occurrences\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Drum")
plt.ylabel("File Count")
plt.xticks(np.arange(0, 128, 8))
plt.bar(instr_c10_occ_sum.keys(), instr_c10_occ_sum.values())
plt.savefig(f"{dataset_short}_drum_occurrences.png", dpi=300)
plt.show()


#%%
# Count occurrences of control change
#sample_size = 15000#dataset_size

control_changes: Dict[Tuple[int, int], int] = {}
control_changes_file_count: Dict[int, int] = {}
def count_file_control_changes(mid):
    keys = []
    controls_used = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == "control_change":
                keys.append((msg.control, msg.value))
                controls_used.add(msg.control)
    return keys, controls_used
for keys, controls_used in map_dataset(count_file_control_changes, sample_size):
    for key in keys:
        control_changes[key] = control_changes.get(key, 0) + 1
    for control in controls_used:
        control_changes_file_count[control] = control_changes_file_count.get(control, 0) + 1

control_change_labels = json.load(open("control_changes.json"))

# sort by most common grouped by control change, plot top with labels
control_changes_grouped: Dict[int, int] = {}
for k, v in control_changes.items():
    control_changes_grouped[k[0]] = control_changes_grouped.get(k[0], 0) + v
control_changes_labeled = {control_change_labels[str(k)]: v for k, v in control_changes_grouped.items()}
sorted_control_changes_top = dict(sorted(control_changes_labeled.items(), key=lambda item: item[1], reverse=True)[:15])
plt.figure(figsize=(10, 5))
plt.title(f"Most Frequent Control Changes\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Control Change")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.bar(sorted_control_changes_top.keys(), sorted_control_changes_top.values())
plt.savefig(f"{dataset_short}_most_common_control_changes.png", dpi=300)
plt.show()

# heatmap
control_changes_array = np.zeros((128, 128))
for key, value in control_changes.items():
    control_changes_array[key] = value
plt.figure(figsize=(10, 10))
plt.title(f"Control Change Heatmap\n{dataset_name}, n={sample_size:,}")
plt.ylabel("Control Change")
plt.xlabel("Value")
cmap = mpl.cm.Blues.copy()
cmap.set_under("white")
cmap.set_over("purple")
plt.imshow(control_changes_array, cmap=cmap, vmin=1, vmax=np.quantile(control_changes_array, 0.999))
plt.yticks(ticks=np.arange(0, 128, 8), labels=np.arange(0, 128, 8))
plt.xticks(ticks=np.arange(0, 128, 16), labels=np.arange(0, 128, 16))
# y axis text label most common control changes with labels on right
sorted_control_changes_top = list(dict(sorted(control_changes_grouped.items(), key=lambda item: item[1], reverse=True)[:20]).keys())
sorted_control_changes_top.sort()
prev_y = -99
for y in sorted_control_changes_top:
    y_offs = max(prev_y - (y-2), 0)
    prev_y = y + y_offs
    plt.text(130, y+y_offs, control_change_labels[str(y)], ha="left", va="center")
    line = plt.Line2D((127.6, 129.5), (y, y+y_offs), color='black', linewidth=0.5)
    line.set_clip_on(False)
    plt.gca().add_line(line)
plt.subplots_adjust(hspace=0.5)
plt.savefig(f"{dataset_short}_control_change_heatmap.png", dpi=300)
plt.show()

# show control changes file presence (top 15)
control_changes_file_count_labeled = {control_change_labels[str(k)]: v for k, v in control_changes_file_count.items()}
sorted_control_changes_file_count_top = dict(sorted(control_changes_file_count_labeled.items(), key=lambda item: item[1], reverse=True)[:15])
plt.figure(figsize=(10, 5))
plt.title(f"Control Changes File Presence\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Control Change")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.bar(sorted_control_changes_file_count_top.keys(), sorted_control_changes_file_count_top.values())
plt.savefig(f"{dataset_short}_control_changes_file_presence.png", dpi=300)
plt.show()


#%%
# Count occurrences of each note per instrument
#sample_size = 176581

note_occ_sum: Dict[Tuple[int, int], int] = {}
note_occ_tuples = []
for i in range(128):
    for j in range(128):
        note_occ_sum[(i, j)] = 0
def count_file_notes(mid):
    file_notes: Dict[Tuple(int, int), int] = {}
    channel_program = {}
    for track in mid.tracks:
        for msg in track:
            if hasattr(msg, "channel"):
                if msg.channel == 9:
                    continue
            if msg.type == "program_change":
                channel_program[msg.channel] = msg.program
            if msg.type == "note_on":
                key = (msg.note, channel_program.get(msg.channel, 0))
                file_notes[key] = file_notes.get(key, 0) + 1
    return file_notes
for file_notes in map_dataset(count_file_notes, sample_size):
    note_occ_tuples.append(file_notes.keys())
    for k, v in file_notes.items():
        note_occ_sum[k] += 1

# heatmap
note_occ_array = np.zeros((128, 128))
for key, value in note_occ_sum.items():
    note_occ_array[key] = value
# normalize by instrument
note_occ_array_norm = note_occ_array / np.max(note_occ_array, axis=0)
plt.figure(figsize=(10, 10))
plt.title(f"Note Occurrences Heatmap\n(normalized by instrument)\n{dataset_name}, n={sample_size:,}")
plt.ylabel("Note")
plt.xlabel("MIDI Instrument")
plt.xticks(ticks=np.arange(0-0.5, 128-0.5, 8), labels=np.arange(0, 128, 8))
plt.yticks(ticks=np.arange(0-0.5, 128-0.5, 12), labels=np.arange(0, 128, 12))
plt.grid(which='major', axis='y', color='blue', linestyle='-', linewidth=0.5, alpha=0.2)
plt.grid(which='major', axis='x', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
cmap = mpl.cm.Blues.copy()
cmap.set_under("white")
cmap.set_over("purple")
plt.imshow(note_occ_array_norm, origin='lower', cmap=cmap, vmin=0.0000000001, vmax=np.quantile(note_occ_array_norm, 0.9999))
plt.subplots_adjust(hspace=0.5)
plt.savefig(f"{dataset_short}_note_occurrences_heatmap.png", dpi=300)
plt.show()


#%%
# Count meta events
#sample_size = 5000

meta_events: Dict[str, int] = {}
def count_file_meta_events(mid):
    keys = []
    for track in mid.tracks:
        for msg in track:
            msg: mido.Message
            if msg.is_meta:
                keys.append(msg.type)
    return keys
for keys in map_dataset(count_file_meta_events, sample_size):
    for key in keys:
        meta_events[key] = meta_events.get(key, 0) + 1

# sort by most common, plot top with labels
sorted_meta_events_top = dict(sorted(meta_events.items(), key=lambda item: item[1], reverse=True)[:15])
plt.figure(figsize=(10, 5))
plt.title(f"Most Frequent Meta Events\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Meta Event")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.bar(sorted_meta_events_top.keys(), sorted_meta_events_top.values())
plt.savefig(f"{dataset_short}_meta_events.png", dpi=300)
plt.show()
