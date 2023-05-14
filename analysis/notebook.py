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
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

dataset_path = "/mnt/e/datasets/music/lakh_midi_v0.1/lmd_full.tar.gz"
dataset_name = "Lakh MIDI v0.1"

def file_generator(max: int):
    count = 0
    with tarfile.open(dataset_path, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith((".mid", ".midi")):
                yield tar.extractfile(member).read()
                count += 1
                if count >= max:
                    break
def load_file(f):
    try:
        return mido.MidiFile(file=io.BytesIO(f))
    except:
        return None

def load_and_exec(func, f):
    x = load_file(f)
    if x is not None:
        return func(x), True
    return None, False

def map_dataset(func, count: int):
    with multiprocessing.Pool(nproc) as pool:
        for (x, valid) in tqdm(pool.imap_unordered(functools.partial(load_and_exec, func), file_generator(count), chunksize=64), total=count):
            if valid:
                yield x


#%%
# Count occurrences of each instrument
sample_size = 5000

instr_occ_sum: Dict[int, int] = {}
instr_occ_tuples = []
for i in range(128):
    instr_occ_sum[i] = 0
def count_file_instruments(mid):
    file_instruments: Dict[int, int] = {}
    for track in mid.tracks:
        for msg in track:
            if msg.type == "program_change":
                file_instruments[msg.program] = file_instruments.get(msg.program, 0) + 1
    return file_instruments
for file_instruments in map_dataset(count_file_instruments, sample_size):
    instr_occ_tuples.append(file_instruments.keys())
    for k, v in file_instruments.items():
        instr_occ_sum[k] += 1

# sort by most common, plot top with labels
instrument_labels = json.load(open("instruments.json"))
sorted_instruments_top = dict(sorted(instr_occ_sum.items(), key=lambda item: item[1], reverse=True)[:35])
instruments_labeled = {instrument_labels[str(k+1)]: v for k, v in sorted_instruments_top.items()}
plt.figure(figsize=(10, 5))
plt.title(f"Most Common Instruments\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Instrument")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.bar(instruments_labeled.keys(), instruments_labeled.values())
plt.show()

# plot least common
sorted_instruments_bot = dict(sorted(instr_occ_sum.items(), key=lambda item: item[1])[:35])
instruments_labeled = {instrument_labels[str(k+1)]: v for k, v in sorted_instruments_bot.items()}
plt.figure(figsize=(10, 5))
plt.title(f"Least Common Instruments\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Instrument")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.bar(instruments_labeled.keys(), instruments_labeled.values())
plt.show()

# plot all, unlabeled, sorted by instrument number
plt.figure(figsize=(10, 5))
plt.title(f"Instrument Occurrences\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Instrument")
plt.ylabel("Count")
plt.xticks(np.arange(0, 128, 8))
plt.bar(instr_occ_sum.keys(), instr_occ_sum.values())
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
cmap = mpl.cm.Blues.copy()
cmap.set_under("white")
cmap.set_over("purple")
plt.imshow(co_occurrences, cmap=cmap, vmin=1, vmax=np.quantile(co_occurrences, 0.999))
plt.show()


#%%
# Count occurrences of control change
sample_size = 5000

control_changes: Dict[Tuple[int, int], int] = {}
def count_file_control_changes(mid):
    keys = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "control_change":
                keys.append((msg.control, msg.value))
    return keys
for keys in map_dataset(count_file_control_changes, sample_size):
    for key in keys:
        control_changes[key] = control_changes.get(key, 0) + 1

control_change_labels = json.load(open("control_changes.json"))

# sort by most common grouped by control change, plot top with labels
control_changes_grouped: Dict[int, int] = {}
for k, v in control_changes.items():
    control_changes_grouped[k[0]] = control_changes_grouped.get(k[0], 0) + v
control_changes_labeled = {control_change_labels[str(k)]: v for k, v in control_changes_grouped.items()}
sorted_control_changes_top = dict(sorted(control_changes_labeled.items(), key=lambda item: item[1], reverse=True)[:15])
plt.figure(figsize=(10, 5))
plt.title(f"Most Common Control Changes\n{dataset_name}, n={sample_size:,}")
plt.xlabel("Control Change")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.bar(sorted_control_changes_top.keys(), sorted_control_changes_top.values())
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
plt.show()


#%%
# Count occurrences of each note per instrument
sample_size = 5000

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
plt.title(f"Note Occurrences Heatmap\n{dataset_name}, n={sample_size:,}")
plt.ylabel("Note (normalized by instrument)")
plt.xlabel("MIDI Instrument")
plt.xticks(ticks=np.arange(0-0.5, 128-0.5, 8), labels=np.arange(0, 128, 8))
plt.yticks(ticks=np.arange(0-0.5, 128-0.5, 12), labels=np.arange(0, 128, 12))
# add slight gridlines at every 8 instruments
plt.grid(which='major', axis='y', color='blue', linestyle='-', linewidth=0.5, alpha=0.2)
plt.grid(which='major', axis='x', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
cmap = mpl.cm.Blues.copy()
cmap.set_under("white")
cmap.set_over("purple")
plt.imshow(note_occ_array_norm, origin='lower', cmap=cmap, vmin=0.0000000001, vmax=np.quantile(note_occ_array_norm, 0.9999))
plt.show()
