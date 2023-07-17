#%%
# process

from typing import Dict, Tuple
import numpy as np
import os
import mido
from math import ceil
import io
import zipfile
import multiprocessing
import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

nproc = max(1, multiprocessing.cpu_count() - 1)

dataset_path = "/mnt/e/datasets/music/los-angeles-midi/Los-Angeles-MIDI-Dataset-Ver-3-0-CC-BY-NC-SA.zip"
dataset_name = "Los Angeles MIDI Dataset 3.0"
dataset_short = "lam3"
dataset_size = 232000
sample_size = 23200

# dataset_path = "/mnt/e/datasets/music/GiantMIDI-Piano/midis_v1.2.zip"
# dataset_name = "GiantMIDI Piano (v1.2)"
# dataset_short = "gmp"
# dataset_size = 10855
# sample_size = dataset_size

def file_generator(max: int):
    count = 0
    with zipfile.ZipFile(dataset_path, "r") as zip:
        for member in zip.infolist():
            if not member.is_dir() and member.filename.endswith((".mid", ".midi")):
                filebytes = zip.read(member.filename)
                yield filebytes
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

def mix_volume(velocity: int, volume: int, expression: int) -> float:
    return velocity * (volume / 127.0) * (expression / 127.0)

vel_occ_sum: Dict[Tuple[int, int], int] = {}
vel_occ_tuples = []
for i in range(128):
    for j in range(128*2):
        vel_occ_sum[(i, j)] = 0
def count_file_notes(mid):
    file_vels: Dict[Tuple(int, int), int] = {}
    channel_program = {i: 0 for i in range(16)}
    channel_volume = {i: 127 for i in range(16)}
    channel_expression = {i: 127 for i in range(16)}
    for track in mid.tracks:
        for msg in track:
            msg_prog = 0
            if hasattr(msg, "channel"):
                msg_prog = channel_program.get(msg.channel, 0)
                if msg.channel == 9 and hasattr(msg, "note"):
                    msg_prog = 128 + msg.note
            if msg.type == "program_change":
                channel_program[msg.channel] = msg.program
            if msg.type == "control_change":
                if msg.control == 7 or msg.control == 39:  # volume
                    channel_volume[msg.channel] = msg.value
                elif msg.control == 11:  # expression
                    channel_expression[msg.channel] = msg.value
            if msg.type == "note_on":
                msg_vel = int(mix_volume(msg.velocity, channel_volume[msg.channel], channel_expression[msg.channel]))
                key = (msg_vel, msg_prog)
                file_vels[key] = file_vels.get(key, 0) + 1
    return file_vels
for file_vels in map_dataset(count_file_notes, sample_size):
    vel_occ_tuples.append(file_vels.keys())
    for k, v in file_vels.items():
        vel_occ_sum[k] += 1

vel_occ_array = np.zeros((128, 128))
drum_vel_occ_array = np.zeros((128, 128))
for key, value in vel_occ_sum.items():
    if key[1] >= 128:
        drum_vel_occ_array[key[0], key[1]-128] = value
    else:
        vel_occ_array[key] = value

#%%
# analyze

velocity_bins = 12

vel_dist = np.cumsum(vel_occ_array.sum(axis=1)[1:])  # discard 0 (note off)
percentiles = np.linspace(0.0, 1.0, velocity_bins)
def where_cumdistr_percentile(d, p):  # d is a cumulative distribution
    m = np.min(d)
    w = np.max(d) - m
    return np.where(d >= m + p * w)[0][0]
pct_idx = [1 + where_cumdistr_percentile(vel_dist, p) for p in percentiles]  # plus one for note off
pct_idx[0] = 0  # fix off-by-one for first bin (1 -> 0)
pct_idx[-1] = 127  # some datasets like GiantMIDI Piano have zero notes at velocity=127

data = vel_occ_array.sum(axis=1)
#data = np.cumsum(vel_occ_array.sum(axis=1))
plt.bar(np.arange(len(data)) + 0.5, data, width=1.0)
plt.xticks(np.arange(0, 128+1, 16))
for i, x in enumerate([0, *np.linspace(1, 128, velocity_bins)]):
    plt.axvline(x=ceil(x), color='purple', linewidth=1.0, linestyle='-', alpha=0.5)
for i in pct_idx:
    plt.axvline(x=i+1, color='red', linewidth=1.0, linestyle='-', alpha=0.5)
plt.xlim(0, 128)
plt.title("Velocity Distribution with Percentile Lines")
plt.xlabel("Velocity\n(adjusted for channel volume and expression)")
plt.ylabel("Number of Notes")
plt.legend(['Linear Bins', 'Percentile Bins'])
leg = plt.gca().get_legend()
leg.legendHandles[0].set_color('purple')
leg.legendHandles[0].set_linewidth(1.0)
leg.legendHandles[0].set_alpha(1.0)
leg.legendHandles[1].set_color('red')
leg.legendHandles[1].set_linewidth(1.0)
leg.legendHandles[1].set_alpha(1.0)
plt.show()

print("Use these values for velocity_bins_override:")
print(pct_idx)

# %%
