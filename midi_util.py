import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple

import mido


@dataclass
class VocabConfig:
    # Number of note events. Should be 128.
    note_events: int
    # Number of wait events. Configurable, must evenly divide max_wait_time.
    wait_events: int
    # Max wait time in milliseconds to be represented by a single token.
    max_wait_time: int
    # Number of velocity events. Should be 128 (or 100? need to check midi standard)
    velocity_events: int
    # Number of bins to quantize velocity into. Should evenly divide velocity_events.
    velocity_bins: int
    # Exponential scaling factor for velocity bin sizes. 1.0 = linear scaling.
    velocity_exp: float
    # Whether to sort tokens by instrument, note. This should improve data reducibility.
    do_token_sorting: bool
    # List of instrument names to use for binning. Must have at most 16 values.
    bin_instrument_names: List[str]
    # Indicates which bin name represents percussion instruments on MIDI channel 10.
    ch10_instrument_bin_name: str
    # Mapping from instrument name to bin name.
    program_name_to_bin_name: Dict[str, str]
    # Mapping from bin name to program name.
    bin_name_to_program_name: Dict[str, str]
    # Mapping from program number to instrument name.
    instrument_names: Dict[str, str]

    def __post_init__(self):
        self.validate()
        
        self._instrument_names_str_to_int = {name: int(i) for i, name in self.instrument_names.items()}
        self._instrument_names_int_to_str = {int(i): name for i, name in self.instrument_names.items()}
        
        self._bin_str_to_int = {name: int(i) for i, name in enumerate(self.bin_instrument_names)}

        self._bin_int_to_instrument_int = [self._instrument_names_str_to_int[self.bin_name_to_program_name[name]] if name != self.ch10_instrument_bin_name else 0 for name in self.bin_instrument_names]
        self._instrument_int_to_bin_int = [self._bin_str_to_int[self.program_name_to_bin_name[instr]] if self.program_name_to_bin_name[instr] != "" else -1 for instr in self.program_name_to_bin_name.keys()]

        self._ch10_bin_int = self._bin_str_to_int[self.ch10_instrument_bin_name] if self.ch10_instrument_bin_name else -1

        self.short_instr_bin_names = []
        for instr in self.bin_instrument_names:
            i = min(1, len(instr))
            while instr[:i] in self.short_instr_bin_names:
                i += 1
            self.short_instr_bin_names.append(instr[:i])
        self._short_instrument_names_str_to_int = {name: int(i) for i, name in enumerate(self.short_instr_bin_names)}

        range_excluding_ch10 = [(i if i < 9 else i+1) for i in range(len(self.bin_instrument_names))]
        bins_excluding_ch10 = [n for n in self.bin_instrument_names if n != self.ch10_instrument_bin_name]
        self.bin_channel_map = {bin: channel for channel, bin in zip(range_excluding_ch10, bins_excluding_ch10)}
        if self.ch10_instrument_bin_name:
            self.bin_channel_map[self.ch10_instrument_bin_name] = 9

    def validate(self):
        if self.max_wait_time % self.wait_events != 0:
            raise ValueError("max_wait_time must be exactly divisible by wait_events")
        if self.velocity_bins < 2:
            raise ValueError("velocity_bins must be at least 2")
        if len(self.bin_instrument_names) > 16:
            raise ValueError("bin_instruments must have at most 16 values")
        if self.ch10_instrument_bin_name and self.ch10_instrument_bin_name not in self.bin_instrument_names:
            raise ValueError("ch10_instrument_bin_name must be in bin_instruments")
        if self.velocity_exp <= 0:
            raise ValueError("velocity_exp must be greater than 0")

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        return cls(**config)


class VocabUtils:
    def __init__(self, cfg: VocabConfig) -> None:
        self.cfg = cfg
    
    @lru_cache(maxsize=128)
    def format_wait_token(self, wait: int) -> str:
        return f"t{wait}"

    @lru_cache(maxsize=128)
    def format_note_token(self, instrument_bin: int, note: int, velocity_bin: int) -> str:
        return f"{self.cfg.short_instr_bin_names[instrument_bin]}:{note:x}:{velocity_bin:x}"

    def velocity_to_bin(self, velocity: float) -> int:
        binsize = self.cfg.velocity_events / (self.cfg.velocity_bins - 1)
        if self.cfg.velocity_exp == 1.0:
            return ceil(velocity / binsize)
        else:
            return ceil((self.cfg.velocity_events*((self.cfg.velocity_exp**(velocity/self.cfg.velocity_events)-1)/(self.cfg.velocity_exp-1))) / binsize)

    def bin_to_velocity(self, bin: int) -> int:
        binsize = self.cfg.velocity_events / (self.cfg.velocity_bins - 1)
        if self.cfg.velocity_exp == 1.0:
            return max(0, ceil(bin * binsize - 1))
        else:
            return max(0, ceil(self.cfg.velocity_events*log(((self.cfg.velocity_exp-1)*binsize*bin)/self.cfg.velocity_events+1, self.cfg.velocity_exp) - 1))

    def delta_to_wait_ids(self, delta_ms: float) -> Iterator[int]:
        def roundi(f: float):
            return ceil(f - 0.5)

        max_wait_ms = self.cfg.max_wait_time
        div = max_wait_ms / self.cfg.wait_events

        #if delta_ms // max_wait_ms > 512:  # arbitrary limit to avoid excessive time_shifts
        #    raise ValueError("delta_time is too large")
        if delta_ms > max_wait_ms * 10:
            delta_ms = max_wait_ms * 10  # truncate time

        for _ in range(floor(delta_ms / max_wait_ms)):
            yield roundi(max_wait_ms / div)
        leftover_time_shift = roundi((delta_ms % max_wait_ms) / div)
        if leftover_time_shift > 0:
            yield leftover_time_shift

    def prog_data_to_token_data(self, program: int, channel: int, note: int, velocity: float) -> Optional[Tuple[int, int, int]]:
        if channel == 9:
            return self.cfg._ch10_bin_int, note, self.velocity_to_bin(velocity)
        
        instrument_bin = self.cfg._instrument_int_to_bin_int[program]
        if instrument_bin != -1:
            return instrument_bin, note, self.velocity_to_bin(velocity)
        return None

    def prog_data_list_to_token_data_list(self, data: List[Tuple[int, int, int, float]]) -> Iterator[Tuple[int, int, int]]:
        for d in data:
            token_data = self.prog_data_to_token_data(*d)
            if token_data is not None:
                yield token_data

    def sort_token_data(self, data: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        data.sort(key=lambda x: (x[0]!=self.cfg._ch10_bin_int, x[0], x[1], x[2]))
        return data

    def data_to_wait_tokens(self, delta_ms: float) -> List[str]:
        if delta_ms == 0.0:
            return []
        return [self.format_wait_token(i) for i in self.delta_to_wait_ids(delta_ms)]
    
    def wait_token_to_delta(self, token: str) -> float:
        return self.cfg.max_wait_time / self.cfg.wait_events * int(token[1:])
    
    def note_token_to_data(self, token: str) -> Tuple[int, int, int]:
        instr_str, note_str, velocity_str = token.strip().split(":")
        instr_bin = self.cfg._short_instrument_names_str_to_int[instr_str]
        note = int(note_str, base=16)
        velocity = self.bin_to_velocity(int(velocity_str, base=16))
        return instr_bin, note, velocity


@dataclass
class AugmentValues:
    instrument_bin_remap: Dict[int, int]
    velocity_mod_factor: float
    transpose_semitones: int
    time_stretch_factor: float

    @classmethod
    def default(cls) -> "AugmentValues":
        return cls(
            instrument_bin_remap={},
            velocity_mod_factor=1.0,
            transpose_semitones=0,
            time_stretch_factor=1.0,
        )


@dataclass
class AugmentConfig:
    # The number of times to augment each MIDI file. The dataset size will be multiplied by this number.
    augment_data_factor: float
    # A list of instrument names to randomly swap with each other.
    instrument_mixups: List[List[str]]
    # A list of percentages to change the note velocity by. 0.0 = no change.
    velocity_mod_pct: List[float]
    # A list of semitones to transpose by. 0 is not included by default; it must be specified manually.
    transpose_semitones: List[int]
    # A list of percentages to stretch the tempo by. 0.0 = no stretch.
    time_stretch_pct: List[float]
    # Random seed to use for reproducibility.
    seed: int

    cfg: VocabConfig

    def __post_init__(self):
        self.validate()
        if len(self.velocity_mod_pct) == 0:
            self.velocity_mod_pct = [0.0]
        if len(self.transpose_semitones) == 0:
            self.transpose_semitones = [0]
        if len(self.time_stretch_pct) == 0:
            self.time_stretch_pct = [0.0]
        
        self._instrument_mixups_int = [[self.cfg._bin_str_to_int[i] for i in l] for l in self.instrument_mixups]
        self._instrument_pool_assignments = {}
        self._mixup_pools = []
        for pool_i, mixup_list in enumerate(self._instrument_mixups_int):
            pool = set()
            for i in mixup_list:
                pool.add(i)
                self._instrument_pool_assignments[i] = pool_i
            self._mixup_pools.append(pool)


    def validate(self):
        used_instruments = set()
        for mixup_list in self.instrument_mixups:
            for n in mixup_list:
                if n in used_instruments:
                    raise ValueError(f"Duplicate instrument name: {n}")
                used_instruments.add(n)

    @classmethod
    def from_json(cls, path: str, cfg: VocabConfig):
        with open(path, "r") as f:
            config = json.load(f)
        config["cfg"] = cfg
        if "seed" not in config:
            config["seed"] = random.randint(0, 2**32 - 1)
        return cls(**config)
    
    def get_augment_values(self, filename: str) -> Iterator[AugmentValues]:
        rng = random.Random(self.seed + hash(filename))
        for _ in range(int(self.augment_data_factor)):
            # randomize order for each pool
            randomized_pools = [list(pool) for pool in self._mixup_pools]
            for pool in randomized_pools:
                rng.shuffle(pool)
            # distribute reassignments
            instrument_bin_remap = {}
            for i, pool in enumerate(randomized_pools):
                for j, instrument in enumerate(pool):
                    instrument_bin_remap[instrument] = randomized_pools[i - 1][j]
            yield AugmentValues(
                instrument_bin_remap=instrument_bin_remap,
                velocity_mod_factor=1.0 + rng.choice(self.velocity_mod_pct),
                transpose_semitones=rng.choice(self.transpose_semitones),
                time_stretch_factor=1.0 + rng.choice(self.time_stretch_pct),
            )


def mix_volume(velocity: int, volume: int, expression: int) -> float:
    return min(velocity * (volume / 127.0) * (expression / 127.0), 127)


def convert_midi_to_str(cfg: VocabConfig, mid: mido.MidiFile, augment: AugmentValues = None) -> str:
    utils = VocabUtils(cfg)
    if augment is None:
        augment = AugmentValues.default()

    # filter out unknown meta messages before merge (https://github.com/mido/mido/pull/286)
    for i in range(len(mid.tracks)):
        mid.tracks[i] = [msg for msg in mid.tracks[i] if msg.type != "unknown_meta"]

    if len(mid.tracks) > 1:
        mid.tracks = [mido.merge_tracks(mid.tracks)]

    delta_time_ms = 0.0
    tempo = 500000
    channel_program = {i: 0 for i in range(16)}
    channel_volume = {i: 127 for i in range(16)}
    channel_expression = {i: 127 for i in range(16)}  # unlikely to be useful. expression usually modifies an already played note.
    channel_notes = {i: {} for i in range(16)}
    pedal_on = False
    pedal_events = {}
    started_flag = False

    output = ["<start>"]
    token_data_buffer: List[Tuple[int, int, int, float]] = []  # need to sort notes between wait tokens

    def flush_token_data_buffer():
        nonlocal token_data_buffer, output, cfg, utils, augment
        token_data = utils.prog_data_list_to_token_data_list(token_data_buffer)
        if augment.instrument_bin_remap or augment.transpose_semitones:
            token_data = [(augment.instrument_bin_remap.get(i, i), n + augment.transpose_semitones, v) for i, n, v in token_data]
        if cfg.do_token_sorting:
            token_data = utils.sort_token_data(token_data)
        output += [utils.format_note_token(*t) for t in token_data]
        token_data_buffer = []

    def consume_note_program_data(prog: int, chan: int, note: int, vel: float):
        nonlocal output, started_flag, delta_time_ms, cfg, utils, token_data_buffer
        is_token_valid = utils.prog_data_to_token_data(prog, chan, note, vel) is not None
        if not is_token_valid:
            return
        if started_flag:
            wait_tokens = utils.data_to_wait_tokens(delta_time_ms)
            if len(wait_tokens) > 0:
                flush_token_data_buffer()
                output += wait_tokens
        delta_time_ms = 0.0
        token_data_buffer.append((prog, chan, note, vel * augment.velocity_mod_factor))
        started_flag = True

    for msg in mid.tracks[0]:
        time_ms = mido.tick2second(msg.time, mid.ticks_per_beat, tempo) * 1000.0
        delta_time_ms += time_ms
        t = msg.type

        if msg.is_meta:
            if t == "set_tempo":
                tempo = msg.tempo * augment.time_stretch_factor
            continue

        if t == "note_on" and msg.velocity == 0:
            t = "note_off"

        if t == "program_change":
            channel_program[msg.channel] = msg.program
        elif t == "note_on":
            consume_note_program_data(
                channel_program[msg.channel],
                msg.channel,
                msg.note,
                mix_volume(msg.velocity, channel_volume[msg.channel], channel_expression[msg.channel]),
            )
            channel_notes[msg.channel][msg.note] = True
        elif t == "note_off":
            if pedal_on:
                pedal_events[(msg.channel, msg.note)] = True
            else:
                consume_note_program_data(
                    channel_program[msg.channel],
                    msg.channel,
                    msg.note,
                    0,
                )
                if msg.note in channel_notes[msg.channel]:
                    del channel_notes[msg.channel][msg.note]
        elif t == "control_change":
            if msg.control == 7 or msg.control == 39:  # volume
                channel_volume[msg.channel] = msg.value
            elif msg.control == 11:  # expression
                channel_expression[msg.channel] = msg.value
            elif msg.control == 64:  # sustain pedal
                pedal_on = msg.value >= 64
                if not pedal_on:
                    for (channel, note) in pedal_events:
                        consume_note_program_data(
                            channel_program[channel],
                            channel,
                            note,
                            0,
                        )
                        if note in channel_notes[channel]:
                            del channel_notes[channel][note]
                    pedal_events = {}
            elif msg.control == 123:  # all notes off
                for channel in channel_notes.keys():
                    for note in channel_notes[channel]:
                        consume_note_program_data(
                            channel_program[channel],
                            channel,
                            note,
                            0,
                        )
                    channel_notes[channel] = {}
        else:
            pass

    flush_token_data_buffer()
    output.append("<end>")
    return " ".join(output)


def convert_str_to_midi(cfg: VocabConfig, data: str, meta_text: str = "Generated by MIDI-LLM-tokenizer") -> mido.MidiFile:
    utils = VocabUtils(cfg)
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = 500000
    if meta_text:
        track.append(mido.MetaMessage("text", text=meta_text, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    for bin_name, channel in cfg.bin_channel_map.items():
        if channel == 9:
            continue
        program = cfg._instrument_names_str_to_int[cfg.bin_name_to_program_name[bin_name]]
        track.append(mido.Message("program_change", program=program, time=0, channel=channel))
    track.append(mido.Message("program_change", program=0, time=0, channel=9))

    delta_ms = 0.0

    data = data.replace("<start>", "").replace("<end>", "").replace("<pad>", "").strip()
    for token in data.split(" "):
        if not token:
            continue
        token = token.strip() # just in case

        if token[0] == "t" and token[1].isdigit():  # wait token
            delta_ms += utils.wait_token_to_delta(token)
        else:  # note token
            bin, note, velocity = utils.note_token_to_data(token)
            channel = cfg.bin_channel_map[cfg.bin_instrument_names[bin]]
            ticks = int(mido.second2tick(delta_ms / 1000.0, mid.ticks_per_beat, tempo))
            delta_ms = 0.0
            if velocity > 0:
                track.append(mido.Message("note_on", note=note, velocity=velocity, time=ticks, channel=channel))
            else:
                track.append(mido.Message("note_off", note=note, velocity=0, time=ticks, channel=channel))
    track.append(mido.MetaMessage("end_of_track", time=0))

    return mid
