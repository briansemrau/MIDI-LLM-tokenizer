import json
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor
from typing import Dict, List, Tuple, Union

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
        # make sure ch10 instrument is in 10th slot
        if self.ch10_instrument_bin_name != self.bin_instrument_names[9]:
            self.bin_instrument_names[9], self.bin_instrument_names[self.bin_instrument_names.index(self.ch10_instrument_bin_name)] = self.ch10_instrument_bin_name, self.bin_instrument_names[9]
        
        self._instrument_names_str_to_int = {name: int(i) for i, name in self.instrument_names.items()}
        self._instrument_names_int_to_str = {int(i): name for i, name in self.instrument_names.items()}
        
        self._bin_int_to_instrument_int = [self._instrument_names_str_to_int[self.bin_name_to_program_name[name]] if name != self.ch10_instrument_bin_name else 0 for name in self.bin_instrument_names]
        self._instrument_int_to_bin_int = [self.bin_instrument_names.index(self.program_name_to_bin_name[instr]) if self.program_name_to_bin_name[instr] != "" else -1 for instr in self.program_name_to_bin_name.keys()]

        self.short_instr_bin_names = []
        for instr in self.bin_instrument_names:
            i = min(1, len(instr))

            while instr[:i] in self.short_instr_bin_names:
                i += 1
            self.short_instr_bin_names.append(instr[:i])
        self._short_instrument_names_str_to_int = {name: int(i) for i, name in enumerate(self.short_instr_bin_names)}

    def validate(self):
        if self.max_wait_time % self.wait_events != 0:
            raise ValueError("max_wait_time must be exactly divisible by wait_events")
        if self.velocity_bins < 2:
            raise ValueError("velocity_bins must be at least 2")
        if len(self.bin_instrument_names) > 16:
            raise ValueError("bin_instruments must have at most 16 values")

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
        return ceil(velocity / binsize)

    def bin_to_velocity(self, bin: int) -> int:
        binsize = self.cfg.velocity_events / (self.cfg.velocity_bins - 1)
        return max(0, ceil(bin * binsize - 1))

    def delta_to_wait_ids(self, delta_ms: float) -> List[int]:
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

    def data_to_note_token(self, program: int, channel: int, note: int, velocity: int) -> str:
        if channel == 9:
            return self.format_note_token(9, note, self.velocity_to_bin(velocity))
        
        instrument_bin = self.cfg._instrument_int_to_bin_int[program]
        if instrument_bin != -1:
            return self.format_note_token(instrument_bin, note, self.velocity_to_bin(velocity))
        return ""

    def data_to_wait_tokens(self, delta_ms: float) -> List[str]:
        for i in self.delta_to_wait_ids(delta_ms):
            yield self.format_wait_token(i)
    
    def wait_token_to_delta(self, token: str) -> float:
        return self.cfg.max_wait_time / self.cfg.wait_events * int(token[1:])
    
    def note_token_to_data(self, token: str) -> Tuple[int, int, int]:
        instr_str, note_str, velocity_str = token.strip().split(":")
        instr_bin = self.cfg._short_instrument_names_str_to_int[instr_str]
        note = int(note_str, base=16)
        velocity = self.bin_to_velocity(int(velocity_str, base=16))
        return instr_bin, note, velocity


def mix_volume(velocity: int, volume: int, expression: int) -> float:
    return min(velocity * (volume / 127.0) * (expression / 127.0), 127)


def convert_midi_to_str(cfg: VocabConfig, mid: mido.MidiFile) -> str:
    utils = VocabUtils(cfg)

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

    def add_to_output(t: Union[str, List[str]]):
        nonlocal output, started_flag, delta_time_ms
        if t:
            if started_flag:
                output += utils.data_to_wait_tokens(delta_time_ms)
            delta_time_ms = 0.0
            if isinstance(t, str):
                output.append(t)
            else:
                output += t
            started_flag = True

    for msg in mid.tracks[0]:
        time_ms = mido.tick2second(msg.time, mid.ticks_per_beat, tempo) * 1000.0
        delta_time_ms += time_ms
        t = msg.type

        if msg.is_meta:
            if t == "set_tempo":
                tempo = msg.tempo
            continue

        if t == "note_on" and msg.velocity == 0:
            t = "note_off"

        if t == "program_change":
            channel_program[msg.channel] = msg.program
        elif t == "note_on":
            token = utils.data_to_note_token(
                channel_program[msg.channel],
                msg.channel,
                msg.note,
                mix_volume(msg.velocity, channel_volume[msg.channel], channel_expression[msg.channel]),
            )
            add_to_output(token)
            channel_notes[msg.channel][msg.note] = True
        elif t == "note_off":
            if pedal_on:
                pedal_events[(msg.channel, msg.note)] = True
            else:
                token = utils.data_to_note_token(
                    channel_program[msg.channel],
                    msg.channel,
                    msg.note,
                    0,
                )
                add_to_output(token)
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
                    tokens = []
                    for (channel, note) in pedal_events:
                        tokens.append(utils.data_to_note_token(
                            channel_program[channel],
                            channel,
                            note,
                            0,
                        ))
                        if note in channel_notes[channel]:
                            del channel_notes[channel][note]
                    pedal_events = {}
                    add_to_output(tokens)
            elif msg.control == 123:  # all notes off
                for channel in channel_notes.keys():
                    tokens = []
                    for note in channel_notes[channel]:
                        tokens.append(utils.data_to_note_token(
                            channel_program[channel],
                            channel,
                            note,
                            0,
                        ))
                    channel_notes[channel] = {}
                    add_to_output(tokens)
        else:
            pass
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
    for channel, bin_name in enumerate(cfg.bin_instrument_names):
        if bin_name == cfg.ch10_instrument_bin_name:
            #assert channel == 9
            track.append(mido.Message("program_change", program=0, time=0, channel=channel))
            continue
        program = cfg._instrument_names_str_to_int[cfg.bin_name_to_program_name[bin_name]]
        track.append(mido.Message("program_change", program=program, time=0, channel=channel))

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
            channel = bin
            ticks = int(mido.second2tick(delta_ms / 1000.0, mid.ticks_per_beat, tempo))
            delta_ms = 0.0
            if velocity > 0:
                track.append(mido.Message("note_on", note=note, velocity=velocity, time=ticks, channel=channel))
            else:
                track.append(mido.Message("note_off", note=note, velocity=0, time=ticks, channel=channel))
    track.append(mido.MetaMessage("end_of_track", time=0))

    return mid
