import json
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict
from math import ceil, floor


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_SCALE_LEN = len(NOTE_NAMES)


@dataclass
class VocabConfig:
    program_to_bin: List[int]
    bin_to_program: List[int]
    bin_instruments: List[str]
    note_events: int
    start_note: int
    start_octave: int
    wait_events: int
    max_wait_time: int
    velocity_events: int
    velocity_bins: int

    def __post_init__(self):
        self.validate()
        self.short_instrument_names = []
        for instr in self.bin_instruments:
            i = min(3, len(instr))
            while instr[:i] in self.short_instrument_names:
                i += 1
            self.short_instrument_names.append(instr[:i])

    def validate(self):
        if self.velocity_events % self.velocity_bins != 0:
            raise ValueError("velocity_max must be exactly divisible by velocity_bins")
        if self.max_wait_time % self.wait_events != 0:
            raise ValueError("max_wait_time must be exactly divisible by wait_events")
        if len(self.bin_instruments) > 16:
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
    def format_note_token(self, instrument_bin: int, velocity_bin: int, note: int) -> str:
        return f"{self.cfg.short_instrument_names[instrument_bin]}{velocity_bin}{NOTE_NAMES[(note+self.cfg.start_note)%NOTE_SCALE_LEN]}{note//NOTE_SCALE_LEN+self.cfg.start_octave} "

    @lru_cache(maxsize=128)
    def format_wait_token(self, wait: int) -> str:
        return f"T{wait} "

    def velocity_to_bin(self, velocity: float) -> int:
        return ceil(velocity / self.cfg.velocity_bins)

    def bin_to_velocity(self, bin: int) -> int:
        return bin * self.cfg.velocity_bins

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

    def data_to_note_token(self, program: int, velocity: int, note: int) -> str:
        return self.format_note_token(self.cfg.program_to_bin[program], self.velocity_to_bin(velocity), note)
    
    def data_to_wait_tokens(self, delta_ms: float) -> str:
        return "".join([self.format_wait_token(i) for i in self.delta_to_wait_ids(delta_ms)])
