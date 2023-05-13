import json
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict


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
        self.short_instrument_names = []
        for instr in self.bin_instruments:
            i = min(3, len(instr))
            while instr[:i] in self.short_instrument_names:
                i += 1
            self.short_instrument_names.append(instr[:i])

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

    def format_wait_token(self, wait: int) -> str:
        return f"T{wait} "

    def velocity_to_bin(self, velocity: int) -> int:
        return velocity // self.cfg.velocity_bins

    def bin_to_velocity(self, bin: int) -> int:
        return bin * self.cfg.velocity_bins

    @lru_cache(maxsize=128)
    def delta_to_wait_tokens(self, delta_ms: int) -> List[int]:
        time_shifts = []

        def roundi(f: float):
            return int(f + 0.5)

        max_wait_ms = self.cfg.max_wait_time
        div = self.cfg.wait_events

        if delta_ms // max_wait_ms > 512:  # arbitrary limit to avoid excessive time_shifts
            raise ValueError("delta_time is too large")

        for _ in range(delta_ms // max_wait_ms):
            time_shifts.append(roundi(max_wait_ms / div))
        leftover_time_shift = roundi((delta_ms % max_wait_ms) / div)
        time_shifts.append(leftover_time_shift) if leftover_time_shift > 0 else None

        return time_shifts
