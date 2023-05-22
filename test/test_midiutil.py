import mido
import os, sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import midi_util

cfg = midi_util.VocabConfig.from_json("vocab_config.json")
utils = midi_util.VocabUtils(cfg)

mid = mido.MidiFile("/mnt/e/datasets/music/midishrine-game-may-2023/files/Duck_Tales/the_moon.mid")
#mid = mido.MidiFile("jazz.mid")

out = midi_util.convert_midi_to_str(cfg, mid)
print(out)

mid_conv = midi_util.convert_str_to_midi(cfg, out)
mid_conv.save("test_out.mid")
