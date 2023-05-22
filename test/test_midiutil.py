import mido
import midiutil

cfg = midiutil.VocabConfig.from_json("vocab_config.json")
utils = midiutil.VocabUtils(cfg)

mid = mido.MidiFile("/mnt/e/datasets/music/midishrine-game-may-2023/files/Duck_Tales/the_moon.mid")
#mid = mido.MidiFile("jazz.mid")

out = midiutil.convert_midi_to_str(cfg, mid)

mid_conv = midiutil.convert_str_to_midi(cfg, out)
