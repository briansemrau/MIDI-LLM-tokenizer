from transformers import PreTrainedTokenizerFast
import json

tokenizer = PreTrainedTokenizerFast.from_pretrained("./tokenizer-midi")

#input = "<start> p:3c:f t10 p:3e:f t10 p:40:f t64 p:3c:0 p:3e:0 p:40:0 <end>"
with open("/mnt/e/datasets/music/lakh_midi_v0.1/lmd_full.jsonl", "r") as f:
    input = json.loads(f.readline())["text"]
print(input[:100])
tokens = tokenizer.encode(input)
output = tokenizer.decode(tokens)
print(output[:100])
assert input == output
