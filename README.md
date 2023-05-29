# MIDI-LLM-tokenizer
Tools for converting .mid files into text for training large language models.
The exact format of the intermediary text is not critical. The primary goals are to 1. create a "legible" text format that can be easily used with existing LLM tech, and 2. encode midi data into an efficient format for training.

Expected workflow would be:
1. [convert midi datasets into jsonl files](#midi_to_jsonlpy)
2. [tokenize](#build_tokenizerpy) jsonl files into binidx
3. train LLM
4. sample LLM
5. [convert text into midi](#str_to_midipy)

For converting jsonl files to binidx format for training, see https://github.com/Abel2076/json2binidx_tool or https://github.com/EleutherAI/gpt-neox

# Vocabulary

MIDI files contain a lot of data, and only some of it can be reasonably learned by a language model.
Inspired by OpenAI MuseNet and Oore et. al, 2018, we have two main types of tokens:
- Wait tokens for timing (125 of them, representing real time)
- Combined note+velocity+instrument tokens (128 notes * 16 quantized velocity * 16 binned instruments = 32768 tokens)
- pad/start/end tokens.

Notes and quantized velocities are encoded as hex, while instruments are encoded as the shortest unique string.

We (knowingly) discard the following information:
- Panning
- Pitch bend
- Modulation
- Key signature
- Time signature
- Track names
- Instrument names (we assume the standard GM instruments)

Simultaneous tokens (e.g. chords) are sorted by instrument, note, then MIDI event order. This reduces unnecessary randomness in the data.

In the future, instrument order could be uniformly randomized to allow constrained sampling where you provide a preexisting track and the model generates a melody.

# Scripts

## build_tokenizer.py
Builds a new huggingface tokenizer and vocab using vocab_config.json

```sh
python ./build_tokenizer.py
```

## midi_to_jsonl.py
Converts a directory or archive of mid/midi files into a jsonl file of note sequences.

```sh
python ./midi_to_jsonl.py --path ~/lmd_full.tar.gz --output ~/lmd_full.jsonl --workers 4
```

## midi/text conversion

```sh
python ./midi_to_str.py test.mid
```

```sh
python ./str_to_midi.py "<start> p:3c:f t10 p:3e:f t10 p:40:f t64 p:3c:0 p:3e:0 p:40:0 <end>" --output test.mid
```
