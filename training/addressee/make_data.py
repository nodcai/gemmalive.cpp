"""Generates clips for the addressee head and streams them to
gemmalive-turn-features, like training/turn/prepare.py.

Speech meant for the assistant: questions and requests (texts.QUESTIONS,
texts.COMMANDS and, if present, OASST2 prompts from oasst_prompts.py) and
interruptions (texts.INTERRUPTIONS).
Speech that is not: side talk (a vocative plus a remark, or a phone call),
backchannels, and background speech from LibriSpeech, quieter than the user.
Texts are spoken by Piper voices on both sides, so the voice tells nothing.
Only voices whose training data allows any use are taken (VOICES).

Like the live session, a clip ends with a short pause (the commit), or for
some longer clips is cut mid-speech (the check after 1.5 s of speech).

The val split holds out voices, speakers, texts and LibriSpeech speakers.

    python make_data.py --split train --n 3000 --seed 1 --meta out/train-00.tsv \
        | gemmalive-turn-features --in - --out out/train-00.f16
"""

import argparse
import glob
import io
import json
import os
import random
import struct
import sys

import numpy as np
import soundfile as sf
import soxr
import onnxruntime
from piper import PiperConfig, PiperVoice, SynthesisConfig

sys.path.insert(0, os.path.dirname(__file__))
import texts  # noqa: E402

RATE = 16000

# Piper voices trained on data that allows any use, including commercial:
# public domain, CC0, CC BY 4.0, Apache 2.0, CMU ARCTIC and the Jenny licence.
# Left out: non-commercial (semaine, hfc_*, ryan, l2arctic, lessac),
# share-alike (northern_english_male) and unlicensed or unclear (alan, amy,
# kusal).
VOICES = {
    "en_GB-alba-medium", "en_GB-aru-medium", "en_GB-cori-medium", "en_GB-jenny_dioco-medium",
    "en_GB-vctk-medium", "en_US-arctic-medium", "en_US-bryce-medium", "en_US-joe-medium",
    "en_US-john-medium", "en_US-kristin-medium", "en_US-libritts_r-medium", "en_US-ljspeech-medium",
    "en_US-mike-medium", "en_US-norman-medium", "en_US-reza_ibrahim-medium", "en_US-sam-medium",
}

# Single-speaker voices held out for val.
VAL_VOICES = {"en_GB-alba-medium", "en_US-mike-medium", "en_US-sam-medium"}

# Share of each kind in the mix; label 1 = meant for the assistant.
KINDS = [
    ("request", 1, 0.30),
    ("interruption", 1, 0.12),
    ("topic_shift", 1, 0.0),
    ("side_talk", 0, 0.28),
    ("marked_side_talk", 0, 0.0),
    ("backchannel", 0, 0.12),
    ("background", 0, 0.18),
]


def split_list(items, split, val_share=0.15, salt=0):
    """Deterministic train/val split of a list."""
    rng = random.Random(1234 + salt)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    k = max(1, int(len(items) * val_share))
    keep = idx[:k] if split == "val" else idx[k:]
    return [items[i] for i in sorted(keep)]


# Threads per Piper voice. onnxruntime's default is one per core, for each of
# the 16 voices of each worker; on a 128-core box that is thousands of threads
# per worker, past a container's pid limit. The workers are the parallelism.
ORT_THREADS = 4


def load_voice(path):
    """PiperVoice.load with a small thread pool."""
    with open(f"{path}.json", encoding="utf-8") as f:
        config = PiperConfig.from_dict(json.load(f))
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = ORT_THREADS
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    return PiperVoice(config=config, session=session)


class Voices:
    def __init__(self, voice_dir, split):
        self.voices = []
        for path in sorted(glob.glob(os.path.join(voice_dir, "*.onnx"))):
            name = os.path.basename(path)[:-5]
            if name not in VOICES:
                continue
            voice = load_voice(path)
            n = voice.config.num_speakers
            if n > 1:
                # Multi-speaker models: the last 15 % of speakers are val.
                cut = int(n * 0.85)
                speakers = list(range(cut, n)) if split == "val" else list(range(cut))
            else:
                if (name in VAL_VOICES) != (split == "val"):
                    continue
                speakers = [None]
            self.voices.append((name, voice, speakers))
        if not self.voices:
            raise SystemExit(f"no Piper voices for {split} in {voice_dir}")

    def say(self, rng, text):
        name, voice, speakers = rng.choice(self.voices)
        speaker = rng.choice(speakers)
        cfg = SynthesisConfig(speaker_id=speaker, length_scale=rng.uniform(0.85, 1.2),
                              noise_scale=rng.uniform(0.5, 0.8), noise_w_scale=rng.uniform(0.6, 1.0))
        audio = np.concatenate([c.audio_float_array for c in voice.synthesize(text, syn_config=cfg)])
        audio = soxr.resample(audio, voice.config.sample_rate, RATE).astype(np.float32)
        return audio, f"{name}:{speaker}"


def librispeech(root, split):
    sub = "test-clean" if split == "val" else "dev-clean"
    files = sorted(glob.glob(os.path.join(root, "LibriSpeech", sub, "*", "*", "*.flac")))
    if not files:
        raise SystemExit(f"no LibriSpeech {sub} under {root}")
    return files


def level(x, db):
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    return (x * (10 ** (db / 20) / rms)).astype(np.float32)


def noise(rng, seconds, db=-70.0):
    n = int(seconds * RATE)
    return (np.random.default_rng(rng.getrandbits(32)).standard_normal(n) * 10 ** (db / 20)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--voices", default="data/piper")
    ap.add_argument("--librispeech", default="data/librispeech")
    ap.add_argument("--prompts", default="data/oasst-prompts.txt", help="more requests, one per line, if present")
    ap.add_argument("--weights", default="", help="kind=weight,... overriding the default mix")
    args = ap.parse_args()

    weights = {k: w for k, _, w in KINDS}
    for item in filter(None, args.weights.split(",")):
        kind, w = item.split("=")
        if kind not in weights:
            raise SystemExit(f"unknown kind {kind}")
        weights[kind] = float(w)

    rng = random.Random(args.seed * 1000 + (args.split == "val"))
    voices = Voices(args.voices, args.split)
    questions = split_list(texts.QUESTIONS, args.split, salt=1) + split_list(texts.COMMANDS, args.split, salt=2)
    if os.path.exists(args.prompts):
        with open(args.prompts) as f:
            questions += split_list([line.strip() for line in f if line.strip()], args.split, salt=10)
    interruptions = split_list(texts.INTERRUPTIONS, args.split, salt=3)
    vocatives = split_list(texts.VOCATIVES, args.split, salt=4)
    remarks = split_list(texts.SIDE_REMARKS, args.split, salt=5)
    others = split_list(texts.OTHER_PARTY, args.split, salt=6)
    backchannels = split_list(texts.BACKCHANNELS, args.split, salt=7)
    markers = split_list(texts.MARKERS, args.split, salt=8)
    statements = split_list(texts.STATEMENTS, args.split, salt=9)
    books = librispeech(args.librispeech, args.split)

    out = sys.stdout.buffer
    with open(args.meta, "w") as meta:
        meta.write("id\taddressed\tkind\ttext\tvoice\tcut\tduration\n")
        for i in range(args.n):
            kind, label, _ = rng.choices(KINDS, weights=[weights[k] for k, _, _ in KINDS])[0]
            db = rng.uniform(-28, -16)  # the user, near the microphone
            if kind == "request":
                text = rng.choice(questions)
                if rng.random() < 0.15:
                    text = rng.choice(["Um, ", "So, ", "Hey, ", "Okay, "]) + text[0].lower() + text[1:]
            elif kind == "interruption":
                text = rng.choice(interruptions)
            elif kind == "side_talk":
                text = (f"{rng.choice(vocatives)}, {rng.choice(remarks)}" if rng.random() < 0.8
                        else rng.choice(others))
            elif kind == "topic_shift":
                rest = rng.choice(statements) if rng.random() < 0.3 else rng.choice(questions)
                text = f"{rng.choice(markers)} {rest[0].lower()}{rest[1:]}"
            elif kind == "marked_side_talk":
                text = f"{rng.choice(markers)} {rng.choice(vocatives)}, {rng.choice(remarks)}"
            elif kind == "backchannel":
                text = rng.choice(backchannels)
                db = rng.uniform(-32, -18)
            if kind == "background":
                path = rng.choice(books)
                speech, sr = sf.read(path, dtype="float32")
                start = rng.randint(0, max(0, len(speech) - int(6 * sr)))
                speech = speech[start : start + int(rng.uniform(1.5, 6.0) * sr)]
                text = os.path.basename(path)[:-5]
                voice = "librispeech"
                db = rng.uniform(-42, -28)  # a TV or people across the room
            else:
                speech, voice = voices.say(rng, text)
            speech = level(speech, db)

            # Ends like a live commit: a short pause, or a cut mid-speech.
            cut = len(speech) > 1.6 * RATE and rng.random() < (0.6 if kind == "background" else 0.3)
            if cut:
                speech = speech[: rng.randint(int(1.5 * RATE), len(speech))]
                tail = np.zeros(0, np.float32)
            else:
                tail = noise(rng, rng.uniform(0.15, 0.4))
            clip = np.concatenate([noise(rng, rng.uniform(0.1, 0.4)), speech, tail])

            out.write(struct.pack("<I", len(clip)))
            out.write(clip.astype("<f4").tobytes())
            text = " ".join(text.split())  # no tabs or newlines in the TSV
            meta.write(f"{args.split}-{args.seed}-{i}\t{bool(label)}\t{kind}\t{text}\t{voice}\t{cut}\t"
                       f"{len(clip) / RATE:.3f}\n")
    out.flush()


if __name__ == "__main__":
    main()
