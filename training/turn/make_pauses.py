"""Synthetic utterances with a pause inside, for the turn head.

Users often open with a marker, pause, and only then ask ("Oh, before I
forget, ... how often should I water this?"). Smart Turn rarely has that, and
the head took such finished questions for unfinished ones. Each clip is a
marker, a pause and a question, request or statement (complete), or stops
after the pause or partway through the rest (incomplete). Clips end with the
same short pause as training/turn/prepare.py and are written as records for
gemmalive-turn-features, with labels in a TSV like Smart Turn's.

    python make_pauses.py --split train --n 4000 --seed 1 --out data/turn-pauses/train-01
"""

import argparse
import os
import random
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "addressee"))
import texts  # noqa: E402
from make_data import RATE, Voices, level, noise, split_list  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="path prefix for .pcm and .tsv")
    ap.add_argument("--voices", default="data/piper")
    args = ap.parse_args()

    rng = random.Random(args.seed * 1000 + 7 + (args.split == "val"))
    voices = Voices(args.voices, args.split)
    rests = (split_list(texts.QUESTIONS, args.split, salt=1) + split_list(texts.COMMANDS, args.split, salt=2)
             + split_list(texts.STATEMENTS, args.split, salt=9))
    markers = split_list(texts.MARKERS, args.split, salt=8)

    with open(args.out + ".pcm", "wb") as out, open(args.out + ".tsv", "w") as meta:
        meta.write("id\tlanguage\tendpoint_bool\tkind\ttext\tvoice\tduration\n")
        for i in range(args.n):
            marker = rng.choice(markers).rstrip(",")
            rest = rng.choice(rests)
            # One speaker for the whole utterance.
            seed = rng.getrandbits(32)
            head, voice = voices.say(random.Random(seed), marker + ",")
            body, _ = voices.say(random.Random(seed), rest[0].upper() + rest[1:])
            db = rng.uniform(-28, -16)
            head, body = level(head, db), level(body, db)
            pause = noise(rng, rng.uniform(0.3, 1.0))

            r = rng.random()
            if r < 0.5:
                kind, complete, speech = "complete", True, [head, pause, body]
            elif r < 0.75:
                kind, complete, speech = "after_marker", False, [head]
            else:
                cut = int(len(body) * rng.uniform(0.3, 0.75))
                kind, complete, speech = "mid_rest", False, [head, pause, body[:cut]]

            clip = np.concatenate([noise(rng, rng.uniform(0.1, 0.4))] + speech + [noise(rng, rng.uniform(0.15, 0.4))])
            out.write(struct.pack("<I", len(clip)))
            out.write(clip.astype("<f4").tobytes())
            text = " ".join(f"{marker}, ... {rest}".split())
            meta.write(f"{args.split}-{args.seed}-{i}\tsynthetic\t{complete}\t{kind}\t{text}\t{voice}\t"
                       f"{len(clip) / RATE:.3f}\n")


if __name__ == "__main__":
    main()
