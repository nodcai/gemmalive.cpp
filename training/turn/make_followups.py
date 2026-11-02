"""Short follow-ups and closings, for the turn head.

After an answer, users ask on in a few words ("And on top of a mountain?",
"Why is that?") or close ("Okay, thanks."). Smart Turn has few such turns,
and the head took them for unfinished ones, waiting for the longest pause
before answering. Each clip is a follow-up or closing, whole (complete) or
stopped partway (incomplete), with the same short pause at the end as
training/turn/prepare.py, written as records for gemmalive-turn-features,
with labels in a TSV like Smart Turn's.

    python make_followups.py --split train --n 3000 --seed 1 --out data/turn-followups/train-01
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
    ap.add_argument("--long", action="store_true", help="long tails, as training/turn/prepare.py --long")
    args = ap.parse_args()

    rng = random.Random(args.seed * 1000 + 11 + (args.split == "val"))
    voices = Voices(args.voices, args.split)
    follow_ups = split_list(texts.FOLLOW_UPS, args.split, salt=10)
    closings = split_list(texts.CLOSINGS, args.split, salt=11)

    with open(args.out + ".pcm", "wb") as out, open(args.out + ".tsv", "w") as meta:
        meta.write("id\tlanguage\tendpoint_bool\tkind\ttext\tvoice\tduration\n")
        for i in range(args.n):
            closing = rng.random() < 0.25
            text = rng.choice(closings if closing else follow_ups)
            speech, voice = voices.say(random.Random(rng.getrandbits(32)), text[0].upper() + text[1:])
            speech = level(speech, rng.uniform(-28, -16))

            # Half are cut partway: into the last word or before it.
            if rng.random() < 0.5:
                complete, kind = True, "closing" if closing else "follow_up"
            else:
                complete, kind = False, "mid_closing" if closing else "mid_follow_up"
                speech = speech[:int(len(speech) * rng.uniform(0.35, 0.8))]

            tail = rng.uniform(0.15, 0.4)
            if args.long and (complete or rng.random() < 0.3):
                tail = rng.uniform(0.5, 1.2)
            clip = np.concatenate([noise(rng, rng.uniform(0.1, 0.4)), speech, noise(rng, tail)])
            out.write(struct.pack("<I", len(clip)))
            out.write(clip.astype("<f4").tobytes())
            meta.write(f"{args.split}-{args.seed}-{i}\tsynthetic\t{complete}\t{kind}\t{text}\t{voice}\t"
                       f"{len(clip) / RATE:.3f}\n")


if __name__ == "__main__":
    main()
