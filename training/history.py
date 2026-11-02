"""Generates earlier user turns for conversations that clips follow in
gemmalive-turn-features (--history): spoken questions and requests, as the
addressee head's requests (Piper voices, texts.QUESTIONS, texts.COMMANDS and
OASST2 prompts), the val split from its held-out voices and texts.

    python training/history.py --split train --n 2000 --out data/history-train.pcm
"""

import argparse
import os
import random
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "addressee"))
import texts  # noqa: E402
from make_data import RATE, Voices, level, noise, split_list  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voices", default="data/piper")
    ap.add_argument("--prompts", default="data/oasst-prompts.txt", help="more requests, one per line, if present")
    args = ap.parse_args()

    rng = random.Random(args.seed * 1000 + 500 + (args.split == "val"))
    voices = Voices(args.voices, args.split)
    requests = split_list(texts.QUESTIONS, args.split, salt=1) + split_list(texts.COMMANDS, args.split, salt=2)
    if os.path.exists(args.prompts):
        with open(args.prompts) as f:
            requests += split_list([line.strip() for line in f if line.strip()], args.split, salt=10)

    with open(args.out, "wb") as out, open(os.path.splitext(args.out)[0] + ".tsv", "w") as meta:
        meta.write("id\ttext\tvoice\tduration\n")
        for i in range(args.n):
            text = rng.choice(requests)
            speech, voice = voices.say(rng, text)
            speech = level(speech, rng.uniform(-28, -16))
            clip = np.concatenate([noise(rng, rng.uniform(0.1, 0.3)), speech, noise(rng, rng.uniform(0.2, 0.5))])
            out.write(struct.pack("<I", len(clip)))
            out.write(clip.astype("<f4").tobytes())
            meta.write(f"{i}\t{' '.join(text.split())}\t{voice}\t{len(clip) / RATE:.3f}\n")
    print(f"{args.n} turns in {args.out}")


if __name__ == "__main__":
    main()
