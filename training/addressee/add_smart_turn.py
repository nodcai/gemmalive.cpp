"""Adds Smart Turn clips, already extracted for the turn head, as speech
meant for the assistant: users talking to a voice agent, complete or not.

    python add_smart_turn.py --features features --out addressee-features --train 10000 --val 1500
"""

import argparse
import glob
import os
import struct

import numpy as np


def sample(features, prefix, n, seed):
    files = sorted(glob.glob(os.path.join(features, f"{prefix}-*.f16")))
    rng = np.random.default_rng(seed)
    per_file = int(np.ceil(n / len(files)))
    rows, header = [], None
    for path in files:
        with open(path, "rb") as f:
            header = f.read(8)
        n_layer, n_embd = struct.unpack("<ii", header)
        x = np.memmap(path, dtype=np.float16, mode="r", offset=8).reshape(-1, n_layer, n_embd)
        pick = np.sort(rng.choice(len(x), min(per_file, len(x)), replace=False))
        # By name: prepare.py's and make_pauses.py's TSVs order the columns differently.
        with open(path[:-4] + ".tsv") as f:
            cols = f.readline().rstrip("\n").split("\t")
            meta = [dict(zip(cols, line.rstrip("\n").split("\t"))) for line in f]
        rows.extend((np.asarray(x[i]), meta[i]) for i in pick)
    rng.shuffle(rows)
    return header, rows[:n]


def write(out, name, header, rows):
    with open(os.path.join(out, name + ".f16"), "wb") as f:
        f.write(header)
        for x, _ in rows:
            f.write(x.astype(np.float16).tobytes())
    with open(os.path.join(out, name + ".tsv"), "w") as f:
        f.write("id\taddressed\tkind\ttext\tvoice\tcut\tduration\n")
        for _, m in rows:
            f.write(f"{m['id']}\tTrue\tsmart_turn\t\t{m.get('voice', '')}\tFalse\t{m['duration']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="turn head features (train-*.f16, test-*.f16)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--val", type=int, default=1500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    header, train = sample(args.features, "train", args.train, 1)
    write(args.out, "train-st", header, train)
    header, val = sample(args.features, "test", args.val, 2)
    write(args.out, "val-st", header, val)
    print(f"added {len(train)} train and {len(val)} val Smart Turn clips")


if __name__ == "__main__":
    main()
