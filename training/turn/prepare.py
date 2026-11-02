"""Streams Smart Turn clips to gemmalive-turn-features.

Each clip gets a short tail of near-silence appended, matching what the live
session commits after the user pauses. Writes the clips as
[uint32 n][float32 x n] records to stdout and their labels to a TSV.

With --long, finished clips get a long tail instead, as when the session asks
the head again later in a pause, and unfinished ones only now and then
(--long-unfinished): a pause that goes on makes a finished turn likelier,
as it does in conversation.

    python prepare.py data/smart-turn-v3.2-test/data/train-00000-of-00010.parquet \
        --meta out/test-00000.tsv | gemmalive-turn-features --in - --out out/test-00000.f16
"""

import argparse
import io
import struct
import sys
import zlib

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

RATE = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--meta", required=True, help="TSV with one row per clip, in output order")
    ap.add_argument("--tail-min", type=float, default=0.15, help="appended silence, seconds")
    ap.add_argument("--tail-max", type=float, default=0.40)
    ap.add_argument("--noise-db", type=float, default=-70.0, help="level of the appended tail")
    ap.add_argument("--long", action="store_true", help="long tails, see above")
    ap.add_argument("--long-min", type=float, default=0.5)
    ap.add_argument("--long-max", type=float, default=1.2)
    ap.add_argument("--long-unfinished", type=float, default=0.3, help="share of unfinished clips with a long tail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many clips")
    args = ap.parse_args()

    rng = np.random.default_rng([args.seed, zlib.crc32(args.parquet.encode())])
    out = sys.stdout.buffer
    cols = ["id", "language", "endpoint_bool", "midfiller", "endfiller", "synthetic", "dataset"]

    pf = pq.ParquetFile(args.parquet)
    n = 0
    with open(args.meta, "w") as meta:
        meta.write("\t".join(cols + ["duration", "tail"]) + "\n")
        for batch in pf.iter_batches(batch_size=256, columns=["audio"] + cols):
            rows = batch.to_pydict()
            for i, audio in enumerate(rows["audio"]):
                x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if sr != RATE:
                    raise SystemExit(f"unexpected sample rate {sr}")

                tail = rng.uniform(args.tail_min, args.tail_max)
                if args.long and (rows["endpoint_bool"][i] or rng.random() < args.long_unfinished):
                    tail = rng.uniform(args.long_min, args.long_max)
                noise = rng.standard_normal(int(tail * RATE)).astype(np.float32)
                noise *= 10 ** (args.noise_db / 20)
                x = np.concatenate([x, noise])

                out.write(struct.pack("<I", len(x)))
                out.write(x.astype("<f4").tobytes())

                values = [str(rows[c][i]) for c in cols]
                meta.write("\t".join(values + [f"{len(x) / RATE:.3f}", f"{tail:.3f}"]) + "\n")
                n += 1
                if args.limit and n >= args.limit:
                    out.flush()
                    return
    out.flush()


if __name__ == "__main__":
    main()
