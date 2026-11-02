"""Runs VoiceBench (https://github.com/MatthewCYM/VoiceBench) against gemmalive.

Each spoken prompt is answered by gemmalive-respond in a fresh conversation,
with the live session's model, audio path and system prompt, and scored with
VoiceBench's own evaluators. alpacaeval, commoneval, wildvoice and sd-qa are
judged by gpt-4o-mini through VoiceBench's api_judge.py, as in the paper,
which needs OPENAI_API_KEY (from the environment or a .env file, --env).
sd-qa uses the US speakers unless a region is given, e.g. sd-qa:gbr.

    git clone https://github.com/MatthewCYM/VoiceBench ../VoiceBench
    uv run --with-requirements bench/requirements.txt python bench/voicebench.py \
        --voicebench ../VoiceBench --subsets openbookqa,mmsu,bbh,ifeval,advbench
"""

import argparse
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import HfApi, hf_hub_download

REPO = "hlt-lab/voicebench"
REVISION = "b02edcef1330480be3a11bd6f7434ac32f05ad08"
RATE = 16000

# subset -> (VoiceBench evaluator, judged by gpt-4o-mini)
EVALUATORS = {
    "openbookqa": ("mcq", False),
    "mmsu": ("mcq", False),
    "bbh": ("bbh", False),
    "ifeval": ("ifeval", False),
    "advbench": ("harm", False),
    "alpacaeval": ("open", True),
    "alpacaeval_full": ("open", True),
    "commoneval": ("open", True),
    "wildvoice": ("open", True),
    "sd-qa": ("qa", True),
}


def load_subset(subset, split, limit):
    prefix = f"{subset}/{split}-" if split else f"{subset}/"
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
             if f.startswith(prefix) and f.endswith(".parquet")]
    rows = []
    for f in sorted(files):
        path = hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION)
        rows.extend(pq.read_table(path).to_pylist())
        if limit and len(rows) >= limit:
            break
    return rows[:limit] if limit else rows


def decode(audio):
    x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != RATE:
        x = soxr.resample(x, sr, RATE).astype(np.float32)
    return x


def respond(tool, rows, extra_args, modality):
    args = [tool, "--in", "-"] + extra_args + ([f"--{modality}"] if modality != "audio" else [])
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=False)

    def feed():
        for row in rows:
            if modality == "text":
                data = row["prompt"].encode()
                proc.stdin.write(struct.pack("<I", len(data)))
                proc.stdin.write(data)
            else:
                x = decode(row["audio"])
                proc.stdin.write(struct.pack("<I", len(x)))
                proc.stdin.write(x.astype("<f4").tobytes())
        proc.stdin.close()

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    results = []
    for i, line in enumerate(proc.stdout):
        results.append(json.loads(line))
        print(f"\r  {i + 1}/{len(rows)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    writer.join()
    if proc.wait() != 0 or len(results) != len(rows):
        raise SystemExit(f"gemmalive-respond failed ({len(results)} of {len(rows)} answered)")
    return results


def import_voicebench(voicebench):
    if voicebench not in sys.path:
        sys.path.insert(0, os.path.abspath(voicebench))
    # PEDANT, the second sd-qa metric, pulls in a model of its own; the
    # leaderboard reports the GPT judgement, so it is stubbed out.
    sys.modules.setdefault("qa_metrics", types.ModuleType("qa_metrics"))
    sys.modules.setdefault("qa_metrics.pedant", types.SimpleNamespace(PEDANT=None))


def judge(voicebench, records, workers=8):
    """Adds VoiceBench's gpt-4o-mini judgements (3 samples) as record["score"]."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (environment or .env)")
    import_voicebench(voicebench)
    import api_judge
    done = [0]

    def one(rec):
        rec = api_judge.generate(rec)
        done[0] += 1
        print(f"\r  judged {done[0]}/{len(records)}", end="", file=sys.stderr, flush=True)
        return rec

    with ThreadPoolExecutor(workers) as pool:
        judged = list(pool.map(one, records))
    print(file=sys.stderr)
    return judged


def evaluate(voicebench, evaluator, records):
    import_voicebench(voicebench)
    if evaluator == "ifeval":
        import nltk
        nltk.download("punkt_tab", quiet=True)
    if evaluator == "qa":
        from src.evaluator.qa import majority_vote
        return {"gpt": float(np.mean([majority_vote(r["score"]) for r in records]) * 100)}
    from src.evaluator import evaluator_mapping
    return evaluator_mapping[evaluator]().evaluate(records)


def load_dotenv(path=".env"):
    """Reads KEY=VALUE lines (e.g. OPENAI_API_KEY for the GPT judge) into the environment."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voicebench", required=True, help="checkout of the VoiceBench repository")
    ap.add_argument("--subsets", default="openbookqa,mmsu,bbh,ifeval,advbench")
    ap.add_argument("--limit", type=int, default=0, help="prompts per subset, 0 = all")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--tool", default="build/gemmalive-respond")
    ap.add_argument("--system", default=None, help="system prompt, default: the live session's")
    ap.add_argument("--model", default=None, help="model, default: the live session's")
    ap.add_argument("--mmproj", default=None, help="audio projector, default: the live session's")
    ap.add_argument("--env", default=".env", help="file with OPENAI_API_KEY, if not in the environment")
    ap.add_argument("--modality", default="audio", choices=["audio", "text", "oneshot"],
                    help="text answers the transcripts, to measure the gap to speech; oneshot "
                         "encodes the audio without streaming, to check the streaming encoder")
    ap.add_argument("--temp", default="0", help="sampling temperature, 0 = greedy")
    args = ap.parse_args()
    load_dotenv(args.env)

    extra = ["--temp", args.temp, "--max-tokens", "1024"]
    if args.system is not None:
        extra += ["--system", args.system]
    if args.model is not None:
        extra += ["--model", args.model]
    if args.mmproj is not None:
        extra += ["--mmproj", args.mmproj]

    # Answers are cached per tool build and arguments: a new binary (e.g. a
    # changed default prompt) or other options answer anew.
    with open(args.tool, "rb") as f:
        config = hashlib.sha256(f.read() + "\0".join(extra).encode()).hexdigest()[:8]
    if args.limit:
        config += f"-limit{args.limit}"

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for spec in args.subsets.split(","):
        subset, _, split = spec.partition(":")
        if subset not in EVALUATORS:
            raise SystemExit(f"unknown subset {subset}")
        if subset == "sd-qa":
            split = split or "usa"
        name = (f"{subset}-{split}" if split else subset) + ("" if args.modality == "audio" else "-" + args.modality)
        print(f"{name}:", file=sys.stderr)
        rows = load_subset(subset, split, args.limit)
        t0 = time.time()
        path = os.path.join(args.out, f"gemmalive-{subset}-{split or 'test'}-{args.modality}-{config}.jsonl")
        records = []
        if os.path.exists(path):
            with open(path) as f:
                records = [json.loads(line) for line in f]
        if len(records) == len(rows):
            print("  reusing answers from " + path, file=sys.stderr)
        else:
            answers = respond(args.tool, rows, extra, args.modality)
            records = []
            for row, ans in zip(rows, answers):
                rec = {k: v for k, v in row.items() if k != "audio"}
                rec.update(ans)
                records.append(rec)
            with open(path, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        entry = {
            "n": len(records),
            "seconds": round(time.time() - t0),
            "first_token_ms_median": float(np.median([r["first_token_ms"] for r in records])),
        }
        evaluator, judged = EVALUATORS[subset]
        if judged:
            records = judge(args.voicebench, records)
            with open(os.path.join(args.out, "result-" + os.path.basename(path)), "w") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        entry["score"] = evaluate(args.voicebench, evaluator, records)
        summary[name] = entry
        print(f"  {json.dumps(entry, default=str)}", file=sys.stderr)

    out_json = os.path.join(args.out, "voicebench.json")
    if os.path.exists(out_json):
        with open(out_json) as f:
            summary = {**json.load(f), **summary}
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
