"""Runs Full-Duplex-Bench v1.5 (https://github.com/DanielLin94144/Full-Duplex-Bench)
against gemmalive.

Every sample is played through gemmalive-duplex in real time twice, with and
without its overlap event (input.wav, clean_input.wav), giving output.wav and
clean_output.wav on the input's timeline. Scoring uses the benchmark's own
scripts: Parakeet transcripts, timing, and GPT-4o behaviour labels (needs
OPENAI_API_KEY, from the environment or a .env file).

    python bench/fullduplex.py run --data fdb/v1_5 --out bench/fdb/run --workers 4
    python bench/fullduplex.py score --fdb Full-Duplex-Bench --out bench/fdb/run \
        --python bench/fdb/.venv-eval/bin/python
    python bench/fullduplex.py summarize --out bench/fdb/run
"""

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import statistics
import time

SUBSETS = ["user_interruption", "user_backchannel", "talking_to_other", "background_speech"]


def load_dotenv(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def samples(out, subset):
    dirs = [d for d in glob.glob(os.path.join(out, subset, "*")) if os.path.exists(os.path.join(d, "input.wav"))]
    return sorted(dirs, key=lambda d: (len(os.path.basename(d)), os.path.basename(d)))


def run(args):
    jobs = []
    for subset in args.subsets.split(","):
        src = os.path.join(args.data, subset)
        dst = os.path.join(args.out, subset)
        if not os.path.exists(dst):
            names = sorted(os.listdir(src), key=lambda n: (len(n), n))
            if args.limit:
                names = [n for n in names if os.path.exists(os.path.join(src, n, "input.wav"))][: args.limit]
            for n in names:
                if os.path.isdir(os.path.join(src, n)):
                    shutil.copytree(os.path.join(src, n), os.path.join(dst, n))
        for d in samples(args.out, subset):
            for name in ("input", "clean_input"):
                out_wav = os.path.join(d, name.replace("input", "output") + ".wav")
                if not os.path.exists(out_wav):
                    jobs.append((os.path.join(d, name + ".wav"), out_wav))

    print(f"{len(jobs)} runs on {args.workers} workers", file=sys.stderr)
    procs = []
    for w in range(args.workers):
        part = jobs[w :: args.workers]
        if not part:
            continue
        list_path = os.path.join(args.out, f"jobs-{w}.tsv")
        with open(list_path, "w") as f:
            f.writelines(f"{i}\t{o}\n" for i, o in part)
        log = open(os.path.join(args.out, f"worker-{w}.log"), "w")
        procs.append(subprocess.Popen([args.tool, "--list", list_path] + shlex.split(args.tool_args),
                                      stdout=log, stderr=subprocess.STDOUT))
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(30)
        done = sum(os.path.exists(o) for _, o in jobs)
        print(f"\r  {done}/{len(jobs)} after {(time.time() - t0) / 60:.0f} min", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    failed = [p.returncode for p in procs if p.returncode]
    if failed:
        raise SystemExit(f"{len(failed)} workers failed, see {args.out}/worker-*.log")


def score(args):
    load_dotenv()
    fdb = os.path.join(args.fdb, "v1_v1.5")
    # The benchmark's scripts run from their own folders.
    python = os.path.abspath(args.python) if os.sep in args.python else args.python
    env = dict(os.environ)
    summary = {}
    for subset in args.subsets.split(","):
        root = os.path.abspath(os.path.join(args.out, subset))
        print(f"{subset}:", file=sys.stderr)
        for audio in ("output.wav", "clean_output.wav"):
            if not all(os.path.exists(os.path.join(d, audio.replace(".wav", ".json"))) for d in samples(args.out, subset)):
                subprocess.run([python, "asr.py", "--root_dir", root, "--task", "default", "--audio_name", audio],
                               cwd=os.path.join(fdb, "get_transcript"), check=True, env=env)
        subprocess.run([python, "get_timing.py", "--root_dir", root],
                       cwd=os.path.join(fdb, "evaluation"), check=True, env=env)
        if os.environ.get("OPENAI_API_KEY"):
            subprocess.run([python, "evaluate.py", "--task", "behavior", "--root_dir", root],
                           cwd=os.path.join(fdb, "evaluation"), check=True, env=env)
        else:
            print("  OPENAI_API_KEY not set, skipping behaviour labels", file=sys.stderr)
    summarize(args)


# What the paper counts as right for each scenario.
EXPECTED = {
    "user_interruption": "C_RESPOND",
    "user_backchannel": "C_RESUME",
    "talking_to_other": "C_RESUME",
    "background_speech": "C_RESUME",
}


def summarize(args):
    """Behaviour shares and the latencies around each sample's overlap event.

    Stop latency: how long the model keeps talking once the overlap starts
    (the model/user overlap that begins in the overlap window). Response
    latency: from the end of the overlapping speech, i.e. the user's last
    pause in the overlap window rather than one inside it, to the model's
    next utterance.
    """
    summary = {}
    for subset in args.subsets.split(","):
        tags, stops, resps = [], [], []
        for d in samples(args.out, subset):
            tag_path = os.path.join(d, "content_tag.json")
            if os.path.exists(tag_path):
                tags.extend(json.load(open(tag_path)).get("behaviour", []))
            lat_path = os.path.join(d, "latency_intervals.json")
            meta = json.load(open(os.path.join(d, "metadata.json")))
            if not os.path.exists(lat_path) or "timestamps" not in meta:
                continue
            start, end = meta["timestamps"]
            lat = json.load(open(lat_path))
            stop = [e - s for s, e in lat["latency_stop_list"] if start - 0.5 <= s <= end]
            ends = [(u, m) for u, m in lat["latency_resp_list"] if start <= u <= end + 1.0]
            resp = [m - u for u, m in sorted(ends, key=lambda um: abs(um[0] - end))]
            if stop:
                stops.append(stop[0])
            if resp:
                resps.append(resp[0])
        counts = {t: tags.count(t) for t in ("C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN")}
        n = max(len(tags), 1)
        summary[subset] = {
            "samples": len(samples(args.out, subset)),
            "behaviour": {k: round(v / n, 3) for k, v in counts.items()},
            "expected": EXPECTED[subset],
            "expected_rate": round(counts.get(EXPECTED[subset], 0) / n, 3),
            "stop_latency_s": round(statistics.mean(stops), 3) if stops else None,
            "stop_latency_n": len(stops),
            "response_latency_s": round(statistics.mean(resps), 3) if resps else None,
            "response_latency_n": len(resps),
        }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True, help="v1.5 data with one folder per subset")
    r.add_argument("--out", required=True)
    r.add_argument("--subsets", default=",".join(SUBSETS))
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=0, help="samples per subset, 0 = all")
    r.add_argument("--tool", default="build/gemmalive-duplex")
    r.add_argument("--tool-args", default="", help="extra arguments for gemmalive-duplex")
    s = sub.add_parser("score")
    s.add_argument("--fdb", required=True, help="checkout of the Full-Duplex-Bench repository")
    s.add_argument("--out", required=True)
    s.add_argument("--subsets", default=",".join(SUBSETS))
    s.add_argument("--python", default=sys.executable, help="interpreter with NeMo and the benchmark's dependencies")
    m = sub.add_parser("summarize")
    m.add_argument("--out", required=True)
    m.add_argument("--subsets", default=",".join(SUBSETS))
    args = ap.parse_args()
    {"run": run, "score": score, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
