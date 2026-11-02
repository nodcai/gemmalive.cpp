"""Writes trained heads (.pt from train.py) as the GGUF file TurnHead loads.

    python export.py models/gemma-4-E2B-it-XS-TS-Q3_K_S-heads.gguf \
        --end-of-turn heads/turn.pt --addressee heads/addressee.pt \
        --any-prompt --threshold addressee=0.3 \
        --model models/gemma-4-E2B-it-XS-TS-Q3_K_S.gguf \
        --mmproj models/mmproj-gemma-4-E2B-it-Q8_0.gguf

The heads of a model go into one file next to it, <model>-heads.gguf, where
gemmalive looks for them (combine.py puts them into the model's own file).
Besides the weights the file records what the
heads were trained on: the layers each reads, the system prompt, and
fingerprints of the model and audio projector files (SHA-256 of their GGUF
header, i.e. all metadata and tensor layout), which gemmalive checks before
using them. Heads trained on features from many system prompts
(--any-prompt) are used with any.
"""

import argparse
import hashlib
import json
import os
import re

import gguf
import numpy as np
import torch

ARCH = "gemmalive-heads"

PIPER = ("Piper voices trained on public domain, CC0, CC BY 4.0 (VCTK, LibriTTS-R, CSTR alba, ARU), "
         "Apache 2.0, CMU ARCTIC and Jenny data")

TASKS = {
    "end_of_turn": (
        "gemmalive turn head",
        "End-of-turn classifier on {model} hidden states at the end of committed user audio.",
        f"pipecat-ai/smart-turn-data-v3.2-train (CC BY 4.0); synthetic paused utterances from {PIPER}",
    ),
    "addressee": (
        "gemmalive addressee head",
        "Classifier on {model} hidden states telling speech meant for the assistant from "
        "side talk, backchannels and background speech.",
        f"synthetic speech from {PIPER}; request texts from OpenAssistant OASST2 (Apache 2.0); "
        "LibriSpeech (CC BY 4.0); pipecat-ai/smart-turn-data-v3.2-train (CC BY 4.0)",
    ),
}
ANY_PROMPT_NOTE = "; under system prompts drawn from training/prompts/train.txt"


def fingerprint(path):
    """SHA-256 of a GGUF file's header: everything before the tensor data."""
    offset = gguf.GGUFReader(path).data_offset
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(offset)).hexdigest()


def default_system_prompt():
    """The system prompt the features were extracted with: Llm's default."""
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src", "llm.h")
    m = re.search(r"system_prompt\s*=\s*((?:\s*\"(?:[^\"\\]|\\.)*\")+)\s*;", open(src).read())
    if not m:
        raise SystemExit("could not find the default system prompt in src/llm.h")
    return "".join(re.findall(r"\"((?:[^\"\\]|\\.)*)\"", m.group(1)))


def per_task(values, what):
    """TASK=VALUE options as a dict."""
    out = {}
    for v in values:
        task, sep, value = v.partition("=")
        if not sep or task not in TASKS:
            raise SystemExit(f"--{what}: expected TASK=VALUE with TASK one of {', '.join(sorted(TASKS))}")
        out[task] = value
    return out


def model_name(path):
    """"Gemma 4 E2B" for models/gemma-4-E2B-it-...gguf, else the file name."""
    m = re.search(r"gemma-(\d+)-(E\d+B)", os.path.basename(path))
    return f"Gemma {m.group(1)} {m.group(2)}" if m else os.path.basename(path)


def report_of(checkpoint):
    """train.py's report next to a checkpoint, or None."""
    path = os.path.splitext(checkpoint)[0] + ".json"
    return json.load(open(path)) if os.path.exists(path) else None


def add_head(w, task, checkpoint, threshold, training_data, model):
    ck = torch.load(checkpoint, map_location="cpu")
    st = {k: v.float().numpy() for k, v in ck["state"].items()}
    layers = ck["layers"]
    hidden = ck["hidden"]
    key, tensor = f"heads.{task}.", f"{task}."
    w.add_array(key + "layers", [int(l) for l in layers])
    w.add_uint32(key + "n_embd", st["mean"].size)
    w.add_uint32(key + "hidden", hidden)
    w.add_float32(key + "threshold", threshold)
    w.add_string(key + "name", TASKS[task][0])
    w.add_string(key + "description", TASKS[task][1].format(model=model))
    w.add_string(key + "training_data", training_data)

    r = report_of(checkpoint)
    if r:
        # Measured at the head's threshold; the report's top level is at 0.5.
        at = r.get("thresholds", {}).get(f"{threshold:g}", r if threshold == 0.5 else None)
        if at is None:
            print(f"{task}: no test metrics at threshold {threshold:g} in the report, left out")
        else:
            if task == "end_of_turn":
                w.add_string(key + "test_data", "pipecat-ai/smart-turn-data-v3.2-test")
            for k in ("accuracy", "fpr", "fnr"):
                w.add_float32(key + f"test.{k}", float(at[k]))

    mix = torch.softmax(torch.from_numpy(st["mix"]), 0).numpy()
    w.add_tensor(tensor + "mix", mix.astype(np.float32))
    w.add_tensor(tensor + "mean", st["mean"].astype(np.float32))
    w.add_tensor(tensor + "std", st["std"].astype(np.float32))
    if hidden:
        w.add_tensor(tensor + "fc1.weight", st["net.0.weight"].astype(np.float32))  # [hidden][n_embd]
        w.add_tensor(tensor + "fc1.bias", st["net.0.bias"].astype(np.float32))
        w.add_tensor(tensor + "fc2.weight", st["net.2.weight"].reshape(-1).astype(np.float32))
        w.add_tensor(tensor + "fc2.bias", st["net.2.bias"].astype(np.float32))
    else:
        w.add_tensor(tensor + "fc2.weight", st["net.weight"].reshape(-1).astype(np.float32))
        w.add_tensor(tensor + "fc2.bias", st["net.bias"].astype(np.float32))
    print(f"{task}: layers {layers}, hidden {hidden}, threshold {threshold}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="<model>-heads.gguf, next to the model")
    ap.add_argument("--end-of-turn", metavar="PT", help="end-of-turn head checkpoint")
    ap.add_argument("--addressee", metavar="PT", help="addressee head checkpoint")
    ap.add_argument("--model", default="models/gemma-4-E2B-it-XS-TS-Q3_K_S.gguf")
    ap.add_argument("--mmproj", default="models/mmproj-gemma-4-E2B-it-Q8_0.gguf")
    ap.add_argument("--system-prompt", default=None, help="default: Llm's default prompt")
    ap.add_argument("--any-prompt", action="store_true",
                    help="trained on many system prompts (training/prompts): works with any")
    ap.add_argument("--threshold", action="append", default=[], metavar="TASK=VALUE",
                    help="decision threshold of a head (default 0.5; train.py's report suggests one)")
    ap.add_argument("--training-data", action="append", default=[], metavar="TASK=TEXT",
                    help="overrides a head's training data note")
    args = ap.parse_args()

    checkpoints = {t: p for t, p in (("end_of_turn", args.end_of_turn), ("addressee", args.addressee)) if p}
    if not checkpoints:
        raise SystemExit("no heads: give --end-of-turn and/or --addressee")
    thresholds = {t: float(v) for t, v in per_task(args.threshold, "threshold").items()}
    notes = per_task(args.training_data, "training-data")
    if args.any_prompt and args.system_prompt is not None:
        raise SystemExit("--system-prompt has no effect with --any-prompt")
    system_prompt = args.system_prompt if args.system_prompt is not None else default_system_prompt()

    w = gguf.GGUFWriter(args.output, ARCH)
    w.add_name("gemmalive heads")
    w.add_description(f"Classifiers on {model_name(args.model)} hidden states at the end of committed user audio.")
    w.add_array("heads.tasks", list(checkpoints))
    if args.any_prompt:
        w.add_bool("heads.any_system_prompt", True)
    else:
        w.add_string("heads.system_prompt", system_prompt)
    w.add_string("heads.model.file", os.path.basename(args.model))
    w.add_string("heads.model.fingerprint", fingerprint(args.model))
    w.add_string("heads.mmproj.file", os.path.basename(args.mmproj))
    w.add_string("heads.mmproj.fingerprint", fingerprint(args.mmproj))
    for task, checkpoint in checkpoints.items():
        note = notes.get(task, TASKS[task][2] + (ANY_PROMPT_NOTE if args.any_prompt else ""))
        add_head(w, task, checkpoint, thresholds.get(task, 0.5), note, model_name(args.model))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
