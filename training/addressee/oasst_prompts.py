"""Extracts short spoken-style requests from OpenAssistant OASST2 (Apache 2.0):
the English prompts that open a conversation, as varied requests to an
assistant for make_data.py.

    python oasst_prompts.py --out data/oasst-prompts.txt
"""

import argparse
import gzip
import json
import re

from huggingface_hub import hf_hub_download

REPO = "OpenAssistant/oasst2"
REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
FILE = "2023-11-05_oasst2_prompts.messages.jsonl.gz"

# Words a person would say: letters, digits and light punctuation only.
SPEAKABLE = re.compile(r"^[A-Za-z0-9 ,.'?!-]+$")


def flagged(msg, name, limit=0.5):
    labels = msg.get("labels") or {}
    names, values = labels.get("name", []), labels.get("value", [])
    return any(n == name and v > limit for n, v in zip(names, values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=6000)
    args = ap.parse_args()

    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    prompts = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            m = json.loads(line)
            text = " ".join(m.get("text", "").split())
            if (m.get("role") != "prompter" or m.get("parent_id") or m.get("lang") != "en" or m.get("deleted")
                    or flagged(m, "spam") or flagged(m, "toxicity") or flagged(m, "not_appropriate")):
                continue
            if 3 <= len(text.split()) <= 20 and SPEAKABLE.match(text):
                prompts.add(text)
    prompts = sorted(prompts)[: args.max]
    with open(args.out, "w") as f:
        f.write("\n".join(prompts) + "\n")
    print(f"{len(prompts)} prompts")


if __name__ == "__main__":
    main()
