#!/usr/bin/env bash
# Generates and extracts addressee clips: train and val shards, a few at a time.
#
#   training/addressee/extract.sh <features dir> [train shards] [val shards] [clips per shard] [workers]
#
# Finished shards are skipped, so the script can be rerun after interruption.
# PROMPTS=<file> draws a system prompt per clip from it (training/prompts),
# logged to <shard>.prompt. HISTORY=<file> puts conversations before the clips
# (training/history.py). MODEL / MMPROJ: another Gemma than E2B.
# WEIGHTS=kind=weight,... changes the mix; FIRST_TRAIN / FIRST_VAL number
# additional shards after existing ones.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
python=${PYTHON:-$root/.venv/bin/python}
tool=${TOOL:-$root/build/gemmalive-turn-features}
model=${MODEL:-$root/models/gemma-4-E2B-it-XS-TS-Q3_K_S.gguf}
mmproj=${MMPROJ:-$root/models/mmproj-gemma-4-E2B-it-Q8_0.gguf}

if [ "$1" = "--shard" ]; then
    split=$2; seed=$3; n=$4; out=$5; weights=${WEIGHTS:-}
    name=$out/$split-$(printf %02d "$seed")
    [ -f "$name.f16" ] && exit 0
    prompts=()
    [ -n "${PROMPTS:-}" ] && prompts=(--prompts "$PROMPTS" --prompt-log "$name.prompt" --seed "$seed")
    [ -n "${HISTORY:-}" ] && prompts+=(--history "$HISTORY")
    (cd "$root" && "$python" "$here/make_data.py" --split "$split" --n "$n" --seed "$seed" --weights "$weights" --meta "$name.tsv.part") |
        "$tool" --in - --out "$name.f16.part" \
            --model "$model" --mmproj "$mmproj" ${prompts[@]+"${prompts[@]}"} 2> "$name.log"
    mv "$name.tsv.part" "$name.tsv"
    mv "$name.f16.part" "$name.f16"
    echo "$name done"
    exit 0
fi

out=$1
n_train=${2:-8}
n_val=${3:-1}
per_shard=${4:-3000}
workers=${5:-4}
mkdir -p "$out"
out=$(cd "$out" && pwd)  # make_data.py runs from the repo root
{
    for s in $(seq "${FIRST_VAL:-1}" "$((${FIRST_VAL:-1} + n_val - 1))"); do echo "val $s"; done
    for s in $(seq "${FIRST_TRAIN:-1}" "$((${FIRST_TRAIN:-1} + n_train - 1))"); do echo "train $s"; done
} | xargs -P "$workers" -L 1 bash -c '"$0" --shard "$3" "$4" "$1" "$2"' "$0" "$per_shard" "$out"
