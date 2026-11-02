#!/usr/bin/env bash
# Extracts features for every Smart Turn shard, a few shards at a time.
#
#   training/turn/extract.sh <data dir> <features dir> [workers]
#
# Finished shards are skipped, so the script can be rerun after interruption.
# PROMPTS=<file> draws a system prompt per clip from it (training/prompts),
# logged to <shard>.prompt. HISTORY=<file> puts conversations before the clips
# (training/history.py). MODEL / MMPROJ: another Gemma than E2B. PREPARE:
# options for prepare.py (--long).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
python=${PYTHON:-$root/.venv/bin/python}
tool=${TOOL:-$root/build/gemmalive-turn-features}
model=${MODEL:-$root/models/gemma-4-E2B-it-XS-TS-Q3_K_S.gguf}
mmproj=${MMPROJ:-$root/models/mmproj-gemma-4-E2B-it-Q8_0.gguf}

if [ "$1" = "--shard" ]; then
    parquet=$2
    out=$3
    case $parquet in
        *-test/*) split=test ;;
        *)        split=train ;;
    esac
    shard=$(basename "$parquet" .parquet | sed 's/^train-\([0-9]*\)-of-.*/\1/')
    name=$out/$split-$shard
    [ -f "$name.f16" ] && exit 0
    prompts=()
    [ -n "${PROMPTS:-}" ] && prompts=(--prompts "$PROMPTS" --prompt-log "$name.prompt" --seed "$((10#$shard + 1))")
    [ -n "${HISTORY:-}" ] && prompts+=(--history "$HISTORY")
    "$python" "$here/prepare.py" "$parquet" --meta "$name.tsv.part" ${PREPARE:-} |
        "$tool" --in - --out "$name.f16.part" \
            --model "$model" --mmproj "$mmproj" ${prompts[@]+"${prompts[@]}"} 2> "$name.log"
    mv "$name.tsv.part" "$name.tsv"
    mv "$name.f16.part" "$name.f16"
    echo "$name done"
    exit 0
fi

data=$1
out=$2
workers=${3:-3}
mkdir -p "$out"

ls "$data"/smart-turn-v3.2-test/data/*.parquet "$data"/smart-turn-v3.2-train/data/*.parquet |
    xargs -P "$workers" -I{} "$0" --shard {} "$out"
