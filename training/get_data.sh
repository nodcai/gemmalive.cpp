#!/usr/bin/env bash
# Downloads the heads' training data into data/: Smart Turn v3.2, the Piper
# voices the synthetic clips are spoken with (make_data.VOICES), LibriSpeech
# dev-clean and test-clean for background speech, and OASST2 requests.
# Present parts are skipped.
#
#   training/get_data.sh
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-$root/.venv/bin/python}
mkdir -p "$root/data"
cd "$root/data"

for split in train test; do
    [ -d "smart-turn-v3.2-$split" ] ||
        "$python" -c "import sys; from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type='dataset', local_dir=sys.argv[2])" \
            "pipecat-ai/smart-turn-data-v3.2-$split" "smart-turn-v3.2-$split"
done

rev=c10ece1aade47bb51c153c893d14e5bf8e5b7117  # rhasspy/piper-voices
mkdir -p piper
for v in en_GB/alba en_GB/aru en_GB/cori en_GB/jenny_dioco en_GB/vctk en_US/arctic en_US/bryce en_US/joe \
         en_US/john en_US/kristin en_US/ljspeech en_US/mike en_US/norman en_US/reza_ibrahim en_US/sam; do
    name=${v%%/*}-${v##*/}-medium
    for ext in onnx onnx.json; do
        [ -f "piper/$name.$ext" ] ||
            curl -sfL -o "piper/$name.$ext" "https://huggingface.co/rhasspy/piper-voices/resolve/$rev/en/$v/medium/$name.$ext"
    done
done

mkdir -p librispeech
for split in dev-clean test-clean; do
    [ -d "librispeech/LibriSpeech/$split" ] ||
        curl -sfL "https://us.openslr.org/resources/12/$split.tar.gz" | tar xz -C librispeech
done

[ -f oasst-prompts.txt ] || (cd "$root" && "$python" training/addressee/oasst_prompts.py --out data/oasst-prompts.txt)
echo "data ready in $root/data"
