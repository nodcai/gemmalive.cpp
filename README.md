# gemmalive.cpp

Cross-platform local AI voice pipeline around Gemma 4, in C++ on
[ggml](https://github.com/ggml-org/ggml).

- LLM: Gemma 4 E2B or E4B, with the audio encoder fed while the user speaks
- TTS: VibeVoice Realtime 0.5B via [livevoice.cpp](https://github.com/nodcai/livevoice.cpp), fed while the reply is written
- Audio: in and out through the app that embeds it, which cancels the echo

## Models

```sh
cmake -P scripts/download_models.cmake              # Gemma 4 E2B
cmake -DGEMMA=E4B -P scripts/download_models.cmake  # Gemma 4 E4B
```

Downloads pinned, hash-checked files into `models/`:

- Gemma 4 E2B, 2.3 GB ([nodcai/gemma-4-E2B-live-gguf](https://huggingface.co/nodcai/gemma-4-E2B-live-gguf)),
  or E4B, 3.5 GB ([nodcai/gemma-4-E4B-live-gguf](https://huggingface.co/nodcai/gemma-4-E4B-live-gguf)):
  the model in TheStage's XS quantization, its mmproj Q8_0 and the
  end-of-turn and addressee heads in one file
- LiveVoice 0.5B, the compact quantization (577 MB, q8_0 attention so rare words come out right) + the Clarissa voice ([nodcai/livevoice-0.5b-gguf](https://huggingface.co/nodcai/livevoice-0.5b-gguf))
- Silero VAD v5.1.2 ([ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad)), so that only speech opens a turn

The program and tools run E2B unless told `--gemma E4B`; another file goes
in with `--model`, and with `--mmproj` if its audio encoder is in a file of
its own.

The end-of-turn and addressee heads are trained here (`training/`) for a
model and its mmproj, and exported next to the model as `<model>-heads.gguf`,
where gemmalive looks for them if they are not in the model's file. Without
them a fixed silence ends a turn and everyone is answered.
`training/turn/combine.py` puts a model, its mmproj and its heads into one
file, as the ones above are made.

## Build

```sh
cmake --preset default
cmake --build --preset default
```

Dependencies are fetched at pinned versions (see `cmake/dependencies.cmake`):

- [ggml](https://github.com/ggml-org/ggml) v0.28.0, the one ggml every model runs on
- [nodcai/livevoice.cpp](https://github.com/nodcai/livevoice.cpp), on that ggml
- [miniaudio](https://github.com/mackron/miniaudio) to read audio files

The models run on the GPU (Metal, CUDA, Vulkan) where there is one, with the
CPU for anything it can't run; `GEMMALIVE_BACKEND=cpu` (or a device name)
picks one.

### How Gemma runs

Gemma 4 is implemented here (`src/gemma4.*`, `src/audio_encoder.*`,
`src/mel.*`, `src/tokenizer.*`), as one ggml graph per decode:

- The weights stay in the memory-mapped GGUF files: a device that can
  address host memory (Metal, the CPU) runs on them in place, others get a
  copy. The per-layer token embeddings (0.7 GiB on E2B XS) never go to the
  device; their rows are read on the host, a row per token.
- The global-attention layers keep every position of the conversation, the
  sliding-window ones 2048: the window, a batch and room for taking back the
  end of a reply. Taking back more, e.g. a whole turn or a conversation back
  to its system prompt, goes to a guard: the end of the system turn and the
  start or end of the current turn keep the window below them, saved once the
  caches would drop it. The last 18 layers read the caches of earlier ones.
- The audio is turned into features as it arrives (`Mel`), and the encoder
  runs a chunk of 12 positions (480 ms) once it was heard: it sees the 12
  positions before a chunk and nothing after, so a chunk is final, and is
  encoded on its own with the keys, values and convolution inputs the
  previous one left. Closing the turn encodes what is left of the last one.
- The tokenizer is Gemma's BPE, with spaces as U+2581, byte tokens for the
  rest and the chat markup as single tokens.

### Embedding

Apps link the pipeline (`Llm`, `Tts`, `Session`, `TurnHead`, `Vad`) as the
`gemmalive` library. As a subproject, gemmalive builds only that; the program
and tools need `-DGEMMALIVE_BUILD_PROGRAMS=ON` / `-DGEMMALIVE_BUILD_TOOLS=ON`.

Besides its events, a `Session` reports each piece of work it does and when
(`Callbacks::on_span`: hearing, commits, the heads, each token, synthesis) and
each microphone frame as it heard it (`on_frame`), for an app that shows where
the time goes.

```cmake
add_subdirectory(gemmalive.cpp)
target_link_libraries(app PRIVATE gemmalive)
```

### iOS

The same build cross-compiles for iOS (arm64), the device or the Simulator:

```sh
cmake -B build-ios -G Ninja -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
      -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0 \
      -DGEMMALIVE_BUILD_PROGRAMS=OFF -DGEMMALIVE_BUILD_TOOLS=OFF \
      -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
cmake --build build-ios --target gemmalive
```

The models run on Metal (`GEMMALIVE_METAL`), with the shaders built into the
library, except in the Simulator (`-DCMAKE_OSX_SYSROOT=iphonesimulator`), whose
Metal can't take the models' buffers. An app links the static libraries of the
build, with Foundation, Metal, MetalKit and Accelerate.

### Developing against local checkouts

The example preset builds livevoice.cpp from a checkout next to this one
(`../livevoice.cpp`; another place goes into the copied file):

```sh
cp CMakeUserPresets.json.example CMakeUserPresets.json
cmake --preset dev
cmake --build --preset dev
```

## Run

Speech-to-speech test from audio files, one conversation turn per `-i`:

```sh
./build/gemmalive -i question.wav -o reply.wav
./build/gemmalive -i turn1.wav -i turn2.wav -o reply.wav   # reply-1.wav, reply-2.wav
./build/gemmalive --gemma E4B -i question.wav              # E4B instead of E2B
```

The user audio is streamed into Gemma in 20 ms chunks, the reply text is streamed
into livevoice as it is generated, and latencies are reported from the end of the
user's speech.

### Device and tools

The assistant knows the date and time zone (in the system prompt, at the start
of each conversation), the time of each turn (after the user's speech) and the
user's locale, which picks the units.

An app gives them in a `Device` (`src/device.h`): the user's locale and their
time zone. The program takes `--locale de_DE` (default: from `LANG`), with
this machine's date and time.

Gemma can call tools of the app, e.g. to look up what changed since it was
trained: the app declares them (`Llm::Params::tools`, `src/llm.h`), says when
to use them in the system prompt (`context`) and answers the calls
(`Session`'s `on_tool_call`, from any thread). A call is made whenever Gemma
gives it a probability of 0.002 or more, not only when it is sampled: Gemma
often talks about looking something up instead. Spoken, Gemma is sure of what
it knows; a reminder in the note after each turn (`turn_context`) that its
knowledge is dated gets it to look things up. A reply that opens with a call
first says "Let me check." or "One moment.", which is spoken while the tool
runs. The reply waits for the answer while speech is still heard; a call cut
short is taken back. The session's events report each call (`tool_call`) and
its answer (`tool_result`).

## Training the heads

The end-of-turn and addressee heads are small classifiers on Gemma's hidden
states at the end of committed user audio, in the conversation. They are
trained on features extracted with `gemmalive-turn-features` under system
prompts drawn from `training/prompts/train.txt`, after conversations of up to
three spoken turns and the model's answers, so that they work with any prompt
at any point of a conversation. With a venv from `training/requirements.txt`
(`.venv`, or `PYTHON=...`):

```sh
cmake -DPARTS=ON -P scripts/download_models.cmake   # the model and mmproj on their own
training/get_data.sh   # Smart Turn v3.2, Piper voices, LibriSpeech, OASST2 into data/

# Earlier turns for the conversations
python training/history.py --split train --n 2000 --seed 1 --out data/history-train.pcm
export PROMPTS=training/prompts/train.txt HISTORY=data/history-train.pcm

# End of turn: Smart Turn, and synthetic utterances with a pause inside
training/turn/extract.sh data features/turn
python training/turn/make_pauses.py --split train --n 4000 --seed 1 --out data/turn-pauses/train-01
build/gemmalive-turn-features --in data/turn-pauses/train-01.pcm --out features/turn/train-q01.f16 \
    --prompts $PROMPTS --history $HISTORY --prompt-log features/turn/train-q01.prompt --seed 101
cp data/turn-pauses/train-01.tsv features/turn/train-q01.tsv

# Addressee: synthetic requests, interruptions, side talk, backchannels,
# background speech, and Smart Turn clips as speech to the assistant
training/addressee/extract.sh features/addressee 15 3
python training/addressee/add_smart_turn.py --features features/turn --out features/addressee

python training/turn/train.py --features features/turn --layers 8,10,12,13 --hidden 256 --out heads/turn
python training/turn/train.py --features features/addressee --layers 8,14,17,21 --hidden 256 \
    --label addressed --test-prefix val --group-by kind --out heads/addressee
python training/turn/export.py models/gemma-4-E2B-it-XS-TS-Q3_K_S-heads.gguf \
    --end-of-turn heads/turn.pt --addressee heads/addressee.pt --any-prompt --threshold addressee=0.3
python training/turn/combine.py models/gemma-4-E2B-live-XS.gguf \
    --model models/gemma-4-E2B-it-XS-TS-Q3_K_S.gguf --mmproj models/mmproj-gemma-4-E2B-it-Q8_0.gguf
```

`training/prompts/heldout.txt` holds prompts left out of training, to check
the heads on prompts they have not seen, and `history.py --split val` turns
by voices and texts left out.

## Tests

```sh
ctest --test-dir build --output-on-failure
```

The tests of the pipeline's own logic (`tests/`) need no model: the tool
markup, the text that goes to speech, the device context, sampling, the audio features and the heads' files. Those of
the tokenizer and of a conversation (taking back an abandoned turn, holding
a reply) are skipped unless they are given a model:

```sh
GEMMALIVE_MODEL=$PWD/models/gemma-4-E2B-live-XS.gguf ctest --test-dir build --output-on-failure
```

A model without its audio encoder takes the mmproj in `GEMMALIVE_MMPROJ`.

`-DGEMMALIVE_BUILD_TESTS=OFF` leaves the tests out; as a subproject,
gemmalive builds none.

## Benchmarks

- `bench/fullduplex.py`: Full-Duplex-Bench v1.5 (interruptions, backchannels,
  side talk, background speech) through `gemmalive-duplex`, which runs
  recorded audio through a live session in real time.
- `bench/voicebench.py`: VoiceBench's spoken-question subsets through
  `gemmalive-respond` (`bench/requirements.txt`).
