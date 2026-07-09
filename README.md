# AICC Voice Pipeline Testing Framework

A modular set of command-line scripts for testing and comparing:

- **Speech-to-Text**: Google Cloud STT vs local faster-whisper
- **Text-to-Speech**: Google Cloud TTS vs local Piper TTS
- **LLM routing**: Ollama (llama3.2) for turning a transcript into a routing decision and a spoken response

You can test each component individually, run any of the four
STT+LLM+TTS pipeline combinations end to end, or batch-run a manifest of
scenarios across all combinations for later comparison.

No sample audio, sample text, sample scenarios, or credentials are
included in this project. You provide all inputs yourself.

## Folder structure

```
input_audio/            Your input .wav audio files
input_text/              Your input .txt text files
config/
  routes.example.json     Schema example - copy to routes.json and fill in your route tree
  manifest.example.csv    Schema example - copy to manifest.csv and fill in your scenarios
output/
  audio/                  Generated speech audio
  transcripts/            Generated transcripts
  llm_outputs/            Raw + parsed Ollama outputs
  logs/                   Small JSON logs from each component test
  results/                component_results.csv, pipeline_results.csv, summary.csv
src/
  utils.py                Shared helpers (CSV/JSON logging, timing)
  stt_google.py            Google Cloud STT backend
  stt_local.py             faster-whisper local STT backend
  tts_google.py            Google Cloud TTS backend
  tts_local.py             Piper local TTS backend
  llm_ollama.py            Ollama LLM backend
  pipeline.py              Full pipeline orchestration (used by run_pipeline.py and batch_runner.py)
  batch_runner.py          Manifest-driven batch testing logic
test_stt.py                Test one STT backend on one audio file
test_tts.py                Test one TTS backend on one text input
test_llm.py                Test the Ollama LLM backend on one transcript
run_pipeline.py            Run a full STT -> LLM -> TTS pipeline on one audio file
run_batch.py                Run pipeline combinations across a manifest of scenarios
compare_results.py          Summarize and compare pipeline_results.csv
app.py                      Flask web prototype (turn-based browser voice demo)
templates/index.html        Web prototype page
static/app.js                Web prototype browser recording + display logic
static/style.css             Web prototype styling
web_uploads/                Temporary browser recordings (not committed)
requirements.txt
.gitignore
```

Output folders are created automatically if they don't already exist.

## Setup

### 1. Activate your Python environment

```
source .venv/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Set up Google Cloud authentication

This project uses Application Default Credentials only. No API keys or
service account JSON files are used or created.

```
gcloud init
gcloud auth application-default login
```

Make sure the Speech-to-Text and Text-to-Speech APIs are enabled on your
Google Cloud project.

### 4. Confirm Ollama is running

```
ollama run llama3.2
```

Leave this running (or run `ollama serve` in the background) before using
any script that calls the LLM.

### 5. Local voice models

- faster-whisper downloads its model automatically the first time you run
  it (cached locally after that).
- Piper voices are **not** downloaded automatically by these scripts. Make
  sure a voice's `.onnx` and `.onnx.json` files exist under
  `models/piper/` before running local TTS (e.g. `en_US-lessac-medium.onnx`
  and `en_US-lessac-medium.onnx.json`).

## Testing individual components

### Google STT

```
python test_stt.py --provider google --audio input_audio/example.wav
```

### Local STT (faster-whisper)

```
python test_stt.py --provider local --audio input_audio/example.wav --model-size base.en
```

### Google TTS

```
python test_tts.py --provider google --text "Hello, this is a test." --output response.mp3
python test_tts.py --provider google --file input_text/response.txt --output response.mp3
```

### Local TTS (Piper)

```
python test_tts.py --provider local --file input_text/response.txt --output response.wav
```

### Ollama LLM

```
python test_llm.py --text "I need to check on my order status" --model llama3.2
python test_llm.py --file output/transcripts/example.txt --model llama3.2 --routes config/routes.json
```

To use a real route tree, copy `config/routes.example.json` to
`config/routes.json` and fill it in. If you skip `--routes`, the LLM still
returns a decision but marks `route_tree_used: false`.

### Prompt modes (latency)

The LLM step is the main latency bottleneck. There are three routing prompt
modes, selectable with `--prompt-mode`:

- `compact`: a short prompt that only asks the model to classify the route
  and action. It returns minimal JSON, and the customer-facing reply
  (`response_to_customer`) is filled in locally from templates
  (`generate_template_response` in `src/llm_ollama.py`) rather than being
  written by the model. This reduces prompt tokens and LLM latency, but on
  ambiguous cases it can over-route instead of asking for clarification.
- `compact_v2` (default): same short format and speed as `compact`, plus a
  few targeted clarification/safety rules for the AICC stress cases (for
  example: when a payment "didn't go through" but the payment type is
  unspecified, ask for clarification instead of guessing "Failed Transfer").
  The rules are phrased generically by what the user says, not tied to any
  scenario filename.
- `full`: the original longer prompt where the model also writes the reply,
  a confidence score, and an explanation. Kept for comparison/debugging.

```
python test_llm.py --file input_text/example.txt --model llama3.2 --routes config/routes.json --prompt-mode compact
python test_llm.py --file input_text/example.txt --model llama3.2 --routes config/routes.json --prompt-mode compact_v2
python test_llm.py --file input_text/example.txt --model llama3.2 --routes config/routes.json --prompt-mode full
```

### Benchmarking prompt modes across audio scenarios

`benchmark_prompt_modes.py` runs every `.wav` in an audio directory through
the full pipeline once per prompt mode and summarizes latency and
route-validity per mode. It defaults to local STT/TTS so no Google APIs are
called.

```
python benchmark_prompt_modes.py \
  --audio-dir input_audio_16k \
  --routes config/routes.json \
  --stt local \
  --tts local \
  --model llama3.2 \
  --modes compact compact_v2 full
```

It writes per-run detail to `output/results/prompt_mode_benchmark.csv` and a
per-mode summary to `output/results/prompt_mode_summary.csv` (both are
overwritten each run), and prints a comparison table. It uses
`src.pipeline.run_pipeline()` directly, so it does **not** modify
`pipeline_results.csv`.

### Router modes (rule-based pre-router)

To cut latency further, a rule-based **pre-router** can answer obvious
requests without calling Ollama at all. `--router-mode` selects the
behavior:

- `pre_router` (default): try `src/pre_router.py` first. If a conservative
  keyword rule matches (fraud, lost/stolen card, locked account, ATM
  withdrawal, scheduled transfer, clear failed transfer, vague card/payment
  clarification, or clearly out-of-scope), the request is routed instantly
  with `llm_latency_seconds = 0` and `routing_source = "pre_router"`. If no
  rule matches, it falls back to the Ollama router using `--prompt-mode`.
- `llm_only`: always use the Ollama router (the pre-router is skipped).

The pre-router is conservative on purpose - when a request is ambiguous or
risky it returns `handled=false` and the LLM handles it, so accuracy is
preserved while latency drops on the easy cases.

```
python run_pipeline.py --audio input_audio_16k/example.wav --stt local --tts local --model llama3.2 --routes config/routes.json --prompt-mode compact_v2 --router-mode pre_router
python run_pipeline.py --audio input_audio_16k/example.wav --stt local --tts local --model llama3.2 --routes config/routes.json --prompt-mode compact_v2 --router-mode llm_only
```

The pipeline result rows gain `router_mode`, `routing_source`,
`pre_router_handled`, `rule_name`, and `route_latency_seconds` (routing time
regardless of source; equals `llm_latency_seconds` in `llm_only` mode).

To compare router modes across every scenario in an audio directory:

```
python benchmark_router_modes.py \
  --audio-dir input_audio_16k \
  --routes config/routes.json \
  --stt local \
  --tts local \
  --model llama3.2 \
  --prompt-mode compact_v2 \
  --router-modes llm_only pre_router
```

It writes `output/results/router_mode_benchmark.csv` and
`output/results/router_mode_summary.csv` (the summary includes
`pre_router_handled_rate`, `route_valid_rate`, and `normalized_count` per
mode) and prints a table sorted by average total latency.

### Route normalization (LLM output post-processing)

Small local LLMs sometimes return a valid sub-route name without the full
path (e.g. `"Failed Transfer"` instead of
`"Transfers and Payments > Failed Transfer"`), or use `->` instead of `>`.
These are semantically correct but fail exact route-tree validation. After
parsing the model's JSON, `route_with_ollama()` runs
`normalize_route_decision()` (in `src/llm_ollama.py`), which:

- rewrites `->` to `>` and trims extra whitespace, and
- expands a sub-route-only label to its exact full path **only when that
  sub-route name appears under exactly one top-level route** (or can be
  disambiguated by the model's own `top_level_route`). Ambiguous or unknown
  labels are left unchanged - never invented.

It also aligns `top_level_route` to the resolved path's parent, then
re-validates. The result/log/CSV gain `route_normalized`,
`original_final_route`, and `normalized_final_route` so every change is
inspectable. Normalization is applied to the LLM path in all prompt modes
(`compact`, `compact_v2`, `full`); the pre-router already returns exact
routes, so nothing is normalized there.

### Benchmark isolation

`benchmark_prompt_modes.py` and `benchmark_models.py` default to
`--router-mode llm_only` so prompt-mode and model comparisons measure the
LLM directly, unaffected by the pre-router. Pass `--router-mode pre_router`
explicitly to include it. `benchmark_router_modes.py` is the script for
comparing the router modes themselves.

## Running a full pipeline

Any combination of STT and TTS backend can be paired with the Ollama LLM:

```
# Google STT + Ollama + Google TTS
python run_pipeline.py --audio input_audio/example.wav --stt google --tts google --model llama3.2 --routes config/routes.json

# Local STT + Ollama + Local Piper TTS
python run_pipeline.py --audio input_audio/example.wav --stt local --tts local --model llama3.2

# Google STT + Ollama + Local Piper TTS
python run_pipeline.py --audio input_audio/example.wav --stt google --tts local --model llama3.2

# Local STT + Ollama + Google TTS
python run_pipeline.py --audio input_audio/example.wav --stt local --tts google --model llama3.2
```

Each run saves the transcript, LLM raw/parsed output, and response audio,
and appends one row to `output/results/pipeline_results.csv`.

Pipelines also accept `--prompt-mode compact|full` (default `compact`) to
choose the routing prompt, e.g.:

```
python run_pipeline.py --audio input_audio/example.wav --stt local --tts local --model llama3.2 --routes config/routes.json --prompt-mode full
```

## Running batch tests from a manifest

Copy `config/manifest.example.csv` to `config/manifest.csv` and fill in
your own scenarios (columns: `scenario_id, audio_path, expected_route,
expected_action, stress_type, notes`).

```
python run_batch.py --manifest config/manifest.csv
```

By default this runs all four provider combinations for every scenario.
To run only specific combinations:

```
python run_batch.py --manifest config/manifest.csv --combinations google_google local_local
```

Valid combination names: `google_google`, `local_local`, `google_local`, `local_google`.

Results (including a blank `manual_success_label` column for later human
review) are appended to `output/results/pipeline_results.csv`.

## Comparing results

```
python compare_results.py --results output/results/pipeline_results.csv
```

Prints a comparison table (run counts, average latencies, error counts,
and manual success rate if labels have been filled in) and saves it to
`output/results/summary.csv`.

## Running the Flask Voice Prototype

A small browser-based prototype lets you speak into your microphone and get
a full STT -> Ollama routing -> TTS round trip back, without using the
command line. It is turn-based (record one message, wait for the reply),
not real-time streaming.

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Install ffmpeg

The app uses `ffmpeg` to convert the browser's recorded audio (webm) into
the 16kHz mono WAV format the STT backends expect.

```
brew install ffmpeg
```

### 3. Confirm Google Cloud authentication

Same Application Default Credentials setup as the CLI scripts - no extra
config needed:

```
gcloud auth application-default login
```

### 4. Confirm Ollama is running

```
ollama run llama3.2
```

### 5. Run the Flask app

```
python app.py
```

### 6. Open the prototype in your browser

```
http://127.0.0.1:5000
```

Choose your STT/TTS providers and Ollama model, click **Start Recording**,
speak, click **Stop Recording**, and wait for the transcript, routing
decision, and spoken response to appear. The response audio plays
automatically.

Generated response audio is saved under `output/web_audio/`, and one JSON
log per turn is saved under `output/logs/web_turns/`. Raw browser
recordings are saved temporarily under `web_uploads/` (not committed to
version control).

## Where outputs are saved

- `output/transcripts/` - STT transcripts
- `output/audio/` - synthesized response audio (CLI scripts)
- `output/web_audio/` - synthesized response audio (Flask prototype)
- `output/llm_outputs/` - raw + parsed Ollama JSON output
- `output/logs/` - small JSON logs from individual component tests
- `output/logs/web_turns/` - one JSON log per Flask prototype voice turn
- `output/results/component_results.csv` - one row per individual component test
- `output/results/pipeline_results.csv` - one row per full pipeline run
- `output/results/summary.csv` - comparison summary

## Important

Do not commit credentials, audio files, transcripts, or generated
outputs. `.gitignore` already excludes `output/`, `input_audio/`,
`input_text/`, `models/`, `web_uploads/`, `.env`, `*.webm`, `*.wav`,
`*.mp3`, and all `*.json` files except the `config/*.example.json` schema
files.
