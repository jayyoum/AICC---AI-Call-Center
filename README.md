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

`app.py` accepts optional flags:

```
python app.py --host 0.0.0.0 --port 5000   # bind all interfaces (for phone/LAN access)
python app.py --debug                       # enable auto-reload + in-browser debugger
```

- `--host` default `127.0.0.1`; use `0.0.0.0` to let a phone or other device
  on your network connect.
- `--port` default `5000`.
- `--debug` is off by default. Do **not** enable `--debug` while binding
  `0.0.0.0` (it would expose the Werkzeug debugger to your network).

### 6. Open the prototype in your browser

```
http://127.0.0.1:5000        # developer dashboard (also at /dev)
http://127.0.0.1:5000/customer   # simple customer call screen
```

On the **developer dashboard**, choose your STT/TTS providers, prompt mode,
and router mode, click **Start Recording** (or the continuous tab), speak,
and the transcript, routing decision, latency, and spoken response appear.

Generated response audio is saved under `output/web_audio/`, and one JSON
log per turn is saved under `output/logs/web_turns/`. Raw browser
recordings are saved temporarily under `web_uploads/` (not committed to
version control).

## Using your phone as the customer device (Mac hosts, phone speaks)

The Mac runs the Flask backend; the phone opens a simple customer call
screen (`/customer`) and acts as the microphone and speaker. STT, routing,
and TTS all still run on the Mac.

### 1. Run the server on all interfaces (on the Mac)

```
python app.py --host 0.0.0.0 --port 5000
```

### 2. Find your Mac's local IP

```
ipconfig getifaddr en0        # Wi-Fi (try en1 if en0 is empty)
```

### 3. Open the customer UI from the phone (same Wi-Fi)

```
http://<MAC_LOCAL_IP>:5000/customer
```

Then tap **Start Call**, allow the microphone, and talk. On silence the
phone sends each utterance to the Mac and plays back the agent's reply, then
resumes listening. Tap **End Call** to stop.

### Important: microphone needs HTTPS on phones

Most phone browsers only allow microphone access over **HTTPS** (or on
`localhost`). Plain `http://<ip>:5000` will typically **not** grant the mic
on a phone. The app does **not** force HTTPS itself — instead, use an HTTPS
tunnel for phone testing:

```
ngrok http 5000
```

Then open the HTTPS URL ngrok prints, with `/customer`:

```
https://<your-ngrok-subdomain>.ngrok-free.app/customer
```

(Install ngrok yourself from https://ngrok.com — it is not installed
automatically.) A plain-HTTP LAN address is fine for quick tests from
another computer, or from the Mac itself at `http://127.0.0.1:5000/customer`.

### Mobile playback note

Phone browsers often block audio autoplay until the user interacts. The
customer UI unlocks audio when you tap **Start Call**; if a reply still
can't autoplay, a **"Tap to play agent response"** button appears — tap it
to hear the agent, and listening resumes afterward.

## Experimental Streaming STT Mode

The customer UI has an **experimental** capture mode that streams microphone
audio to the Mac while you speak and shows live interim/final transcripts
via Google Cloud streaming recognition, instead of uploading one blob per
utterance. It is opt-in — the stable "Continuous (stable)" blob mode remains
the default and the fallback.

Requires the extra dependencies (already in `requirements.txt`):

```
pip install -r requirements.txt      # adds flask-socketio, simple-websocket
```

### 1. Run the server (on the Mac)

```
python app.py --host 0.0.0.0 --port 5000
```

### 2. Start Ollama

```
ollama run llama3.2
```

### 3. Make sure Google ADC is configured

Streaming STT uses Google Cloud (there is no local streaming STT), so
Application Default Credentials must be set up:

```
gcloud auth application-default login
```

### 4. Open the customer UI from your phone

Same Wi-Fi (local network):

```
http://<MAC_LOCAL_IP>:5000/customer
```

### 5. HTTPS note (required for a phone mic)

Phone microphone access almost always requires **HTTPS** (or localhost). For
phone testing, use an HTTPS tunnel:

```
ngrok http 5000
```

then open:

```
https://<ngrok-url>/customer
```

(Install ngrok yourself; it is not installed automatically.) The streaming
mode also loads the Socket.IO browser client from a CDN, so the phone needs
internet access.

### How streaming mode works

1. The browser captures mic audio, downsamples it to **16kHz mono Int16
   PCM** in JavaScript, and sends chunks over Socket.IO (`audio_chunk`).
2. The Mac feeds those chunks into a **Google streaming recognizer** and
   emits `stt_interim` / `stt_final` transcript events back to the browser.
3. Interim text appears live; on a **final** transcript the browser calls
   `POST /api/text-turn` — which **skips STT** (it already happened) and runs
   the existing pre-router / LLM routing + TTS — then plays the reply.
4. After the reply finishes, a fresh listening stream starts automatically.

Socket.IO events: client emits `stt_start`, `audio_chunk`, `stt_stop`;
server emits `stt_started`, `stt_interim`, `stt_final`, `stt_stopped`,
`stt_error`. Streaming turn logs are written to `output/logs/streaming_stt/`.

### Fallback

If streaming STT fails (no internet for the Socket.IO client, Google ADC not
configured, an unsupported browser, etc.), switch the toggle back to
**Continuous (stable)** — the original MediaRecorder + VAD + `/api/voice-turn`
flow is unchanged and still works.

## Experimental Streaming TTS Mode

By default the agent's reply is synthesized to a **file** and then played
(stable, and the only path that supports local Piper). There is also an
**experimental** streaming-TTS mode where the backend streams Google TTS
audio chunks to the browser as they are produced, instead of waiting for the
whole file.

- Google streaming TTS is a **preview-style** feature and only works with
  compatible **Chirp 3 HD** voices (default here:
  `en-US-Chirp3-HD-Charon`). It is Google-only — there is no local streaming
  TTS.
- The **stable fallback remains file-based TTS** (local Piper or Google
  file TTS). Streaming TTS never replaces it.

### Setup

```
pip install -r requirements.txt
gcloud auth application-default login
gcloud services enable texttospeech.googleapis.com
```

### Run + use

```
python app.py --host 0.0.0.0 --port 5000     # (or --port 5001, see AirPlay note)
```

Open `/customer` on your phone (HTTPS/ngrok for the mic — see the streaming
STT section). Then:

1. Choose **Streaming STT (experimental)** capture mode.
2. In **Advanced / debug**, set **TTS output = Streaming (experimental)**.
3. Tap **Start Call** and speak.

### How it works (streaming STT + streaming TTS together)

1. Mic audio streams to the Mac; interim/final transcripts come back live.
2. On a **final** transcript the browser calls **`POST /api/route-text`**
   (routing only — no STT, no TTS) to get `response_to_customer`.
3. The browser emits **`tts_stream_start`** with that text; the backend runs
   Google streaming synthesis and emits **`tts_audio_chunk`** (base64 PCM)
   events, then **`tts_stream_done`**.
4. The browser wraps the streamed PCM into a WAV Blob and plays it
   ("streamed transport, buffered playback"), then resumes listening.

Socket.IO TTS events: client emits `tts_stream_start`; server emits
`tts_stream_started`, `tts_audio_chunk`, `tts_stream_done`,
`tts_stream_error`. Streamed audio is also saved to `output/web_audio/` (a
`_tts_stream.wav`) and timings are logged to `output/logs/streaming_stt/`.

### Fallback

If streaming TTS fails (Chirp 3 HD voice not enabled, TTS API not enabled,
an unsupported browser, etc.), the UI shows a readable error and
automatically falls back to file-based TTS via `/api/text-turn` for that
turn. You can also switch **TTS output** back to **File (stable)** at any
time. `/api/text-turn`, `/api/voice-turn`, streaming STT, local Piper, and
Google file TTS are all unchanged.

## Session logging

Every call made through the web UI is logged as a **session**, and every
utterance inside it as a **turn**, so you can reconstruct exactly what
happened without re-running anything.

```
output/logs/sessions/
  <session_id>/
    session_events.jsonl          call-level events
    turns/
      <turn_id>_events.jsonl      step-by-step events for one utterance
      <turn_id>_summary.json      one-file summary of that utterance
```

The browser creates a `session_id` when you tap **Start Call**, sends it with
every `/api/voice-turn`, `/api/text-turn`, `/api/route-text` request and every
Socket.IO streaming event, and clears it on **End Call** (so the next call gets
a fresh id). It's shown under **Advanced / debug** in the customer UI, and the
server also returns `session_id` / `turn_id` in every API response.

Typical events per turn: `voice_turn_received`, `audio_saved`,
`audio_converted`, `stt_started`, `stt_finished`, `routing_started`,
`routing_finished`, `tts_started`, `tts_finished`, `voice_turn_completed`
(plus `*_error` variants). Streaming adds `streaming_stt_started`,
`streaming_stt_first_interim`, `streaming_stt_final`, `streaming_stt_stopped`,
`streaming_tts_started`, `streaming_tts_first_chunk`, `streaming_tts_done`,
and `streaming_error`.

Logs are sanitized before writing: credential-looking keys are redacted and
long strings / audio blobs are truncated, so no secrets or raw audio end up in
the logs. Logging failures never break a live call.

## Inspecting a session

```
python inspect_session.py --session-id latest
python inspect_session.py --list                       # show available sessions
python inspect_session.py --session-id s_9f2a1c4b77de
python inspect_session.py --session-id latest --events  # full event timeline
```

Prints a per-turn timeline: transcript, routing source/rule, predicted
route/action, agent response, latency breakdown, and any errors.

## Replaying transcripts without STT/TTS

`benchmark_transcripts.py` re-routes **saved transcripts** — no audio, no STT,
no TTS — so you can compare router modes, prompt modes, and models quickly.
It measures routing latency, route validity, and route normalization only
(**not** semantic correctness).

```
python benchmark_transcripts.py \
  --transcript-dir output/transcripts \
  --routes config/routes.json \
  --router-modes llm_only pre_router \
  --prompt-modes compact_v2 full \
  --model llama3.2
```

Other input sources:

```
python benchmark_transcripts.py --text "I lost my debit card." --routes config/routes.json --router-modes pre_router
python benchmark_transcripts.py --csv output/results/router_mode_benchmark.csv --routes config/routes.json
```

Writes `output/results/transcript_replay_benchmark.csv` and
`output/results/transcript_replay_summary.csv`. `pre_router` runs need nothing
external; `llm_only` (and pre-router fall-through) calls your local Ollama.

## Generating the metrics report

```
python generate_metrics_report.py
```

Collects whichever summary CSVs exist (`prompt_mode_summary.csv`,
`model_summary.csv`, `router_mode_summary.csv`,
`transcript_replay_summary.csv`) into one static page with tables and
automatically-derived takeaways (fastest total latency, fastest routing step,
best/worst `route_valid_rate`, highest `pre_router_handled_rate`).

Open:

```
output/reports/metrics_report.html
```

It is plain HTML + CSS — no server, no JavaScript frameworks, no internet.

## Health check endpoint

```
curl http://127.0.0.1:5000/api/health
curl "http://127.0.0.1:5000/api/health?check_ollama=true"
```

Returns server status, whether `config/routes.json` loaded, the Ollama URL,
whether the output dirs exist, and whether streaming STT/TTS are available.
It makes **no** network calls unless you pass `?check_ollama=true`, which adds
a lightweight Ollama `/api/tags` probe.

## Where outputs are saved

- `output/transcripts/` - STT transcripts
- `output/audio/` - synthesized response audio (CLI scripts)
- `output/web_audio/` - synthesized response audio (Flask prototype)
- `output/llm_outputs/` - raw + parsed Ollama JSON output
- `output/logs/` - small JSON logs from individual component tests
- `output/logs/web_turns/` - one JSON log per Flask voice turn (developer UI and phone customer UI; the customer turns are logged with `input_mode="customer_continuous"`)
- `output/logs/streaming_stt/` - one JSON log per streaming-STT text turn (`/api/text-turn`)
- `output/logs/sessions/<session_id>/` - per-call session + turn event logs and turn summaries
- `output/reports/metrics_report.html` - static metrics report from the benchmark summaries
- `output/results/component_results.csv` - one row per individual component test
- `output/results/pipeline_results.csv` - one row per full pipeline run
- `output/results/summary.csv` - comparison summary

## Important

Do not commit credentials, audio files, transcripts, or generated
outputs. `.gitignore` already excludes `output/`, `input_audio/`,
`input_text/`, `models/`, `web_uploads/`, `.env`, `*.webm`, `*.wav`,
`*.mp3`, and all `*.json` files except the `config/*.example.json` schema
files.
