# Methods — What We Tested

This was a tool setup and baseline pipeline test, not a controlled research
evaluation. The purpose was to confirm the toolchain works and to get an
early look at local LLM routing behavior.

## Components tested

- Google Cloud Speech-to-Text (cloud STT)
- Local faster-whisper Speech-to-Text (local STT)
- Ollama with Llama 3.2 as the local LLM (routing + response generation)
- Google Cloud Text-to-Speech (cloud TTS)
- Local Piper Text-to-Speech (local TTS)

## Test volume

- 39 component-level tests total (individual STT / TTS / LLM calls).
- 48 full pipeline tests total (audio in, spoken response out).
- 12 audio scenarios, each run through 4 pipeline combinations (12 × 4 = 48).

## Pipeline combinations

Every scenario was run through all four combinations of STT and TTS
provider, always paired with the same local LLM (Llama 3.2) for routing:

1. Google STT + Llama 3.2 + Google TTS
2. Google STT + Llama 3.2 + Local Piper TTS
3. Local faster-whisper STT + Llama 3.2 + Google TTS
4. Local faster-whisper STT + Llama 3.2 + Local Piper TTS

## Routing reference

LLM routing decisions were checked against the route tree in
`config/routes.json`. See `RESULTS_INTERPRETATION_EN.md` for what "checked
against" (`route_valid`) does and does not mean.

## Scope note

Llama 3.2 was used here only because it is a convenient, already-installed
local baseline model — it is not the final architecture we intend to
propose. Its mistakes are expected at this stage and are treated as useful
signal for what the eventual architecture needs to handle, not as a
verdict on the pipeline design itself.
