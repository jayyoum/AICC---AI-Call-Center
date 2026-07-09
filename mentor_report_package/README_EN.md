# AICC Voice Pipeline — Tool Test Report

Status: preliminary tool/pipeline validation result, not a final research result.

## What this is

This package documents a baseline tool-setup test of a modular AICC voice
pipeline. The goal of this round of testing was to confirm that the
pipeline components work end to end and to get an early, honest look at
where a weak local LLM baseline (Llama 3.2) breaks down. It is not a
demonstration of the final proposed architecture and it is not a
performance claim.

## What's in this package

```
README_EN.md / README_KR.md                             This file, in English and Korean
METHODS_EN.md / METHODS_KR.md                            What was tested and how
RESULTS_INTERPRETATION_EN.md / _KR.md                    What the numbers mean
LIMITATIONS_AND_NEXT_STEPS_EN.md / _KR.md                Known weaknesses and what's next
results/
  component_results.csv                                  One row per individual STT/TTS/LLM component test
  pipeline_results.csv                                    One row per full pipeline run (48 runs)
  summary.csv                                              Aggregated latency/error stats per pipeline combination
outputs_for_inspection/
  transcripts/                                             STT transcripts for manual spot-checking
  llm_outputs/                                              Raw + parsed Ollama JSON output for manual spot-checking
config/
  routes.json                                              The route tree used for LLM routing decisions
  manifest.csv                                              The scenario manifest used to drive batch testing
code_snapshot/
  src/, test_stt.py, test_tts.py, test_llm.py,
  run_pipeline.py, run_batch.py, compare_results.py,
  requirements.txt                                          Code needed to understand or reproduce the test
```

## How the test was run

1. Each component (Google STT, local faster-whisper STT, Google TTS, local
   Piper TTS, Ollama/Llama 3.2) was first tested individually.
2. The scenarios in `config/manifest.csv` were then run through
   `run_batch.py` across all four STT+LLM+TTS pipeline combinations.
3. Every run's timing, transcript, routing decision, and response text was
   logged automatically to `results/pipeline_results.csv`.
4. `compare_results.py` aggregated those rows into `results/summary.csv`.

## Which files to look at first

1. **RESULTS_INTERPRETATION_EN.md** — the actual findings, in plain language.
2. **results/summary.csv** — the latency/error numbers behind those findings.
3. **LIMITATIONS_AND_NEXT_STEPS_EN.md** — why this is a baseline, not a conclusion.
4. **outputs_for_inspection/** — spot-check a few transcripts and LLM outputs directly if you want to see raw model behavior.
5. **results/pipeline_results.csv** — the full row-by-row data if you want to dig deeper into a specific scenario.

## What's intentionally excluded

- Generated response audio files are **not** included in this package by
  default (they would make the zip much larger and aren't needed to review
  the results). Ask if you'd like a separate audio sample package.
- No credentials, no `.venv/`, no downloaded model files (`models/`), no
  `__pycache__/`, no `.DS_Store`.

## One-line summary

The pipeline works end to end and logs cleanly; the local LLM baseline
(Llama 3.2) is not yet reliable enough at exact route-schema compliance,
which is expected for a first baseline and motivates the next round of
architecture work rather than being a final result.
