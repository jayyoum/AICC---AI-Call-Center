# Limitations and Next Steps

## What this experiment was and wasn't

This experiment was not intended to prove the final proposed AICC
architecture. It was an implementation test and baseline study using
Llama 3.2 — a small, locally-available model — purely as a convenient
stand-in LLM to validate the pipeline. Its routing mistakes are expected at
this stage and are a useful, intended output of the test: they show
concretely where a weak single-model baseline breaks down, which is exactly
the information needed to motivate the next architecture step.

## What this test does show

1. The full modular pipeline works end to end.
2. Cloud and local speech components (STT and TTS) can be swapped freely
   without breaking the pipeline.
3. Latency and outputs can be logged automatically for later analysis.
4. Local STT/TTS are practical enough for experimentation — local STT in
   particular was faster than the cloud alternative on clean short audio.
5. A single weak local LLM is not reliable enough for final routing under
   ambiguity, out-of-scope requests, and strict route-tree compliance.
6. These weaknesses directly motivate the next research step: architecture
   design work such as hierarchy-aware routing, clarification logic, output
   validation, uncertainty detection, and selective stronger-model
   verification.

## Known limitations of this round of testing

- `route_valid` measures exact format compliance only — it is not a
  Routing Success Rate (RSR) and should not be quoted as one.
- `manual_success_label` in `pipeline_results.csv` is present but currently
  blank; no human-reviewed success rate exists yet.
- The scenario set (12 audio clips) is small and was not designed as a
  stress test; it does not yet cover systematic noise, accents, or
  adversarial phrasing.
- Only one local LLM (Llama 3.2) was tested; no comparison against other
  local or larger models exists yet.

## Next steps

- Add manual success labels to `pipeline_results.csv`.
- Separate `route_valid` from Routing Success Rate (RSR) as distinct,
  clearly-named metrics.
- Improve the LLM router prompt and schema enforcement.
- Add post-processing or validation that can detect invalid routes and
  force clarification or retry.
- Test additional local models, such as Mistral, Qwen, Gemma, or larger
  Llama variants.
- Compare weak local model baselines against proposed architecture
  variants.
- Expand from tool testing to controlled stress-test evaluation.
- Measure Routing Success Rate (RSR), Stress RSR, Stress Degradation,
  cost/latency, and error sources.
