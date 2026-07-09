# Results Interpretation

This is a plain-language read of `results/summary.csv` and
`results/pipeline_results.csv`. It is a progress report on a baseline test,
not a final performance claim.

## Technical success

The toolchain worked successfully overall.

- Component-level tests had no errors.
- Full pipeline tests had only one logged error out of 48 runs. That one
  error occurred because the LLM produced no `response_to_customer`, so the
  TTS step was skipped. This is a useful LLM-output robustness issue rather
  than a setup or integration failure.

## Latency summary

From `summary.csv`:

| Pipeline combination | Runs | Avg total (s) | Avg STT (s) | Avg LLM (s) | Avg TTS (s) | Errors |
|---|---|---|---|---|---|---|
| Google STT + Google TTS | 12 | 8.489 | 2.101 | 5.427 | 0.961 | 0 |
| Google STT + Local TTS | 12 | 5.689 | 2.079 | 2.760 | 0.851 | 1 |
| Local STT + Google TTS | 12 | 4.305 | 0.590 | 2.746 | 0.969 | 0 |
| Local STT + Local TTS | 12 | 6.242 | 0.603 | 4.575 | 1.065 | 0 |

## Key latency interpretation

- Local faster-whisper STT was much faster than Google STT on these short,
  clean recordings: Google STT averaged about 2.1 seconds, local
  faster-whisper averaged about 0.6 seconds.
- TTS latency was similar between Google TTS and Piper, usually around 0.85
  to 1.1 seconds.
- The largest and most variable bottleneck was the LLM step.
- The fastest full pipeline was Local STT + Llama 3.2 + Google TTS,
  averaging 4.305 seconds. This does not mean it is the best final system —
  it only means that in this small setup, local STT was fast and the local
  LLM dominated total latency regardless of which STT/TTS was paired with
  it.

## STT interpretation

Both Google STT and faster-whisper produced usable transcripts in most
cases. Some differences were observed:

- Google STT often added cleaner punctuation and normalized numbers, such
  as "July 15th" and "4291."
- Local faster-whisper sometimes transcribed number sequences as separated
  digits, such as "4, 2, 9, 1."
- Local faster-whisper once transcribed "card" as "car" in the fraud
  scenario, but the meaning was still close enough that a robust LLM should
  often recover.

Overall, STT was not the main bottleneck in this clean test set. The larger
issue was LLM routing consistency.

## LLM routing interpretation

Llama 3.2 successfully produced route decisions and responses, but it
showed limitations as a single weak local baseline model. Observed issues:

- It sometimes chose the correct general category but gave a final route
  that did not exactly match the route-tree format.
- It sometimes returned sub-route labels like "Failed Transfer" instead of
  the full allowed path "Transfers and Payments > Failed Transfer."
- It sometimes overused clarification.
- It sometimes failed out-of-scope handling, especially for the
  flight-booking-with-bank-points scenario.
- It sometimes routed ambiguous cases inconsistently across different
  STT/TTS combinations, even when the transcript was very similar.

**Important interpretation:** `route_valid` is not the same as actual
routing success. It only checks whether the returned route label exactly
matches the allowed route-tree format. Some outputs were behaviorally close
but marked `route_valid=false` because the final route was not formatted
exactly. This is useful because it shows a need for stricter output
control, but it should not be interpreted as final Routing Success Rate
(RSR).

## route_valid summary

From `pipeline_results.csv`:

- 25 outputs were `route_valid=true`.
- 23 outputs were `route_valid=false`.

This indicates that the local LLM often understood the domain but did not
reliably follow the exact route schema. This motivates stronger prompt
constraints, output validation, post-processing, or a more robust
weak/strong model architecture.

## Scenario-level observations

- Clean lost-card and fraud cases were generally understood.
- Failed transfer, locked account, and scheduled transfer cases were often
  semantically correct but sometimes failed `route_valid` due to formatting
  mismatch.
- Vague card issue produced mixed behavior: some runs correctly asked for
  clarification, while others treated it as out-of-scope or routed too
  confidently.
- Multi-intent lost-card-plus-fraud cases were inconsistent. Some runs
  correctly prioritized fraud/security, while others focused too much on
  lost-card replacement.
- Out-of-scope handling was weak. The flight-booking-with-bank-points
  scenario often did not produce the desired
  "Out of Scope > Unsupported Service" behavior.
- Spoken disfluency was handled reasonably at the STT level, but LLM
  routing still varied.
