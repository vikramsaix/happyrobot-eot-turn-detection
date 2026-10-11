# Streaming Turn-End Detection — Assessment

## Executive summary

I implemented a small, inspectable baseline that combines the current streaming ASR text hypothesis with upstream VAD silence duration and ASR-hypothesis stability. A hand-engineered feature vector feeds an L2-regularized logistic model. A conservative policy gate prevents a response when silence is under 180 ms or the transcript has been stable for under 100 ms; otherwise the model compares its endpoint score to a validation-selected threshold. The service returns `wait` or `respond`, a score, version, and reason codes through FastAPI.

This is a **working engineering prototype, not a production-validated detector**. The authored corpus is synthetic. The point of its score is to exercise deterministic training, a grouped split, threshold selection, serving, and metrics end-to-end—not to claim accuracy on real calls.

## Assumptions and scope

- Upstream components provide partial/final transcript text, a VAD-derived no-speech duration, and ASR hypothesis stability duration.
- The detector is called on streaming updates. In the proposed integration, a VAD change from silence back to speech would cancel any pending agent response; the current service does not implement orchestration or cancellation.
- English is the target language. I included a small Spanish path as a bonus demonstration; it is not a robust multilingual model.
- Missing audio prevents an honest acoustic-model experiment. The current baseline therefore accepts text and timing metadata rather than pretending to process waveform audio.
- False interruption is more costly than a modestly late response; validation threshold selection assigns false-responds 4× the cost of false-waits.

## Model and serving design

Feature groups: normalized pause and transcript-stability durations; word count; terminal/question/open punctuation; final-token indicators for conjunctions, prepositions, pronouns, and fillers; a small set of completion markers; and language ID. A compact logistic regression produces `p_end`. The policy adds minimum-pause/stability gates and reports simple reason codes. Features and model are plain Python/JSON; inference is fast and deterministic.

The FastAPI endpoint accepts a bounded request and returns a response suitable for a voice-agent orchestrator. `GET /health` exposes model version. `Dockerfile` packages a one-worker Uvicorn service with a health check. A production deployment should put the service close to ASR/VAD, use request deadlines, enforce schema/size limits, emit model versions, and scale workers based on the target platform's measured concurrency. Transcript text should not be retained in ordinary telemetry.

### Suggested agent integration

```text
LISTENING -> VAD silence -> CANDIDATE_END
  each update: score(current_text, silence_ms, stable_ms)
  speech resumes? cancel candidate -> LISTENING
  decision is wait? keep listening; enforce a maximum wait
  decision is respond? send the chosen transcript to LLM/TTS
  agent speaking? continue barge-in detection
```

The detector must be causal: only use information available at the decision time. In practice, use a short hysteresis/debounce window, tune thresholds by call type, and make barge-in cancellation reliable. Avoid blocking on a single final ASR token if the user is still speaking.

## Data and evaluation

The deterministic corpus contains **3,040 rows** from 38 authored templates, in English and Spanish. It varies pause and transcript-stability signals and adds label noise to mimic ambiguous endpoints. The exact prompt template is kept within one split: 1,920 train, 560 validation, 560 held-out test. The validation threshold minimizes `4 × false positive + false negative` **after applying the serving pause/stability gates**; the test split is not used for threshold selection.

| Held-out synthetic-test metric | Raw classifier | Served policy (with gates) |
|---|---:|---:|
| Precision | 0.854 | 0.943 |
| Recall | 0.754 | 0.754 |
| F1 | 0.801 | 0.838 |
| False-respond rate among true negatives | 0.097 | 0.034 |
| False-wait rate among true positives | 0.246 | 0.246 |
| Confusion counts (TP / FP / FN / TN) | 181 / 31 / 59 / 289 | 181 / 11 / 59 / 309 |
| Selected probability threshold | 0.58 | 0.58 |

**Interpretation:** these results indicate the code path learns the toy signal. The API's hard gates reduce false responds on this toy test while recall stays unchanged at this threshold; this is not proof of real-world benefit. The labels are generated from scenario type and timing priors, not annotated human conversations; templates and lexical cues are intentionally narrow. Even with a grouped split, synthetic train/test distributions share the same hand-authored construction process. Do not use these metrics for deployment or compare them to a real-world target. The next meaningful score requires human-labelled, temporally held-out calls.

## Latency and performance

The Docker image was built and run successfully on an Apple Silicon Mac. Health and prediction requests succeeded. The recorded local Docker benchmark used 30 warm-up requests, 1,200 sequential requests, and 1,200 requests at 20-client concurrency, including HTTP/JSON overhead:

| Mode | p50 | p95 | p99 | Throughput |
|---|---:|---:|---:|---:|
| Sequential | 0.904 ms | 1.009 ms | 1.152 ms | — |
| 20 concurrent clients | 9.088 ms | 11.818 ms | 13.394 ms | 2,134.53 req/s |

Source: [recorded Docker benchmark](../artifacts/benchmark-docker.json). The benchmark repeats one payload in a short run; it is not a sustained capacity test or deployment SLO. Service latency was below the 100 ms target in this test. Full turn-taking latency additionally includes upstream ASR/VAD, the deliberate silence/stability gates, external network and queues, and downstream LLM/TTS.

## Proposed production monitoring plan (≈1M calls/month)

The following instrumentation and rollout processes are not implemented in the current service.

1. **Instrument every decision without storing raw transcripts by default.** Emit timestamp, model/policy version, ASR/VAD version, language, pause/stability buckets, score, threshold, action, policy reason, endpoint latency, request/trace ID, and whether the user resumed speech before the agent began. Restrict any sampled transcript/audio review to approved consent, privacy, retention, and access controls.
2. **Monitor service health:** request rate/concurrency; p50/p95/p99 service and end-to-end latency; errors/timeouts; queue delay; CPU/memory; model-load failures; language mix; and missing/invalid feature rates. Alert on elevated timeouts, p95 budget burn, and sudden traffic/schema/version changes.
3. **Monitor product outcomes:** interruption/false-end rate (speech resumes immediately after agent starts), over-wait/late-response rate, user abandonment or repeated prompts, barge-in success, agent response-start latency, and human-review agreement. Break down by locale/language, call intent, acoustic conditions, ASR/VAD version, and device/channel to uncover slice regressions.
4. **Create labels from production carefully:** sample calls stratified by score/confidence, model disagreement, early user resumption, long waits, new domains, and underrepresented slices. Have trained annotators mark user turn boundaries and ambiguity; double-label a subset and track agreement. Use privacy-approved redaction and retention. Active learning makes the 1M monthly calls a valuable pool, not an automatic training set.
5. **Improve in a controlled loop:** keep speaker/call-disjoint and temporal train/dev/test sets; establish an audio/VAD and simple rules baseline; compare text-only, audio-only, and multimodal models; calibrate thresholds against interruption-vs-latency costs; shadow-run a candidate; canary by slice; monitor outcomes; roll back on a guardrail breach. Version datasets, labels, features, thresholds, models, and policies together.
6. **Watch for drift:** monitor feature and score distributions, language/domain mix, calibration and outcome rates, plus changes in ASR punctuation, partial stability, VAD, and telephony. Drift is a review trigger, not an instruction to retrain blindly.

## Limits and next model/data steps

- **Modeling:** linear decision boundary and a few hand-curated lexical cues; no waveform, pitch/energy/prosody, overlap, turn-taking context, speaker roles, discourse intent, confidence, or language-ID uncertainty. Sparse completions can be ambiguous even for a human. It has no online hysteresis state other than the upstream time features.
- **Data:** synthetic, small, narrow lexical and language coverage, hand-authored labels, and no real noise/accent/channel diversity. Threshold and probabilities are not calibrated for production.
- **Next data:** define an annotation guide for “end of user turn” versus within-turn pause/backchannel; label boundary timestamps and uncertainty; use privacy-approved real calls and controlled consent; double annotate edge cases; balance by speaker, locale, intent, overlap, noise, ASR/VAD versions; reserve speaker-disjoint temporal holdouts.
- **Next modeling:** a strong first upgrade is a calibrated tree/boosting baseline over text features, pause trajectory, ASR stability/confidence, VAD energy, and recent history. Then compare a small causal acoustic encoder or audio-text fusion model. Tune thresholds and hysteresis per latency/interrupt cost, not aggregate F1 alone.

## Discussion prompts

**Audio vs text vs multimodal?** Text is cheap, interpretable, and can use syntax, punctuation, and intent, but depends on ASR latency/error and may miss prosody. Audio captures final lengthening, breath, pitch, energy, and silence without waiting for a transcript, but costs compute and needs representative audio labels; noise, channel, accent, and overlap matter. Multimodal fusion can be stronger if trained on synchronized labels and carefully calibrated, but adds complexity and failure modes. VAD is necessary for speech/silence timing and efficient gating, but silence is not semantic completion: users pause to think, and endpointing can occur with no clean silence.

**Can a transcriber solve it?** A streaming transcriber can contribute token timing, partial/final status, confidence, punctuation, and stability, but transcription alone is not an endpoint label. Train a causal second head on prefixes aligned to annotated end-of-turn times, using transcript tokens plus duration/stability and optionally audio features; emit a calibrated turn-end score. Or fine-tune an encoder with a boundary/endpointing auxiliary head while preserving transcription. Evaluate early triggers and latency on speaker-disjoint, temporal holdouts.

**Would VAD alone work?** It is an essential low-latency signal and baseline, but cannot distinguish “I went to the store … and” from a completed sentence with the same silence. Use it as a gating/time feature, not the sole decision criterion.

## Artifacts

This repository includes runnable code, `Dockerfile`, tests, the trained model, evaluation JSON, and the recorded Docker benchmark. Run `python train.py` to regenerate the synthetic dataset and training outputs. Video and local presentation materials are managed separately.
