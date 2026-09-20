# Evaluate behavior, not only imitation

Use three separate checks:

1. **Artifact validity:** finite updated adapter tensors; portable keys; strict reload with key, shape, and value equality; conversion identity checks.
2. **Mechanical behavior:** response shape, evidence quotations, grounded state changes, abstention when inputs are absent, and completion without truncation. A JSON object alone is not a successful answer.
3. **Usefulness:** whether the response addresses the latest participant, preserves continuity, leaves people in control, adds useful detail, and avoids invented rules or facts.

Reserve whole conversation groups before tuning. Keep a development set for iteration and a separate final test set. Include counterfactual pairs that differ only in a participant's latest decision, ambiguous attribution, corrections, missing rules, long exchanges, and multi-turn continuity. Review error categories before adding examples or increasing training duration.

## HTTP benchmark case format

The benchmark takes a JSON array. Each case has `id`, `task` (`classifier`, `storyteller`, `rules`, or `player`), `prompt` chat messages, and `provenance.split` (`validation`, `test`, or `synthetic`). Optional `schema` supplies a JSON schema to the local runtime. Optional `checks` has:

- `equals`: mapping of dot-separated response paths to exact expected values. Numeric path elements select array positions.
- `nonempty`: top-level fields that must contain a truthy value.

These checks cover only the stated contract. They do not certify factual grounding or prose quality; add targeted cases and independent review for those claims. The included synthetic cases are intentionally small harness smoke checks.

Production sampling and greedy diagnostics answer different questions. The default streaming benchmark uses task-specific production sampling; `--profile greedy` provides a deterministic decoding diagnostic. Hold the case, context, seed, maximum output, schema, runtime, quantization, and adapter inventory constant when comparing candidates. Repeat across seeds for quality comparisons; record warm-up and queueing for latency comparisons.

## Review

Read `review.html` before opening `review-key.json` or `results.json`. The former displays only prompts and shuffled responses; the latter files reveal candidate identities and execution metadata. Record separate judgments for responsiveness, continuity, agency, groundedness, and usefulness. Retain disagreements and representative failures rather than reducing everything to a single score.

The original continuation is one plausible response, not unique ground truth. Completion loss measures imitation under a known target; it cannot decide whether an alternative response is more helpful. Semantic quality remains pending until reviewed. Publish results only from data and artifacts that are appropriate to release, with the actual protocol and limitations.

## Integrated three-adapter workflow

`ttrpg-pipeline run` applies the same source/label contracts used for training preparation to generated outputs. Classifier checks distinguish source-grounded event structure from exact agreement with reviewed event fields. Rules checks distinguish valid quotations, tool arguments, and missing-information decisions from the meaning of the answer. Storyteller shape checks leave semantic review pending. Shared rule excerpts across question splits are reported during preparation; they do not establish unseen-document generalization.

The pipeline stores validation and test comparisons separately. Reusing test failures to tune the next candidate changes their role to development evidence. Keep a new untouched assessment for subsequent claims. The generated demo intentionally shares simple synthetic patterns across splits and is only an integration test.
