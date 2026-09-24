# Storyteller data and preference experiment

Four supervised runs and a subsequent DPO run completed on two 24 GB GPUs. The new preference model wrote shorter responses and received more editorial preference credit, but failed more held-out source-and-task checks than the base. **It was not promoted to the application default.** The base writer remains selected; trained candidates remain available for controlled evaluation.

## Data and review

The experiment reviewed 210 candidate decisions and kept 97 source-edited exchanges: 81 training, eight validation and eight test. Training covered 13 recorded story groups. Recorded validation and test each came from a different held-out story group. Text-based speaker attribution was checked against source quotations, not independently verified against audio. These are assistant reviews, not human gold.

Another 32 original authored training responses covered 16 two-turn scenario families. Six separate authored validation scenarios and 12 separate authored test scenarios supplied explicit facts, player choices and constraints. The real-only training set therefore had 81 responses; the mixed set had 113. The same 14 validation and 20 test cases served both conditions.

Mechanical source-span checks rejected altered or ambiguous quotations, overlaps and future response text in the input. Editorial review handled mixed voices, missing context, agency and usefulness. An initial local-model screen accepted mixed-speaker examples and was stopped; its approvals were not used as acceptance labels. The training targets and preference decisions received a separate assistant review.

Native-tokenizer audits retained every selected input and completion, with no truncation or omitted examples. The longest SFT example was 1,479 tokens within a 4,096-token budget. Real-only training supplied 4,403 completion tokens per epoch; mixed training supplied 6,452. Validation contained 838 completion tokens.

Short recorded snippets exposed an evaluation limitation: some could not support a factual-recall judgment because necessary prior information was absent. During validation, before model labels were opened, the protocol separated these examples into a format/usability stress subset. Fully specified authored scenarios supplied the primary semantic checks. Failed outputs remain in the private review; they were not discarded to improve results. Matching the recorded continuation is not the creative objective.

## Supervised comparisons

All four runs used the same 27B base, 4-bit QLoRA, rank 16 / alpha 32, two epochs, completion-only loss, FSDP2 and CPU offload. The mixed dataset received more tokens and optimizer updates, so the data comparison is **not compute-matched**. Times below include training-stage startup, but exclude the subsequent independent reload and conversion.

| Training data | Learning rate | Updates | Training minutes | Validation completion CE |
| --- | ---: | ---: | ---: | ---: |
| Base, no training | — | — | — | 1.4928 |
| 81 source-edited responses | 2e-5 | 21 | 16.3 | 1.1552 |
| 81 source-edited responses | 8e-5 | 21 | 16.4 | 1.0740 |
| 113 mixed responses | 2e-5 | 29 | 22.0 | 1.1033 |
| 113 mixed responses | 8e-5 | 29 | 21.9 | 1.0360 |

All four exports passed independent reload comparisons of all 672 adapter tensors and serving-format conversion. Sampled peak device memory across these training stages ranged from 19,095 to 21,713 MiB; polling may miss brief higher peaks.

The higher-rate mixed run was added after validation exposed an interaction worth testing: the lower-rate mixed model produced four empty answers in 14 cases, while the higher-rate real-only model avoided empties but had weaker authored-scenario results. The higher-rate mixed model produced 14 usable completions and passed five of six authored validation cases, matching the base on those six checks. It was selected as the DPO starting point, not as a replacement default. Final test responses had not yet been generated.

Diagnostic changes to presence penalty and a generic nonempty-response reminder did not reliably fix the failures; neither changed production defaults. The rendered training and serving prompt text matched exactly in a separate template check.

## Preference training

Review produced 64 training pairs and 11 validation pairs, with seven ties or exclusions. Positives included 30 source-based edits, 16 authored references and 29 sampled or edited alternatives across both splits. All chosen responses carried an explicit validity judgment. The labels were assistant-generated and reviewed, with no independent human calibration.

Alternatives came from the lower-rate mixed model, with base-model fallbacks when both alternatives were unusable. These were **off-policy pairs** for the selected higher-rate checkpoint. DPO recomputed reference probabilities from a frozen copy of that selected SFT adapter. It did not use the sampling model or the unadapted base as its reference.

The full run used one process with model placement across both GPUs, NF4 quantization, BF16 computation, activation checkpointing, batch size one, four-step accumulation, learning rate 5e-6, beta 0.1 and one epoch: 16 optimizer updates. All pair prefixes and completions matched the native token hashes; the longest was 2,150 tokens. Reference precomputation and the full run took about 17.7 minutes, including 12.1 minutes reported by the training loop. Reload and conversion were separate.

All 672 trainable adapter tensors changed and remained finite. The frozen reference was unchanged. Independent reload compared all 672 exported tensors, and GGUF conversion passed. Peak PyTorch allocated memory was 17.9 and 16.4 GiB; sampled whole-device peaks were 23,043 and 17,513 MiB. Those measurements include different allocations and are not interchangeable.

The initial smoke attempt failed during checkpoint metadata creation because a local C++ runtime conflicted with the environment's SQLite dependency. A process-level library-loading correction fixed it. A fresh two-step smoke passed, and the full run restarted from the selected SFT checkpoint; smoke updates were not carried forward. Serving services were restored after both the failure and the completed run.

DPO validation loss was 0.5656, with positive chosen-versus-rejected reward margins on eight of 11 validation pairs. Completion CE on the independent SFT validation responses was 1.0418, slightly above the selected SFT's 1.0360. Neither statistic establishes story quality.

## Locked test comparison

The base, [previous response adapter](storyteller-results.md), selected mixed SFT and DPO model received identical application prompts, schema, production sampling and a 700-token cap. Twenty cases at two seeds produced 160 outputs. Generation order and review labels were randomized separately. Ratings were saved before opening either seed's model key.

The raw comparison used the common `NarrationAnswer` schema. The complete application additionally routes factual answers to the base with `DirectAnswer`, and constrains creative replies with `SceneAnswer` (an empty direct-answer field). The raw comparison therefore evaluates writers under a shared contract; it is not a measurement of the application's routing or final edited output.

Semantic results below cover **12 independent authored scenarios repeated at two seeds**, not 24 independent scenarios. The other eight inputs came from one held-out recorded story and remained a separate sparse-context stress subset. Review was by one assistant and was not calibrated against human judgments. Small differences are not evidence of general superiority.

| Measure | Base | Previous response adapter | Selected SFT | DPO |
| --- | ---: | ---: | ---: | ---: |
| Complete, nonempty responses / 40 | 40 | 40 | 40 | 40 |
| Authored strict passes / 24 | 18 | 19 | 15 | 15 |
| Authored voice points / 48 | 25 | 34 | 32 | 33 |
| Editorial preference credit / 24 comparisons | 2 | 6.5 | 6.5 | 9 |
| Median raw generation seconds, all 40 | 5.16 | 4.19 | 3.02 | 2.89 |

A strict pass required answering the current need while preserving established facts, uncertainty, player agency and private knowledge. Minor staging and consistent new world details were permitted; consequential player choices and protected unknowns were not. Voice used a separate 0–2 scale. Preference allowed ties and selected usable responses that passed the semantic criteria; it is not a model win rate against each opponent. Latency reflects these response lengths and serving conditions, not full application latency.

The new models remained vulnerable to assigning NPC answers to player speakers, deciding unresolved facts, and omitting part of a request. DPO improved some concise replies but did not fix those reliability problems. The previous adapter's one-pass advantage over the base on this small test does not erase its failures in the earlier study or justify promoting it automatically.

No test response was used to tune another model or change the production prompt in this experiment. These now-exposed failures can inform future development, but any subsequent generalization claim requires fresh held-out scenarios.

## Application replay

The candidate also ran through Story Copilot using eight retained authored-audio transcriptions. All 58 workflow assertions passed, covering import idempotency, duplicate suppression, observed inventory, verified affordability and the separation of suggestions from recorded speech or player publication. A further real-generation check rejected an in-flight draft after its source changed. These are operational checks, not a blanket quality score.

Two of the eight replies became guarded factual fallbacks, losing useful NPC dialogue. The replay exposed an application defect: uncertainty detection recognized ASCII contractions but missed typographic apostrophes. The fix recognizes both while preserving original source text and offsets. Regression tests cover unknown-to-definite errors and invented player speech as well as the valid contractions. Replaying the two original drafts through the actual model reviewer after the fix retained both replies. This is a development repair, not a revised held-out model score.

Four retained natural-audio chunks passed all 16 record/publication assertions. Two produced useful clarifications of incomplete speech; two returned generic guarded notes that were not useful assistance. Recognition and diarization were reused, not rerun or independently scored. This does not establish natural-speech WER or DER.

Four additional rule controls compared the base, selected SFT and DPO model: two new checks requiring exact citations, an already resolved check, and ordinary conversation with an irrelevant rule. All 12 outputs preserved the expected check-field/citation behavior. These inputs were used during preference construction, so this is a development regression result, not held-out generalization or a complete prose-quality score.

Before the punctuation repair, median full assistance time was 46.0 seconds for the authored replay (range 26.6–72.8), and 46.5 seconds for the natural replay (18.3–57.8). These sequential replay times exclude speaking time, fresh transcription and queueing. They include routing, state/tool work and review, and should not be compared directly with raw writer generation time. The two targeted reviewer replays took 12.3 and 15.8 seconds; they do not establish a new full-workflow median.

Source text, response traces, recordings and trained weights remain private. The public repository provides the [review and training procedure](story-preferences.md) for another dataset.
