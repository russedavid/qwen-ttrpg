# Agent RL development results

20 September 2026. A small decision policy learned to retrieve current evidence, answer factual questions, and request missing information. On this authored suite, 100 GRPO updates after a supervised warm-up outperformed both the warm-up alone and an additional 100 supervised updates. The RL extension took about twice as long as the extra supervised training.

## Controlled comparison

All candidates used [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, the same BF16 base, read-only evidence tools, prompt/template, context budget, and evaluation decoding. Adapters updated 15,237,120 parameters with rank 8 and alpha 16; the base, vision encoder, embeddings, and output head remained frozen.

The training snapshot contained 144 procedural scenarios and 288 supervised decision examples. A separate 24-case validation set selected the RL checkpoint: 20 updates passed 21/24; 100 updates passed 23/24. The selection was recorded before test evaluation. An extra-supervised-training control was fixed before reading independent test results, with no additional hyperparameter search.

Independent assessment used 24 new cases from the training task families and 12 cases with unseen structures: two successive corrections, another character's later update, and a revised rule containing both obsolete and effective costs. Each case was run with generation seeds 42 and 314: **72 attempts over 36 distinct scenarios per candidate**, not 72 independent scenarios. No test or transfer feedback was used to revise the policy, reward, or environment.

| Candidate | Familiar structures, 48 attempts | Unseen structures, 24 attempts | Combined task passes | Correct conclusion fields |
| --- | ---: | ---: | ---: | ---: |
| Base model | 17 | 7 | 24/72 (33.3%) | 37/72 |
| 40 supervised updates | 33 | 10 | 43/72 (59.7%) | 48/72 |
| 140 supervised updates | 40 | 16 | 56/72 (77.8%) | 56/72 |
| 40 supervised + 100 RL updates | 46 | 20 | **66/72 (91.7%)** | **67/72** |

A task pass requires the correct structured conclusion and an accepted set of source identifiers actually observed by the agent. Correct conclusion fields are reported separately because a valid answer can still fail the citation contract. Free-form prose quality is not scored by this checker.

Across matched case/seed pairs, RL gained 23 passes over the short supervised baseline and 10 over the longer supervised control, with no pass-to-fail regressions against either in this run. This is a small, deliberately constructed study; it does not establish deployment accuracy or general superiority of RL.

## What changed in behavior

The longer supervised control still failed all eight ordinary activation attempts and all eight revised-cost attempts. Review of those 16 failures found that it inspected the starting sheet without retrieving the available inventory correction. It sometimes asked for information already available in the dialogue.

RL more often chose the history lookup that supplied the current balance. Its six remaining failures were inspected against the complete sources and tool traces:

- One used a starting inventory of six after a correction to zero and incorrectly allowed activation.
- Four attempts, covering two distinct revised-rule cases, retrieved the right sources but misapplied the effective costs and current balances.
- One calculated the correct result but cited the current revision identifier rather than the required message identifier. This was a citation-contract failure, not an invented source or a wrong numerical answer.

Successful RL clarification questions were also inspected and targeted the stated missing balance or rule. This was assistant source/trace review, not a calibrated human judgment panel. All raw responses and reviewer notes remain available in private review artifacts. The score does not establish storytelling quality.

## Cost and hardware

Training used one RTX 3090 24 GiB. The existing 27B service remained on the other card; its idle speech worker was paused during training/evaluation and restored afterward. The experiment used a separate Python environment and separate checkpoints.

| Run | Measured training time |
| --- | ---: |
| Shared supervised warm-up, 40 updates | 3.58 min |
| Extra supervised training, 100 updates | 8.65 min |
| RL extension, 100 updates | 17.58 min |

These times include model initialization and export for each training launch. The RL extension includes the first 20 updates and the resumed 80-update continuation. The extra-supervised control matches the number of added optimizer updates, **not compute, generated tokens, or exposure to alternative trajectories**.

The RL run peaked at about 11.70 GiB of Torch-allocated memory and 23.15 GiB reserved by the allocator. Reserved memory includes cached blocks. Both figures matter: this demonstrated a working run on a 24 GiB card, not a guarantee for a smaller card or a longer context.

Combined median end-to-end evaluation times were 2.46 seconds for base, 2.62 for the short supervised baseline, 2.21 for the longer supervised control, and 2.20 for RL. Model loading is excluded; these are single-request local observations, not a concurrency benchmark. The longer supervised control and RL both averaged 1.25 evidence calls per attempt, while choosing different sources.

## Training and verification

The learner used TRL 1.9.0's GRPO trainer with DAPO loss normalization, beta 0, four attempts per task, gradient microbatch 1, accumulation 4, and a constant learning rate of 0.00001. The supervised rate was 0.0001. Context was capped at 3,072 tokens, with at most 160 new tokens per decision and three evidence calls followed by a terminal decision. Evaluation used temperature 0.7, top-p 1, no top-k restriction, and identical budgets for every candidate.

Actual model tokens were preserved across tool turns using TRL's training template and an exact prefix check. External evidence was masked from the policy loss. In the first 20-update run, the traces contained 4,200 generated decision tokens and 15,842 external tokens excluded from that target. A batched tensor test checked independent episode identity, padding removal, and the masks after some trajectories ended early.

All 400 adapter tensors changed in each training stage and remained finite. Reload checked keys, shapes, and equality to the saved values. The 100-update RL run resumed the step-20 checkpoint with optimizer and RNG state; the longer supervised control resumed its step-40 checkpoint. Dataset hashes, environment source hash, decoding settings, adapter hashes, timing, and full trajectories were retained.

## Limits and reproduction

The rewards test narrow structured outcomes. The supervised data supplies one authored path per task, while RL explores alternative paths and receives outcome feedback. The observed gain therefore includes differences in training signal and trajectory coverage. More diverse supervised demonstrations, different hyperparameters, or a compute-matched study could change the comparison.

The `unnecessary_calls` counter measures calls beyond an oracle-authored reference path; it can include reasonable exploration. It is a diagnostic, not proof that a lookup was wasteful. Revised-rule interpretation remains an observed weakness even when citations are valid. The experimental adapter has not replaced the live copilot's configured model.

The [reproduction guide](agent-rl.md) covers preparation, training, checkpoint continuation, independent evaluation, and side-by-side trace review. The source baseline is Story Copilot `05cac8a` and this toolkit's training implementation `dc80a56`. The repository provides the method and original fixture generator; training outputs and weights remain outside Git.


## Serving integration and quantization check

The trained policy was subsequently exported for the live Story Copilot. This is a separate development check, not a replacement for the reserved 72-attempt comparison above. The trained adapter was unchanged.

On 24 fresh authored development tasks with greedy decoding, the BF16 and Q8_0 serving copies produced the same structured conclusions and citation lists in all 24 cases. Both passed 23/24 complete contracts; the same remaining numerical error failed both. Median complete tool-trajectory time was 1.37 seconds for BF16 and 1.04 seconds for Q8_0 in this run. These small, sequential measurements do not establish a general speed or accuracy advantage.

The check caught a serialization failure before promotion: an alphabetically property-ordered JSON Schema grammar prevented the trained `action, value, sources` continuation from emitting citations after `value`. Exact-contract passes were initially 0/24 despite many correct values. Using a JSON-object grammar followed by full schema validation restored the expected behavior. The original trained contract also did not require the live application's intent field; the bridge derives missing labels only from typed evidence fields. General narrative and social decisions retain the main planner because they were outside the RL curriculum.

The Q8_0 base occupies approximately 4,386.5 MiB on disk. With the policy and speech worker sharing one 24 GiB GPU, two synthesized clips of approximately 35 and 33 seconds completed while policy inference overlapped speech processing. The sampled GPU-memory peak was 21,177 MiB. Warm speech processing took 2.36 and 2.06 seconds; model loading added 6.88 seconds to the first clip. The larger writing model remained on the other GPU. These are bounded coexistence checks with clean synthetic voices and a known speaker-count prior, not proof of capacity for every context, batch size, or audio workload.

The live application never treats the policy's terminal numbers as verified rulings. It checks source-bound calculator inputs, rejects explicitly superseded values, and displays deterministic results. See [live integration](https://github.com/russedavid/story-copilot/blob/main/docs/live-planner.md) and the [audio-workflow evaluation](https://github.com/russedavid/story-copilot/blob/main/docs/evaluation.md#audio-through-the-full-workflow).
