# Improving a responsive storyteller

The objective is a useful next response in an ongoing interaction: answer the participants, preserve continuity and agency, and offer something the facilitator can use. Reproducing the sound of a transcript is insufficient.

The [completed four-run SFT and DPO experiment](storyteller-preference-results.md) applies this sequence. It improved some editorial preferences without earning promotion over the base writer. The guidance below remains an experimental method, not a promise that more training improves every measure.

## Data first

The [previous pilot](storyteller-results.md) improved voice preference without beating the base on strict source-and-task checks. Its exposed failures now belong to development regressions. Improvements against those cases are not fresh held-out results.

Audit real exchanges for transcription errors, wrong speaker boundaries, incomplete responses, insufficient context, missing rules and weak writing. Repair the earliest fault. Polishing a mixed-speaker target can teach the wrong behavior more fluently. The [workshop](story-workshop.md) preserves originals and records boundary repairs and editorial targets separately.

A working experiment size is 100–200 carefully reviewed responses initially, expanding toward 400–800 if review yields enough useful variety. These are planning targets, not established minimums. More turns from one story do not create independent story groups. Balance sources and scene functions before increasing volume.

Include discoveries, NPC motives, conflict, consequences, deliberate pauses, scene transitions, several participants contributing, corrections, and uncertainty that actually needs clarification. Preserve quiet and tense passages. Remove production chatter and redundant repetition while retaining specificity and pacing. A damaged phrase should be completed only when the source supports the repair; otherwise return to the audio or exclude it.

Use synthetic examples for identified coverage gaps and keep their origins visible. Start with a minority and compare against real-only supervision. Hamel Husain and Shreya Shankar recommend observed failure analysis and structured synthetic variation, while warning that synthetic cases do not establish real-world prevalence. [Evaluation guidance](https://hamel.dev/blog/posts/evals-faq/)

## Responsive inputs and creative freedom

Use the application's actual system instruction and response fields. Derive state, summaries, character knowledge and retrieved rules only from material preceding the response. A polished target must not leak its own answer into the input.

Include short sequences of two to six related responses, conditioned on the actual prior conversation, with loss only on the selected facilitator completion. Keep each story family in one split. Also evaluate interactive rollouts: recorded-prefix replay does not establish behavior after the model changes the conversation. Prompt/completion supervision is supported directly by the standard SFT approach. [TRL documentation](https://huggingface.co/docs/trl/main/en/sft_trainer)

Teach permitted invention explicitly: speakable NPC dialogue, new descriptive clues, and prospective developments. Preserve the boundary around established facts, private knowledge, unresolved checks and player choices. Score dull but correct answers separately from false ones. Otherwise an apparent fidelity improvement may simply mean that the model has stopped contributing.

## Controlled training sequence

Keep the model and serving topology fixed initially. The existing 27B recipe has run on two 24 GB cards; larger context and preference objectives need new memory measurements. Training competes with serving for those same GPUs.

| Experiment | Hold fixed | Vary |
| --- | --- | --- |
| Clean supervised training | Existing rank-16 / 2e-5 recipe and application contract | Carefully reviewed real responses |
| Data mixture | Recipe and evaluation | Real-only versus real plus targeted authored examples |
| Learning-rate pilot | Selected dataset and adapter rank | 2e-5 versus 8e-5 |
| Capacity, if needed | Selected data and measured training budget | Rank 16 versus 32 with compatible scaling |
| Preference training, conditional | Selected SFT checkpoint and evaluation | SFT versus SFT plus reviewed preferences |

These are proposed comparisons, not measured winners. Change one factor at a time. Record training tokens, updates, effective batch size, epochs, time and peak memory. Report compute differences between differently sized datasets. Save intermediate checkpoints and select with validation responses as well as loss. Keep final test families out of selection.

Start with completion-only masks, no sequence packing and the validated 4,096-token window. Do not silently truncate the triggering exchange. Compare prefix-only summaries with larger context only after auditing what those summaries preserve; 8K training is a memory experiment. LoRA research supports testing learning rate and including MLP as well as attention layers, but does not establish optimal settings for this particular task. The existing recipe already covers both. [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)

## Preferences after good supervised responses

Generate competing responses on training-only prompts from the selected local checkpoint. Compare plausible alternatives: vivid versus bland while both remain grounded; responsive versus stale; an open choice versus scripting the player. Record fidelity failures separately from prose preference, allow ties, vary presentation order and inspect length bias. Artificially terrible rejected responses are a weak test of useful distinctions.

DPO learns from chosen/rejected pairs without a separate online reward model. It is a reasonable next experiment, provided the labels are worth learning. Strong post-training recipes also emphasize deliberate data mixtures and evaluation rather than an objective alone. [DPO paper](https://arxiv.org/abs/2305.18290), [Tülu 3](https://arxiv.org/abs/2411.15124)

For the two-card setup, test quantized adapters, checkpointing and precomputed reference log probabilities before a long preference run. Freeze the **selected SFT reference**: disabling its adapter and accidentally using the original base changes the objective. Bind caches to checkpoint, tokenizer, template, prompt and completions. TRL documents PEFT and reference options, but the installed version and distributed configuration still need verification. [TRL DPO documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer)

Keep the verifiable evidence-policy RL experiment separate. A reward such as “make the story exciting” is not yet a reliable GRPO objective. Before story-focused RL, establish that the reward cannot be inflated through empty caution, length, repetition or invented outcomes.

## Evaluation and promotion

Freeze independent story families before editing targets. Check alternate transcripts, recaps and adaptations across splits. Repeated seeds do not create new independent scenarios. Use known failures as regressions and a separate final test for generalization.

Compare raw writers and the final application separately under identical prompts, sampling and output caps. A runtime editor can hide a weak writer; record its repair rate and latency. Exact-reference similarity is not the creative objective.

Review source fidelity, player agency, current-task response, protected knowledge and usable completion separately. Then compare voice, specificity, pacing and what players can engage with. Task-dependent writing criteria have precedent in [WritingBench](https://arxiv.org/abs/2503.05244); its benchmark is not itself evidence of quality in interactive games.

Include changing tasks, noisy and incomplete input, corrections, long-range callbacks and different facilitator directions. Report source-group results, severe failures and uncertainty alongside preference. Promote only after useful gains survive fidelity/agency checks and the complete workflow. Lower loss or a prettier isolated paragraph is insufficient.
