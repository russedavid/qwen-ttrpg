# Storyteller retraining pilot

This is the earlier 47-example pilot. The [subsequent data, learning-rate and DPO experiment](storyteller-preference-results.md) reports a separate held-out comparison; its figures should not be pooled with this one.

A response-focused QLoRA pilot improved editorial preference and validation loss, but did not beat the base model on strict source-and-task checks. The new adapter is retained as an experimental candidate; Story Copilot keeps the base writer as its default. Factual and numerical answers use the base writer even when a workspace selects a creative adapter.

## Training and verification

The frozen dataset contained 47 training examples: 23 assistant-edited responses from the existing training partition and 24 original authored examples. Each response included the preceding exchange and the application's structured output contract. These are assistant-reviewed targets, not independently verified human gold. No recorded validation or test material was moved into training.

Four separate authored scenarios supplied eight validation turns. Eight other authored scenarios supplied 16 test turns, fixed before training. Turns from a scenario remain in one partition. Completion-only masks exclude source dialogue from the loss; all examples fit the 4,096-token training budget without omission.

The run used the 27B base, 4-bit QLoRA, rank 16 / alpha 32, learning rate 2e-5, two epochs and 12 optimizer updates, with FSDP2 and CPU offload on two 24 GB GPUs. Training took about 11.6 minutes including stage startup. Sampled device memory peaked at 21,543 MiB and 21,419 MiB; polling can miss brief higher peaks. Portable reload matched all 672 adapter tensors, with no missing or unexpected keys. Conversion produced a 171,488,384-byte GGUF adapter. A missing formatter dependency interrupted post-training validation; validation and conversion were rerun successfully without repeating training.

Token-weighted completion cross entropy on eight validation responses (478 target tokens) fell from **1.4348 to 1.1588**. This measures fit to these references, not story quality or dialogue accuracy.

## Blinded comparison

Base, previous adapter and new adapter received identical application prompts, schema, production sampling and a 700-token output cap. Each of the 16 test turns ran at two fixed seeds: 32 attempts per candidate, 96 generations total. Those are repeated observations from **eight independent authored scenario groups**, not 32 independent scenarios. Generation order and blinded review labels were randomized separately.

An assistant saved ratings before opening the candidate key. Reviews were not calibrated against human judgments. A strict pass required respecting source constraints, answering the current task, leaving player decisions open, and providing at least usable prose. Preference was a separate overall editorial judgment; a preferred response could still contain an error. Ties received half credit.

| Measure | Base | Previous adapter | New adapter |
| --- | ---: | ---: | ---: |
| Strict passes / 32 attempts | 22 | 16 | 21 |
| Clear contradiction or role-confusion failures | 2 | 3 | 1 |
| Failures to answer the current task | 0 | 3 | 2 |
| Voice score / 64 possible points | 40 | 37 | 44 |
| Preference credit / 32 comparisons | 8 | 7.5 | 16.5 |

The new candidate more often supplied concise, speakable NPC replies. It still sometimes repeated the request instead of answering, gave a goal instead of a concrete first step, or inferred an ability beyond the supplied evidence. A subsequent application replay also exposed an invented answer to an explicitly unknown event. These failures prevent an unconditional promotion despite better preference scores.

Failed test responses became development regressions for application editing **after** unblinding. Improvements to those repairs are not fresh held-out model results. No further training on these test responses is included in the figures above. Source material, response text, private comparison records and weights are not distributed.

## Reproduction

Use [your own reviewed data](bring-your-own-data.md), the existing response QLoRA recipe, strict adapter reload, GGUF conversion, and the [multi-candidate comparison command](evaluation.md#http-benchmark-case-format). Freeze scenario groups and evaluation targets before training. Record both raw model quality and the final application's usefulness: suppressing an unsupported sentence is not sufficient if nothing useful remains.
