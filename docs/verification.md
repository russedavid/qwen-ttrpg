# Workflow verification

The complete three-adapter workflow was exercised locally on two 24 GB GPUs and a 128 GB host, using the pinned Qwen model/runtime configuration and the repository's original synthetic example generator.

| Check | Observed result | What it establishes |
| --- | --- | --- |
| Separate task preparation | Classifier, rules, and storyteller snapshots built with the real model tokenizer | The data contracts and completion masks connect to training |
| GPU training | All three one-step runs completed with finite, updated LoRA weights | The training launch, optimizer, saving, and export path work on the reference hardware |
| Strict reload | All 672 adapter tensors compared by name, shape, and loaded value for each task | The saved adapter actually reaches inference |
| Conversion | All three adapters converted with verified low-rank permutation identities | The exported weights pass the supported conversion checks |
| Paired generation | 16 responses recorded across validation and test comparisons | The shared-base server, task routing, response validation, and review exports connect end to end |
| Output limits | Four responses were truncated at the configured 512-token budget and counted as failures | Failed model responses remain visible without aborting the rest of the study |
| Resume | Repeating the completed run created no new stage attempts | Completed artifacts are verified and reused without another training run |
| Final data preparation | All nine task/split token streams matched the GPU-tested inputs | Preparation remained consistent through the integration checks |
| Live label proposals | Classifier and rules proposals generated through the local untuned base; both stayed pending review | Shape/citation checks do not silently approve generated supervision |

These are integration results, **not a model-quality benchmark**. The example contains only one classifier training row, one rules row, and two narrator rows. Its simple authored patterns appear across splits. One optimizer step cannot establish useful adaptation, generalization, or scientific superiority. Semantic response review remains pending.

CPU checks cover source and target changes, grouping/leakage, explicit label origins, protected context, exact quotations, declared calculation inputs, tokenizer changes, interrupted stage recovery, artifact tampering, virtual-environment execution, and public-file boundaries. The Qwen package's 33 tests passed on Python 3.10 and 3.12, including an installed-wheel run; the companion formatter has its own 18 tests. CI runs the CPU tests, not the GPU trial.

Separate browser checks exercised label-review and A/B pages, case navigation, and narrow-screen layout. Raw sources, generated outputs, checkpoint weights, screenshots, and private run directories are not published. Use `init-demo`, `prepare`, and `run` from the README to exercise the method on your machine; runtime, sampling, and hardware can affect outputs and timings.
