# Qwen TTRPG

Created by [David Russell](https://github.com/russedavid).

Bring your own conversations and rule references. Prepare and review the examples, fine-tune three Qwen adapters, verify the saved weights, compare them with the base model, and serve them from **one shared quantized base**.

| Adapter | Training example |
| --- | --- |
| Game-state classifier | A dialogue window → source-supported actions, claims, outcomes, and state changes |
| Rules assistant | A question and supplied excerpts → an answer with exact citations and explicit missing information |
| Storyteller | The preceding exchange → the facilitator's complete response to the players |

The game system, speaker mappings, terminology, rule excerpts, and labels are supplied externally. No recordings, source conversations, rulebooks, private datasets, or trained weights are distributed. The included example generator creates short, original synthetic fixtures in a directory you choose outside Git.

## Install

Python 3.10+ runs the CPU data/review tooling on Linux or macOS. Git is required during installation: the companion [Conversational Dataset Formatter](https://github.com/russedavid/format_conversation_dataset) is pinned to a specific source revision and installed automatically.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,dev]'
ttrpg-pipeline --help
```

GPU training uses **Linux, Python 3.12, two 24 GB NVIDIA GPUs, and a 128 GB host** as the reference configuration. Install its dependencies separately:

```sh
uv venv .training --python 3.12
uv pip install --python .training/bin/python --torch-backend cu128 \
  -r qwen_ttrpg/recipes/training-requirements.txt -e '.[data]'
```

Download the recipe's **Qwen3.8-27B** model locally, obtain a compatible quantized base, and build a CUDA-enabled **llama.cpp b11043** checkout. No model download is performed by the pipeline. The architecture implementation is named `Qwen3_5DecoderLayer`; changing Qwen architectures requires reviewing the LoRA targets and converter.

## Exercise the complete workflow

Create a small original example outside the repository. These commands use the training environment throughout:

```sh
.training/bin/python -m qwen_ttrpg.pipeline init-demo /path/to/private/example \
  --model /path/to/local/qwen \
  --runtime /path/to/llama.cpp \
  --base-gguf /path/to/local/base.gguf \
  --python "$PWD/.training/bin/python"

.training/bin/python -m qwen_ttrpg.pipeline prepare /path/to/private/example/experiment.yaml
.training/bin/python -m qwen_ttrpg.pipeline plan /path/to/private/example/experiment.yaml
.training/bin/python -m qwen_ttrpg.pipeline run /path/to/private/example/experiment.yaml
```

`init-demo` records synthetic label provenance and limits each task to one optimizer step. It exercises installation, data preparation, training, reload, conversion, and inference. It is **not meaningful model training or a quality benchmark**. Stop other jobs using the GPUs before `run`; the public launcher refuses busy GPUs and does not stop unrelated services.

The runner performs each adapter's training, strict reload, and conversion in sequence. It then starts a temporary loopback server, runs paired base/adapter comparisons on validation and test examples, saves A/B review pages, and stops that server. Semantic review remains pending. To keep the verified adapters available for an application:

```sh
.training/bin/python -m qwen_ttrpg.pipeline serve /path/to/private/example/experiment.yaml
```

This runs a foreground server, with the configured loopback port and context limit. Its output directory contains the routing file for clients. A storyteller response is private guidance; this toolkit does not insert generated suggestions into an official conversation record or publish them to players.

## Bring your own data

Start with the [complete configuration and review guide](docs/bring-your-own-data.md). The working sequence is:

```text
external sources + speaker mappings + question catalog
                         |
               candidate review queue
             /           |           \
       classifier       rules      storyteller
       annotations    citations    source responses
             \           |           /
         reviewed, source-bound task examples
                         |
         three separate dataset snapshots
                         |
      train -> strict reload -> convert, per task
                         |
      shared base -> paired evaluation -> serve
```

```sh
ttrpg-pipeline candidates /path/to/private/experiment.yaml
# Review the generated HTML and edit proposed targets in the local review JSON.
ttrpg-pipeline review /path/to/private/experiment.yaml \
  --id CASE_ID --decision keep --origin human --reviewer operator \
  --reason "Checked the source and label meaning." --response-complete
ttrpg-pipeline prepare /path/to/private/experiment.yaml
ttrpg-pipeline run /path/to/private/experiment.yaml
```

Optional `annotate` uses a local untuned base to propose classifier/rules labels. Those proposals stay pending. Model-reviewed labels require explicit opt-in and retain their origin. A raw conversation supplies narrator responses, but does not automatically provide reliable event labels or rule answers.

## What is verified

- Reviews bind to source content, response boundaries, task instructions, and target hashes. Source edits invalidate dependent reviews.
- Conversation groups and source identities stay within one split across all adapters. Rules question families are split separately; shared supplied references are reported explicitly.
- The full triggering exchange and narrator response are preserved. Native chat-template checks assign loss only to completion tokens and the end token.
- Training uses rank-16 QLoRA, FSDP2, CPU offload, and activation checkpointing. Saved adapters must contain finite, updated tensors.
- Portable export preserves tensor values; strict reload checks keys, shapes, and actual values. GGUF conversion verifies the low-rank head-permutation identity.
- The experiment records input/runtime identities, separate stage attempts, logs, and artifact hashes. Repeating `run` skips verified completed stages and resumes interrupted training from a saved checkpoint when available.
- Benchmarks verify the server's adapter inventory, apply the training contracts to outputs, and report latency and control results separately from subjective usefulness. A/B display order is independent of execution order.

Source and target checks cannot prove semantic truth. Lower loss does not establish better storytelling. CI exercises code behavior; private data and weights are needed to reproduce a particular model result. This repository reproduces the **method with your data**, not a previously trained model's quality claims.

[Data contracts](docs/dataset-interface.md) · [Evaluation protocol](docs/evaluation.md) · [Workflow verification](docs/verification.md) · [Individual stage commands](docs/individual-stages.md)

## Development

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_public_tree.py
```

CI runs CPU tests on Python 3.10 and 3.12. GPU checks use the separate local environment and actual models. Generated reviews, datasets, reports, weights, and configuration belong outside Git; contributor attribution remains public.
