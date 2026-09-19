# Qwen TTRPG

Created by [David Russell](https://github.com/russedavid).

Fine-tune and evaluate a local Qwen model for tabletop roleplaying assistance: recognizing game actions, responding to players, and consulting supplied rules. Train separate task adapters, verify that they actually reload, then serve them from **one shared quantized base model**.

This is training and inference tooling, independent of a campaign application. The game system, setting, source material, and mechanics are supplied externally. No recordings, source conversations, rulebooks, datasets, trained weights, or private experiment reports are included.

## The workflow

```text
Reviewed conversations / grounded task examples
                     |
     Conversational Dataset Formatter
       split checks + completion masks
                     |
          verified dataset snapshot
                     |
        QLoRA + FSDP2 on two GPUs
                     |
      portable adapter + strict reload
                     |
  paired base/adapter evaluation and review
                     |
       GGUF conversion -> shared base
                         |-- classifier adapter
                         |-- storyteller adapter
                         `-- rules adapter
```

| Task | Learn / evaluate | Keep outside the model's authority |
| --- | --- | --- |
| Classifier | Distinguish proposed actions from established outcomes; quote supporting dialogue | Unsupported facts and premature state changes |
| Storyteller | Respond to participant choices with continuity, useful detail, and restraint | Participant decisions and the official conversation record |
| Rules | Use supplied excerpts, cite evidence, and ask for missing information | Invented or remembered mechanics from another system |

`qwen_ttrpg.tasks` provides system-neutral prompt contracts and original smoke cases. A storyteller response is private, point-in-time guidance. The actual conversation determines what happened; generating or refreshing a suggestion does not establish a game event.

## Install

Python 3.10+ runs the lightweight validation, serving launcher, and HTTP benchmark client:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

GPU training uses a separate **Python 3.12** environment. The recipes target **Qwen3.8-27B**, two 24 GB NVIDIA GPUs, and substantial host RAM for FSDP2 CPU offload. A 128 GB host is the reference configuration. These are bounded recipes, not a claim that arbitrary context sizes or concurrency fit that hardware.

```sh
uv venv .training --python 3.12
uv pip install --python .training/bin/python --torch-backend cu128 \
  -r qwen_ttrpg/recipes/training-requirements.txt -e .
```

Install the current **format_conversation_dataset** checkout in that environment for tokenizer/loss evaluation:

```sh
uv pip install --python .training/bin/python \
  -e '../format_conversation_dataset[tokenize]'
```

Use the named, locally downloaded model revision in the recipe. The Qwen3.8 configuration uses the `Qwen3_5DecoderLayer` implementation name; that is an architecture class name, not a different model selection. The recipes include linear-attention projections as LoRA targets. Do not assume another Qwen architecture uses the same targets or converter.

## 1. Prepare and verify inputs

The formatter produces `manifest.json` and `train`, `validation`, and optional `test` tokenized JSONL files. Every row contains `input_ids`, `attention_mask`, and `labels`; only the assistant completion has non-`-100` labels. Keep these files outside this checkout.

Use genuinely separate conversation groups for held-out evaluation. For classifiers and rule assistants, prepare reviewed structured responses grounded in the supplied input. Narration alone is not training data for all three tasks. The input contract is described in [the dataset interface](docs/dataset-interface.md).

The launcher verifies snapshot hashes, row counts, label masks, and sequence limits before producing a run configuration. A configuration-only smoke check does not reserve GPUs or start training:

```sh
.training/bin/python -m qwen_ttrpg.train /path/to/private/snapshot \
  --model /path/to/local/qwen \
  --output /path/to/private/config-check --prepare-only
```

## 2. Train an adapter

Stop inference and speech jobs on the training GPUs first. The launcher refuses to start when either GPU is busy; it does not stop other processes itself.

```sh
.training/bin/python -m qwen_ttrpg.train /path/to/private/snapshot \
  --model /path/to/local/qwen \
  --output /path/to/private/training-run
```

The default response recipe uses 4-bit QLoRA, rank 16, BF16 computation, FSDP2 CPU offload, and activation checkpointing. It initializes CPU/GPU collectives before the trainer to support offloaded gradient norms. `--max-steps 1` is a launch smoke test; `--epochs` changes the training duration. Neither is a quality criterion.

Each run records the materialized recipe, package versions, dataset manifest, elapsed time, training log, and adapter checks. Training exports a portable adapter by removing activation-checkpoint wrapper components from tensor names **without changing tensor values**. It refuses nonfinite tensors, name collisions, or an adapter with no demonstrated LoRA update.

## 3. Verify reload and compare against the base

```sh
.training/bin/python -m qwen_ttrpg.evaluate_adapter \
  /path/to/private/snapshot/validation.sources.jsonl \
  --model /path/to/local/qwen \
  --adapter /path/to/private/training-run/portable-adapter \
  --output /path/to/private/paired-generation.json
```

This loads the same base for both variants and checks every adapter key, shape, and actual loaded value. `--loss-only --sequence-len 4096` computes token-weighted completion loss instead. Direct Transformers generation here is greedy and diagnostic; production-style sampling and streaming timings are evaluated through the serving benchmark below.

A lower reference loss does not establish a better game response. Evaluate responsiveness to changed participant intent, continuity, agency, unsupported facts, missing-information behavior, and rules citations. The recorded continuation is one reference, not the only correct answer. See [evaluation guidance](docs/evaluation.md).

## 4. Convert and serve one base with task adapters

Build a CUDA-enabled **llama.cpp b11043** checkout and obtain a compatible quantized base separately. The conversion wrapper is tied to that runtime's Qwen converter internals; inspect and retest when changing runtime revisions.

```sh
.training/bin/python -m qwen_ttrpg.convert_adapter \
  --runtime /path/to/llama.cpp --base /path/to/local/qwen \
  --adapter /path/to/private/training-run/portable-adapter \
  --output /path/to/private/storyteller.gguf
```

The wrapper preserves low-rank factors through Qwen head permutations and runs a numerical row/column identity check before conversion. Conversion is a file-format check, not a substitute for comparing generated responses after conversion.

```sh
ttrpg-serve --runtime /path/to/llama.cpp \
  --base /path/to/local/base.gguf \
  --adapter classifier=/path/to/private/classifier.gguf \
  --adapter storyteller=/path/to/private/storyteller.gguf \
  --adapter rules=/path/to/private/rules.gguf \
  --layout split --tensor-split 1,1 --context 16384 --slots 1 \
  --output /path/to/private/serving-run
```

The default CLI layout splits the base across two GPUs. `--layout single` is available when measured headroom permits it. Context is **per slot**; more slots increase KV-cache demand. An asymmetric split can reserve VRAM for another service, but should be capacity-tested on the actual workload.

The server binds to loopback. Each request explicitly selects one adapter and sets every other scale to zero; the base comparison disables all adapters. Prompt cache reuse across adapters is disabled. Different adapter configurations may queue separately: one shared base saves model memory, but does not promise simultaneous execution of all tasks.

Set `TTRPG_MODEL_ROUTING` to the generated `routing.json` when using `routing.selection()` from another application. `--candidate NAME=/path/to/candidate.gguf` loads an unselected comparison adapter. Auditor requests must use the untuned base.

## 5. Run paired serving benchmarks

Generate original smoke cases into a private output directory, or supply your own held-out evaluation cases:

```sh
python -m qwen_ttrpg.tasks > /path/to/private/smoke-cases.json

ttrpg-benchmark /path/to/private/smoke-cases.json \
  --routing /path/to/private/serving-run/routing.json \
  --output /path/to/private/benchmark-run
```

Open `review.html` for a navigable A/B comparison. Candidate labels are shuffled independently of generation order; the answer key and timing/adapter metadata are in separate files. `results.json` contains completion status, time to first visible text, total latency, token usage, explicit control checks, and provenance. Run identical cases/settings for both variants and account for warm-up.

Benchmarks accept loopback endpoints only. Use an SSH tunnel for a separate GPU machine. Public smoke cases check the harness and a few response contracts; they are not representative benchmarks or evidence of model quality.

## Development and public releases

```sh
python -m pytest -q
python scripts/check_public_tree.py
```

CI runs CPU tests on Python 3.10 and 3.12. GPU training, strict reloads, and throughput need the actual hardware and local models; CI does not claim to perform those jobs. Generated outputs, adapters, local configuration, and source materials are excluded from the repository. The optional release scanner can inspect reachable history with an external private-identifier list.
