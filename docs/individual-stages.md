# Individual training and serving commands

The [BYOD workflow](bring-your-own-data.md) orchestrates all three adapters. These commands remain available when inspecting or running an individual stage.

## 1. Prepare and verify inputs

The formatter produces `manifest.json` and `train`, `validation`, and optional `test` tokenized JSONL files. Every row contains `input_ids`, `attention_mask`, and `labels`; only the assistant completion has non-`-100` labels. Keep these files outside this checkout.

Use genuinely separate conversation groups for held-out evaluation. For classifiers and rule assistants, prepare reviewed structured responses grounded in the supplied input. Narration alone is not training data for all three tasks. The input contract is described in [the dataset interface](dataset-interface.md).

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

A lower reference loss does not establish a better game response. Evaluate responsiveness to changed participant intent, continuity, agency, unsupported facts, missing-information behavior, and rules citations. The recorded continuation is one reference, not the only correct answer. See [evaluation guidance](evaluation.md).

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

