# Train a source-grounded decision policy

This optional experiment trains a Qwen3.5-4B LoRA to retrieve evidence, answer structured factual questions, and ask for missing information. It uses Story Copilot's actual evidence tools in an authored environment. Its score checks explicit conclusions and sources; free-form writing quality requires separate review.

The [recorded development results](agent-rl-results.md) compare base, short and extended supervised training, and RL, including the independent cases, failures, runtime, and memory measurements.

The 4B adapter belongs to the 4B base. It does not replace or attach to the existing 27B task/player adapters. Model files and every generated artifact stay outside the repository.

## Environment

Use Linux, a managed CPython 3.12 environment, and one idle NVIDIA GPU with at least 14 GiB free before launch. A 24 GiB card is the reference device. Set `CUDA_VISIBLE_DEVICES` to select exactly one card. The runner does not stop existing applications. The initial managed runtime also avoids mixing a Conda SQLite build with a different C++ runtime imported by Torch.

Install the pinned GPU dependencies, this toolkit, and a checkout of Story Copilot containing `rl_environment.py`:

```sh
uv venv .rl --python 3.12 --python-preference only-managed
uv pip install --python .rl/bin/python --torch-backend cu128 \
  -r qwen_ttrpg/recipes/agent-rl-requirements.txt -e '.[dev]' -e ../story-copilot
```

Download [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) separately into a local model directory. The runner loads local files only. Use the same model revision, frozen environment, and decoding settings for every comparison.

## Prepare and establish baselines

```sh
.rl/bin/python -m qwen_ttrpg.agent_rl prepare --output /path/to/private/decision-data

CUDA_VISIBLE_DEVICES=0 .rl/bin/python -m qwen_ttrpg.agent_rl evaluate \
  --data /path/to/private/decision-data --model /path/to/local/qwen-4b \
  --output /path/to/private/base-validation --split validation --label base

CUDA_VISIBLE_DEVICES=0 .rl/bin/python -m qwen_ttrpg.agent_rl train \
  --data /path/to/private/decision-data --model /path/to/local/qwen-4b \
  --output /path/to/private/sft --phase sft --steps 40 --learning-rate 0.0001
```

Preparation creates 144 training cases, 24 validation cases, 24 test cases, and 12 transfer cases with different task structures. The generated demonstrations use only the training partition. Input hashes and the environment's source hash are verified before each run. The fixture generator supplies known facts and alternative valid evidence sets; it uses no source recordings or real game-system rules.

Evaluate the supervised adapter with the same validation command, adding `--adapter /path/to/private/sft/adapter`, a new output directory, and `--label sft`. Keep the test and transfer partitions untouched while choosing a training configuration.

## GRPO pilot

```sh
CUDA_VISIBLE_DEVICES=0 .rl/bin/python -m qwen_ttrpg.agent_rl train \
  --data /path/to/private/decision-data --model /path/to/local/qwen-4b \
  --adapter /path/to/private/sft/adapter --output /path/to/private/grpo-pilot \
  --phase grpo --steps 20 --learning-rate 0.00001
```

The learner updates rank-8, alpha-16 adapters on language decoder projections. The base remains frozen. Four independent attempts share an initial task and one in-process model; their environments and evidence remain separate. A trajectory has at most three evidence calls and one terminal decision. Generation is batched; gradient microbatches contain one trajectory, accumulated over four.

TRL 1.9.0 performs the GRPO-family update with DAPO loss normalization, temperature 1, and beta 0. The run has no learned critic or separate KL reference model. The environment supplies the reward; model-authored reward text cannot change it. Failed attempts can receive small evidence-discovery credit, while exact task success is recorded separately.

The native default template removes empty reasoning prefixes from earlier turns. The runner uses TRL's prefix-preserving training template consistently for baselines, supervised examples, RL, and evaluation. It verifies that every continuation retains the exact preceding model tokens. Tool results and their delimiters have zero policy-loss mask. Checkpoint reload verifies adapter keys, shapes, and values. See [TRL's GRPO documentation](https://huggingface.co/docs/trl/v1.9.0/en/grpo_trainer).

Inspect `run.json`, `metrics.jsonl`, and `rollouts.jsonl`. Track success, individual reward components, reward variance, unnecessary calls, invalid actions, truncation, gradient norms, changed tensors, and memory. A rising shaped reward alone does not establish better decisions. A zero-variance group supplies no relative learning signal.

The recorded `unnecessary_calls` field counts calls above the authored reference path's length. That path knows the scenario's construction; an extra lookup can be reasonable before the agent knows which input is missing. Interpret this counter as distance from the reference, not a judgment that every extra call was wasteful. Task success depends on the conclusion and declared valid evidence sets, not following the demonstration's action order.

To continue a compatible checkpoint toward a larger total step target:

```sh
CUDA_VISIBLE_DEVICES=0 .rl/bin/python -m qwen_ttrpg.agent_rl train \
  --data /path/to/private/decision-data --model /path/to/local/qwen-4b \
  --adapter /path/to/private/grpo-pilot/checkpoints/checkpoint-20 \
  --resume-checkpoint /path/to/private/grpo-pilot/checkpoints/checkpoint-20 \
  --output /path/to/private/grpo-100 --phase grpo --steps 100 \
  --learning-rate 0.00001
```

The dataset, phase, model, seed, context budget, and learning rate must match. The new run preserves the original outputs and resumes the optimizer, sampler/RNG state, and step count through Trainer.

For an additional-training control, resume the supervised checkpoint at step 40 toward 140 total supervised updates, retaining `--phase sft` and its original learning rate. This matches the number of added optimizer updates, not the compute cost or number of generated tokens. Compare measured runtime as well as task outcomes; an RL gain over the earlier supervised checkpoint alone does not establish that RL is preferable to more supervised training.

## Independent assessment

For each candidate, run `evaluate` with `--split test` and `--split transfer`, new output directories, explicit labels, and the same `--seeds` (default 42 and 314). Multiple seeds are repeated attempts on each scenario, not additional independent scenarios. Preserve the raw attempts and compare successful conclusions, supported citations, unnecessary calls, and latency. Inspect wording separately, especially clarification questions and rationales.

Create an aligned review for one partition:

```sh
.rl/bin/python -m qwen_ttrpg.rl_review \
  /path/to/private/base-test/report.json \
  /path/to/private/sft-test/report.json \
  /path/to/private/grpo-test/report.json \
  --output /path/to/private/test-review
```

Open `review.html`. It shows the task, each candidate's decisions and tool results, the source world, the checker target, and acceptable citations. Dataset/settings mismatches are rejected. Set Story Copilot's `STORY_EXPERIMENTS` to the parent of private review directories to browse them from the local app.

Optional `--notes /path/to/private/review.json` adds source-review annotations. Its `findings` list binds each note to `candidate`, `id`, and `seed`, with `reviewer` and `finding` text. Keep reviewer identity and calibration status explicit; automatic checks and assistant review are different evidence.

CPU tests exercise reward shortcuts and report alignment. With the training environment, also run `python -m pytest tests/test_rl_tensor_contract.py -q` to check batched episode identity, padding removal, and external-token masks. Actual GPU training and held-out model behavior remain separate checks.


## Serve the learned decision policy

Export the adapter using the same HF base revision, then serve the matching base and adapter together. The saved training template keeps multi-turn prefixes consistent with training:

```sh
python -m qwen_ttrpg.serve_models \
  --runtime /path/to/llama.cpp --base /path/to/private/4b-base.gguf \
  --adapter planner=/path/to/private/rl-adapter.gguf \
  --chat-template /path/to/private/rl/adapter/chat_template.jinja \
  --model-alias evidence-policy --layout single --gpu 1 \
  --context 8192 --port 8093 --output /path/to/private/planner-serving
```

The server writes its ordered adapter inventory and routing configuration. Point Story Copilot's optional **Dedicated decision planner** at this endpoint and use that inventory. The larger writing model can remain on the other GPU. Measure the actual loaded models, context caches, and simultaneous speech workload on your machine before relying on this arrangement; idle memory alone does not establish concurrent capacity. The live bridge falls back to the main model if the policy endpoint fails or its complete context does not fit. Its numerical proposals are not accepted as rulings without separate source validation and calculation.
