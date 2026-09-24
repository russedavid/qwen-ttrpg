# Reviewed storyteller preferences

`qwen_ttrpg.story_preferences` continues a verified supervised adapter with direct preference optimization (DPO). It trains on two reviewed responses to the same input, using the selected supervised checkpoint as a frozen reference. It does not train a separate reward model or generate its own approval labels.

## Prepare pairs

Keep the dataset outside Git. The JSON document contains a `pairs` list. Each pair needs:

| Field | Contents |
| --- | --- |
| `id` | Unique, stable pair identifier |
| `prompt` | Chat messages, including the application's actual system instruction and the context available before the response |
| `chosen`, `rejected` | Different, nonempty strings containing complete responses in the application's format |
| `provenance` | `split` (`train` or `validation`), `group`, and `source_identity`; retain source and generation hashes and each response's origin here too |
| `review` | `verdict: prefer_chosen`, `chosen_valid: true`, `origin: human` or `model`, `reviewer`, `reason`, and `content_sha256` |

Calculate the review hash **after reviewing the final content**:

```python
from qwen_ttrpg.util import digest, packed

row['review']['content_sha256'] = digest(packed({
    key: row[key]
    for key in ('prompt', 'chosen', 'rejected', 'provenance')
}))
```

Hashing is not approval. Inspect the source, both responses, and the reason for the preference. A polished answer that decides a player's action or changes a protected fact is not a valid positive. Exclude ties and cases with two unacceptable responses unless an explicitly identified replacement is independently reviewed. Keep authored, source-edited, sampled and reviewer-edited responses distinguishable. Assistant review is not human calibration.

The trainer rejects duplicate IDs, cross-split source/story identities, stale hashes, missing reviews, empty splits and test pairs. It checks the full native chat template and completion end tokens before training. Overlength examples fail; they are not silently truncated. Semantic correctness and completeness of the supplied history still require review. Related story adaptations should be grouped before creating this file.

Off-policy pairs are supported: alternatives can come from an earlier checkpoint or the base. Record which model generated them. Reference probabilities are always recomputed from the checkpoint supplied to `--adapter`; cached probabilities from a different sampling policy are not substituted.

## Run on two GPUs

Install the optional pinned environment described in [individual stages](individual-stages.md), using `qwen_ttrpg/recipes/preference-requirements.txt` instead of the supervised requirements file. This adds the tested TRL version. The runner requires exactly two visible CUDA GPUs, each with at least 20 GiB free. Pause your serving processes first and arrange to restore them after the run, including on failure. This command does not manage other processes.

Use a small, separate smoke dataset with independent training and validation pairs, including the longest examples:

```sh
.training/bin/python -m qwen_ttrpg.story_preferences \
  --data /path/to/private/smoke-pairs.json \
  --model /path/to/local/qwen \
  --adapter /path/to/private/selected-sft/portable-adapter \
  --output /path/to/private/dpo-smoke \
  --steps 2
```

After memory, tokenization and tensor checks pass, start the substantive run from the **original selected supervised adapter**, not the smoke output:

```sh
.training/bin/python -m qwen_ttrpg.story_preferences \
  --data /path/to/private/pairs.json \
  --model /path/to/local/qwen \
  --adapter /path/to/private/selected-sft/portable-adapter \
  --output /path/to/private/dpo-run \
  --learning-rate 5e-6 --beta 0.1 --epochs 1 --max-length 4096
```

Every run needs a new output directory. This is one process with balanced model placement across both GPUs, NF4 quantization, BF16 computation, activation checkpointing, batch size one and four-step gradient accumulation. Do not launch it with the supervised FSDP command. Two 24 GB cards have completed the 27B pilot with inputs up to 2,150 tokens; that does not establish that every 4,096-token pair fits.

## Verify and compare

`run.json` records package versions, data/checkpoint/template hashes, exact prompt and completion token hashes, reference checks, elapsed time, tensor updates and peak allocated memory. `metrics.jsonl` records training and validation metrics. The reference adapter must equal the initial supervised weights and remain frozen and unchanged after training. The trainable adapter must contain finite, changed tensors.

The exported `portable-adapter` still needs the [independent reload and conversion checks](individual-stages.md#3-verify-reload-and-compare-against-the-base). Export alone does not establish a successful reload or useful behavior. Compare the base, selected SFT and DPO candidate on the same held-out inputs and production sampling, then replay the complete application. Save blinded semantic ratings before opening model labels. Report failed and empty outputs, editing frequency and response time alongside prose preference.

Preference loss, reference cross entropy and the fraction of validation pairs with a positive DPO reward margin measure different things. None is a percentage of good stories. Keep the existing serving default unless actual responses justify replacing it.
