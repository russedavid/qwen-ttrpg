# Dataset interface

A dataset snapshot is a new directory containing:

- `manifest.json`: sequence length, per-split example counts, and `tokenized_sha256` hashes; formatter exports also include `sources_sha256`.
- `train.tokens.jsonl`, `validation.tokens.jsonl`, optionally `test.tokens.jsonl`: unpadded rows with equally sized `input_ids`, `attention_mask`, and `labels` arrays. The prompt is masked with `-100`, followed by one contiguous assistant completion including its end token.
- `{split}.sources.jsonl`: reviewed `prompt` and `completion` chat messages and provenance, needed for generation/loss evaluation.
- `{split}.mask-audit.json`: completion text, token boundaries, removed context, and review provenance. These are private audit outputs.

The Conversational Dataset Formatter creates this contract from reviewed responses. For other supervised tasks, use its `pipeline.prepare(datasets, output, tokenizer, sequence_len)` API. Each row needs:

```python
row = {
    "id": "unique-content-identifier",
    "prompt": [{"role": "system", "content": "Task instructions"},
               {"role": "user", "content": "Source-backed input"}],
    "completion": [{"role": "assistant", "content": "Reviewed target response"}],
    "provenance": {
        "group": "independent-source-group",
        "source_sha256": "sha256-of-original-source",
        "source_identity": "shared-identity-for-transcription-variants",
        "split": "train",
        "target_turn": 2,
        "target_turns": [2],
        "context_turns": [1],
        "protected_context_turns": [1],
        "review_level": "reviewed",
        "reviewer": "operator"
    }
}
```

Use real source hashes and meaningful provenance in local data. For generated labels, mark the review origin honestly and validate labels independently; a model's agreement with its own labels is not an independent evaluation. Keep the same source group out of validation/test when it contributed training examples. Rules questions must include the excerpts used to support their answers.

The training runner accepts the original snapshot manifest shape as well as formatter v0.2 snapshots. It validates the tokenized files before training; it cannot reconstruct a missing upstream review or infer that two nominally different source groups actually overlap.
