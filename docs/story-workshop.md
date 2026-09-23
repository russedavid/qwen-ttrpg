# Storyteller response workshop

Use this offline tool when source conversations need boundary repair or editorial improvement before training. Original transcripts, corrected speaker boundaries, and edited training targets remain separate. The existing three-task pipeline retains its stricter verbatim-source path.

## Collect complete exchanges

Supply a private JSON file with a `sources` list. Each source has `group` (story/campaign), `split` (`train`, `validation`, `test`), a stable `source_identity`, and `turns`. An optional `family` links related scenarios or adaptations. Each turn has an increasing integer `ordinal`, `speaker`, `role`, and `text`. Roles are `facilitator`, `player`, `observer`, `unknown`, or `mixed`. Optional fields include `character`, `visibility`, `category`, `status`, revision IDs and timestamps.

Supply a second JSON file with the application's actual writer `system` instruction, response `schema`, and `empty_response` object containing a `narration` field. Mining fills narration with the original response. The prompt contains `dialogue`, `new_player_input`, and empty reference-material lists. No state summary or future revelation is inferred from the answer.

```sh
python -m qwen_ttrpg.story_workshop collect \
  /path/to/private/sources.json /path/to/private/writer-contract.json \
  /path/to/private/workshop

python -m qwen_ttrpg.story_workshop review \
  /path/to/private/workshop/workshop.json \
  /path/to/private/review-batch --split train --limit 100
```

Open `review-batch/review.html`. Sampling rotates between story groups and recordings; it is a diagnostic queue, not an estimate of corpus prevalence. Text flags are reminders, not training labels. The default context is 24 turns, extended to preserve the previous facilitator response and entire player stimulus. Shorter windows need renewed context-sufficiency review.

The collector excludes unresolved context roles and explicitly private exchanges. Review must also catch unmarked whispers, mislabeled speech and missing scenario knowledge. If a target needs earlier context or a rule, recover material available **before** that response. Never backfill facts from the reference answer into its prompt.

## Repair boundaries without losing source text

The corrections file contains `source_snapshot_sha256` and a `turns` list. Each correction records `source_identity`, original `ordinal`, `turn_sha256`, review `origin` (`human` or `model`), `reviewer`, `reason`, and `spans`. Each span has zero-based character offsets `start` and exclusive `end`, plus its `role` and `speaker`. Use `qwen_ttrpg.story_workshop.fingerprint` for snapshot and turn hashes.

Spans must cover every source character in order, without gaps or overlaps. Uncertain spans remain `unknown` or `mixed`. A local correction never labels all appearances of a diarization speaker. Text inference does not establish acoustic identity or transcription accuracy.

```sh
python -m qwen_ttrpg.story_workshop repair \
  /path/to/private/sources.json /path/to/private/corrections.json \
  /path/to/private/repaired
```

Recollect from `repaired/sources.json` into a new workshop. Original turn positions, character spans and review provenance remain attached. Split turns retain their original time range as a reference; no word timestamps are invented. The original source is untouched. Source changes invalidate dependent reviews.

## Review and export

Edit the batch's `reviews.json`. Keep, reject, or leave each case pending. Kept targets need the actual reviewer and origin, an edited structured `target`, a reason, response `skills`, and a description of the permitted `creative_scope`. Explicit checks cover roles, visibility, context, response completeness, established facts, player agency and immediate usefulness.

An NPC reply or new sensory detail can be useful prospective invention. It cannot decide a player's response, manufacture a resolved roll, or convert a protected unknown into certainty. Do not approve an example just because its JSON parses. Retain model authorship and review provenance; do not call it independent human gold.

```sh
python -m qwen_ttrpg.story_workshop finalize \
  /path/to/private/workshop/workshop.json \
  /path/to/private/review-batch/reviews.json \
  /path/to/private/reviewed --allow-model-reviews

python -m qwen_ttrpg.story_workshop export \
  /path/to/private/workshop/workshop.json \
  /path/to/private/reviewed/reviews.json \
  /path/to/private/edited-sources --allow-model-reviews
```

Omit the model-review flag for human-only labels. Finalization binds targets to completed reviews; it never fills checks or approves pending cases. Later edits require another explicit finalization. Exports retain source/target hashes, source positions, split and review provenance. No training starts automatically.

The exported `*.sources.jsonl` files use the existing dataset interface. Combine reviewed batches while preserving source/group identities, then tokenize:

```python
import json
from pathlib import Path
from transformers import AutoTokenizer
from format_conversation_dataset.pipeline import prepare

root = Path('/path/to/private/edited-sources')
datasets = {
    split: [json.loads(line) for line in (root / f'{split}.sources.jsonl').read_text().splitlines()]
    for split in ('train', 'validation', 'test')
}
tokenizer = AutoTokenizer.from_pretrained('/path/to/local/model', local_files_only=True)
prepare(datasets, '/path/to/private/tokenized-snapshot', tokenizer, sequence_len=4096)
```

Both training and independent validation must be nonempty. Inspect mask audits and omissions. For the workshop's `dialogue` field, overlength examples are omitted rather than silently shortened. Recover a sufficient input or validate a larger window. See the [training strategy](storyteller-strategy.md) before selecting another checkpoint.
