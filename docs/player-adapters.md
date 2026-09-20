# Individual player adapters

A player adapter learns a response style for a player identity. The character sheet and permitted knowledge belong to the application context, so the identity can be assigned different characters. The `player-turn-v1` response is JSON with `utterance` and `recipient` (`table`, `facilitator`, or a supplied character ID).

This workflow preserves source responses. It does not assume that `SPEAKER_01` identifies the same person in different recordings. For invented personalities, review responses for their fit to the declared style. Modelling a particular real participant requires reliable source-specific identity mappings.

Create a private JSON configuration containing `profiles` (an ID-to-instructions object), `conversations`, and optionally `context_turns`, `seed`, and `max_candidates_by_split`. Each conversation needs `path`, `format`, `group`, `source_identity`, `split`, and `speakers`. Formats are `turns-json` (an object with a `turns` list containing `speaker` and `text`) or supported speaker-text/word-aligned input. Speaker mappings use `facilitator`, `player`, or `observer`.

Use meaningful, independent groups for train/validation/test and one transcription variant per recording. The collector joins contiguous responses from one player and protects the preceding stimulus. It excludes unresolved context speakers, missing stimuli, and excessive response lengths. Optional `strict_text_mining: true` applies conservative flags for incomplete tails, possible mixed voices, reported dice results, and lengthy responses; inspect its omission counts before relying on it.

```sh
python -m qwen_ttrpg.player_data collect /path/to/private/config.json \
  --output /path/to/private/candidates.json

python -m qwen_ttrpg.player_data screen /path/to/private/candidates.json \
  --output /path/to/private/screened.json \
  --url http://127.0.0.1:8091/v1 --model local-model --adapter-count 0
```

The screening server must be local. Supply its exact loaded adapter count, even when those adapters are not selected; the screen explicitly disables them all. Screening is attributed to the model and does not count as human approval. Review the source, complete response, personality fit, and speaker boundaries before preparing a dataset. A person or another reviewer can update only a case's `review` object, recording `verdict`, `profile`, `origin`, `reviewer`, `reason`, `response_complete`, and `target_sha256`. Source/context/target edits require a new collection and review. Keep the original review as audit metadata inside that object when superseding it.

To resume a partial screen, use the last saved JSON as input and a new output filename; already reviewed cases are preserved. The full protected stimulus remains available even when a screening request uses a shorter evidence view than the eventual training context.

Prepare one immutable, completion-masked snapshot per personality:

```sh
python -m qwen_ttrpg.player_data prepare /path/to/private/reviewed.json \
  --profile curious --model /path/to/local/base \
  --output /path/to/private/curious-data --sequence-len 4096 \
  --allow-model-reviews

ttrpg-train /path/to/private/curious-data --model /path/to/local/base \
  --output /path/to/private/curious-run
```

Omit `--allow-model-reviews` when every retained response has a human review. The formatter checks grouping, source identity, future-turn exclusion, and completion-only token masks. Exact repeated completions are retained in held-out data before training data. Loss measures fit to those particular reference responses, not whether a new player contribution is the best possible choice.

Use the existing strict reload, conversion, and evaluation tools for each resulting adapter. The serving command can load multiple player adapters with repeated `--candidate NAME=/path/to/adapter.gguf`; their inventory IDs are then selected per player in Story Copilot. Compare each against the base using identical contexts and sampling, with judgments for player agency, knowledge boundaries, responsiveness, and personality. Keep all transcripts, reviews, snapshots, weights, and generated comparisons outside Git.
