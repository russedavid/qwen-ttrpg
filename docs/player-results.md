# Player-adapter development results

Two separate LoRAs were trained for invented player personalities: curious (observe, connect clues, ask useful questions) and decisive (propose a bounded next action, invite cooperation). Personality is separate from the character being played. The adapters share the same Qwen3.8-27B base and can be selected per request.

## Data and training

Reviewed responses retained their original wording and preceding conversational stimulus. Whole story groups separated train, validation, and test, with one transcription variant per source recording and duplicate-response checks. Screening and subsequent text review were performed by models and an assistant, not a calibrated human panel. Audio was not reverified. Uncertain speaker boundaries and incomplete responses were excluded; remaining transcription and attribution errors are possible.

| Measure | Curious | Decisive |
| --- | ---: | ---: |
| Training responses | 48 | 75 |
| Validation responses | 19 | 13 |
| Held-out responses | 14 | 15 |
| Training launch to completion | 13.68 min | 19.49 min |
| Held-out target tokens | 567 | 756 |
| Base completion cross-entropy | 2.362 | 2.573 |
| Adapter completion cross-entropy | 2.098 | 2.197 |

Both used the existing rank-16 QLoRA recipe, alpha 32, learning rate 0.00002, a 4,096-token window, microbatch 1, accumulation 4, and two configured epochs on two RTX 3090 24 GB cards with FSDP2, CPU offload, and activation checkpointing. Loss applies only to the response and its end-of-turn tokens. Three otherwise retained examples were omitted because their protected stimulus and response could not fit. The decisive trainer reported a final epoch of 1.947 under its step schedule.

Strict NF4 reload verified all 672 tensors for each adapter, with no missing or unexpected keys and equality to the exported weights. GGUF conversion passed the low-rank permutation checks. These checks establish artifact validity, not response quality. Completion loss is token-weighted fit to the particular source responses; the small token totals limit what can be inferred.

## Generated responses

Six held-out cases per identity were generated with the base and the relevant adapter. Case, context, schema, seed (42), output limit (384 tokens), server, and quantization were held constant. Execution order and displayed labels were shuffled. An assistant reviewed the pairs before opening their identity keys; this was not a human-calibrated assessment or a multi-seed study.

| Initial comparison | Curious | Decisive |
| --- | ---: | ---: |
| Prefer base | 3 | 3 |
| Prefer adapter | 1 | 2 |
| Neither sufficiently supported | 2 | 1 |
| Complete base responses | 6/6 | 6/6 |
| Complete adapter responses | 6/6 | 5/6 |

Both candidates sometimes assigned an observation to the wrong speaker or introduced an unsupported detail. The source-based prompts lacked a verified mapping between speaker IDs and character identities, which complicates interpretation. The application supplies an explicit character and scoped sheet; its separate integration scenarios do not resolve the weakness of these training comparisons. One decisive adapter response repeated until the token limit.

The initial run used temperature 0.7, top-p 0.8, top-k 20, and presence penalty 0. The app and benchmark now use the vendor's non-thinking narrative preset, including presence penalty 1.5. This follows the [official model guidance](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices), which discusses presence penalty as a repetition control with possible quality tradeoffs. Repeating the six decisive cases completed 6/6 responses for both candidates, with median generation-request times of 3.10 seconds for base and 3.13 seconds for adapter. This fixes the observed completion regression in that rerun; it does not establish a general quality gain. Those cases are now development evidence.

The evidence supports working, independently selectable adapters and improved reference fit. It does **not** support claiming that either personality adapter consistently outwrites the base. Better speaker attribution, more complete responses, independent review, and untouched multi-seed comparisons are needed before making that claim. All source material, generated responses, review keys, and weights remain private; the repository provides the reproducible workflow.

## Integrated serving

The quantized base with the three existing task adapters and both player adapters runs on one 24 GB card with a configured 65,536-token server window. An idle speech worker occupies the other card. A snapshot showed about 20.0 GiB and 4.3 GiB of total device memory in use respectively; these are observations, not maximum load guarantees. The application serializes shared-model work rather than allocating five model copies.

Six original application scenarios passed 34 specified checks for perspective isolation, adapter routing, labelled contributions, stale-output rejection, duplicate suppression, and unchanged world state. Warm end-to-end turns took 6.62–8.37 seconds. These software checks do not measure narrative quality. See [Story Copilot's findings](https://github.com/russedavid/story-copilot/blob/main/docs/validation.md#player-agents-20-september-2026).
