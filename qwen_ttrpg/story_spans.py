"""Validate mined response exchanges against exact, chronological source spans."""

from __future__ import annotations

import copy
import re

from jsonschema import Draft202012Validator

from .util import packed, digest


def locate(turns, span):
    if span.get("role") not in {"player", "facilitator", "observer"}:
        raise ValueError("A source span needs an explicit role.")
    turn = next((t for t in turns if t["ordinal"] == span["turn"]), None)
    quote = span.get("quote")
    if turn is None or not isinstance(quote, str) or not quote.strip():
        raise ValueError("Missing source turn or quotation.")
    matches = list(re.finditer(r"\s+".join(re.escape(s) for s in quote.split()), turn["text"]))
    if len(matches) != 1:
        raise ValueError("A quote must identify exactly one source span.")
    match = matches[0]
    return {"turn": turn["ordinal"], "start": match.start(), "end": match.end(),
            "role": span["role"], "quote": match.group(), "revision": turn.get("revision")}


def candidate(source, proposal, contract):
    """Mechanical validation only; the returned record still needs semantic review."""
    if source["split"] not in {"train", "validation", "test"}:
        raise ValueError("Use a preassigned source split.")
    located = {k: [locate(source["turns"], s) for s in proposal[k]]
               for k in ("background", "participant", "facilitator")}
    if not located["participant"] or not located["facilitator"]:
        raise ValueError("Require both a participant stimulus and a source response.")
    if any(s["role"] != "player" for s in located["participant"]) or any(
        s["role"] != "facilitator" for s in located["facilitator"]
    ):
        raise ValueError("Stimulus and response roles are inconsistent.")
    key = lambda s: (s["turn"], s["start"])
    targets = sorted(located["facilitator"], key=key)
    # A selected earlier quote cannot include even the beginning of a target.
    context = sorted(located["background"] + located["participant"], key=key)
    unique = {}
    for s in context:
        identity = (s["turn"], s["start"], s["end"])
        if identity in unique and unique[identity]["role"] != s["role"]:
            raise ValueError("The same source speech has conflicting roles.")
        unique[identity] = s
    context = list(unique.values())
    if any((s["turn"], s["end"]) > key(targets[0]) for s in context):
        raise ValueError("Context contains response or future speech.")
    for spans in [context, targets]:
        for a, b in zip(spans, spans[1:]):
            if a["turn"] == b["turn"] and a["end"] > b["start"]:
                raise ValueError("Selected source spans overlap.")
    text = proposal.get("rewritten", "").strip()
    if not text:
        raise ValueError("An edited response is required.")
    target = {**copy.deepcopy(contract["empty_response"]), "narration": text}
    Draft202012Validator(contract["schema"]).validate(target)
    dialogue = [{"ordinal": i, "speaker": s["role"] + "-text-attributed", "role": s["role"],
                 "text": s["quote"], "visibility": "public"} for i,s in enumerate(context, 1)]
    body = {"dialogue": dialogue, "new_player_input": "\n".join(s["quote"] for s in located["participant"]),
            "private_facilitator_direction": "", "rules": [], "documents": [],
            "historical_claims_and_hypotheses": [], "historical_source_passages": [],
            "instruction": "Respond to the current participants. Suggestions do not establish observed facts."}
    # Local ordinals index the selected speech, with original coordinates retained
    # separately. Both same-turn voice changes and source chronology remain clear.
    context_ordinals = list(range(1, len(context) + 1))
    target_ordinals = list(range(len(context) + 1, len(context) + 1 + len(targets)))
    provenance = {"group": source["group"], "family": source.get("family", source["group"]),
                  "split": source["split"], "source_identity": source["source_identity"],
                  "source_sha256": digest(packed(source["turns"])),
                  "context_turns": context_ordinals, "protected_context_turns": context_ordinals,
                  "target_turn": target_ordinals[0], "target_turns": target_ordinals,
                  "original_context_spans": context, "original_target_spans": targets,
                  "source_response_sha256": digest(packed(targets)),
                  "label_origin": "model-edited", "role_origin": "text-inferred; audio unverified"}
    result = {"prompt": [{"role": "system", "content": contract["system"]},
                         {"role": "user", "content": packed(body)}],
              "target": target, "schema": contract["schema"], "provenance": provenance,
              "editorial_notes": {k: proposal[k] for k in ("skill", "reason", "creative_scope")},
              "review": {"verdict": "pending"}}
    result["id"] = digest(packed(result))
    return result
