"""Mine response exchanges, preserve originals, and export explicitly reviewed edits.

This offline workshop never changes a source library, infers speaker roles, or
turns a screening heuristic into a training label. Outputs belong outside Git.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import html
import json
from pathlib import Path
import random
import re

from jsonschema import Draft202012Validator
from format_conversation_dataset.pipeline import verify_splits

from .data import private_output, write_json
from .util import digest, packed, now

SPLITS = ("train", "validation", "test")
CHECKS = ("roles_clear", "visibility_clear", "context_sufficient", "response_complete", "facts_respected",
          "player_agency", "useful_response")
SKILLS = ("discovery", "npc_dialogue", "tension", "consequence", "scene_transition",
          "clarification", "factual_answer", "multiple_players", "task_change")


def fingerprint(case):
    return digest(packed(case))


def repair_sources(document, corrections):
    """Apply reviewed, lossless speaker-boundary edits to a separate source copy.

    Every character stays in order, including whitespace and rejected speech.
    Unknown spans remain unknown; repairing one turn never labels a whole speaker.
    These text-based judgments do not establish acoustic speaker accuracy.
    """
    if corrections.get("source_snapshot_sha256") != fingerprint(document):
        raise ValueError("Source snapshot changed before boundary repair.")
    result = copy.deepcopy(document)
    sources = {s["source_identity"]: s for s in result["sources"]}
    if len(sources) != len(result["sources"]):
        raise ValueError("Duplicate source identity.")
    indexed = {(s["source_identity"], t["ordinal"]): t for s in result["sources"] for t in s["turns"]}
    replacements = {}
    for correction in corrections["turns"]:
        key = (correction["source_identity"], correction["ordinal"])
        if key in replacements or key not in indexed:
            raise ValueError("Unknown or duplicate source turn repair.")
        turn = indexed[key]
        if correction.get("turn_sha256") != fingerprint(turn):
            raise ValueError("Source turn changed before boundary repair.")
        if (correction.get("origin") not in {"human", "model"}
                or not correction.get("reviewer", "").strip()
                or not correction.get("reason", "").strip()):
            raise ValueError("Boundary repairs need an explicit origin, reviewer and reason.")
        end = 0
        spans = []
        for span in correction["spans"]:
            if (type(span.get("start")) is not int or type(span.get("end")) is not int
                    or span["start"] != end or not end < span["end"] <= len(turn["text"])
                    or span.get("role") not in {"facilitator", "player", "observer", "unknown", "mixed"}
                    or not isinstance(span.get("speaker"), str) or not span["speaker"].strip()):
                raise ValueError("Repair spans must cover exact consecutive characters with explicit roles.")
            item = {**turn, "text": turn["text"][span["start"]:span["end"]],
                    "role": span["role"], "speaker": span["speaker"],
                    "source_span": {"ordinal": turn["ordinal"], "start": span["start"], "end": span["end"],
                                    "turn_sha256": fingerprint(turn)},
                    "boundary_review": {k: correction[k] for k in ("origin", "reviewer", "reason")}}
            # Word-level times cannot be inferred from character positions.
            if len(correction["spans"]) > 1:
                item["source_time_range"] = {k: turn[k] for k in ("start", "end") if k in turn}
                item.pop("start", None)
                item.pop("end", None)
            spans.append(item)
            end = span["end"]
        if end != len(turn["text"]) or not spans:
            raise ValueError("Boundary repair must preserve every source character.")
        replacements[key] = spans
    for source in result["sources"]:
        repaired = []
        for turn in source["turns"]:
            values = replacements.get((source["source_identity"], turn["ordinal"]), [turn])
            for value in values:
                item = copy.deepcopy(value)
                item.setdefault("source_span", {"ordinal": turn["ordinal"], "start": 0,
                                                 "end": len(turn["text"]), "turn_sha256": fingerprint(turn)})
                item["ordinal"] = len(repaired) + 1
                repaired.append(item)
        source["turns"] = repaired
    result["boundary_repair"] = {"source_snapshot_sha256": fingerprint(document),
                                  "corrections_sha256": fingerprint(corrections), "turns": len(replacements)}
    return result


def collect(document, contract, *, context_turns=24):
    """Only an actual player stimulus followed by a contiguous facilitator reply.

    Roles are attached to turns, not assumed constant for a diarization label.
    Context includes the preceding facilitator reply and the entire stimulus.
    Future turns and the reference response are never inserted into the prompt.
    """
    if type(context_turns) is not int or context_turns < 1:
        raise ValueError("Choose a positive context window.")
    if not isinstance(contract.get("system"), str) or not contract["system"].strip():
        raise ValueError("Supply the actual writer system instruction.")
    schema = contract["schema"]
    Draft202012Validator.check_schema(schema)
    template = contract["empty_response"]
    if not isinstance(template, dict) or "narration" not in template:
        raise ValueError("Supply the writer's empty response, including narration.")
    ownership, seen_sources = {}, set()
    cases, omitted, role_counts = [], Counter(), Counter()
    for source in document["sources"]:
        group, split, identity = source["group"], source["split"], source["source_identity"]
        if split not in SPLITS or not group or not identity:
            raise ValueError("Every source requires an explicit group, split and identity.")
        if identity in seen_sources:
            raise ValueError("Choose one transcript variant per source identity.")
        seen_sources.add(identity)
        turns = source["turns"]
        source_hash = digest(packed(turns))
        for key in [("group", group), ("source", source_hash),
                    ("family", source.get("family", group))]:
            if key in ownership and ownership[key] != split:
                raise ValueError("A story family or source crosses evaluation splits.")
            ownership[key] = split
        ordinals = [t["ordinal"] for t in turns]
        if any(type(n) is not int or n < 1 for n in ordinals) or ordinals != sorted(set(ordinals)):
            raise ValueError("Turn ordinals must be unique, positive and increasing.")
        for t in turns:
            if (t.get("role") not in {"facilitator", "player", "observer", "unknown", "mixed"}
                    or not isinstance(t.get("text"), str) or not isinstance(t.get("speaker"), str)):
                raise ValueError("Every turn needs text, speaker and an explicit role.")
            role_counts[t["role"]] += 1
        i = 0
        while i < len(turns):
            if turns[i]["role"] != "facilitator":
                i += 1
                continue
            end = i + 1
            while (end < len(turns) and turns[end]["role"] == "facilitator"
                   and turns[end]["speaker"] == turns[i]["speaker"]
                   and turns[end]["ordinal"] == turns[end - 1]["ordinal"] + 1):
                end += 1
            targets = turns[i:end]
            start = i
            while start > 0 and turns[start - 1]["role"] != "facilitator":
                start -= 1
            stimulus = turns[start:i]
            protected_start = start
            if start:
                prior_speaker = turns[start - 1]["speaker"]
                while (protected_start > 0 and turns[protected_start - 1]["role"] == "facilitator"
                       and turns[protected_start - 1]["speaker"] == prior_speaker):
                    protected_start -= 1
            context = turns[min(max(0, i - context_turns), protected_start):i]
            text = " ".join(t["text"] for t in targets)
            reason = None
            if not stimulus or not any(t["role"] == "player" for t in stimulus):
                reason = "no_player_stimulus"
            elif any(t["role"] in {"unknown", "mixed"} or not t["speaker"].strip()
                     or t["speaker"].lower() == "unknown" for t in context):
                reason = "unresolved_context_roles"
            elif any(t.get("category") in {"production", "chatter"} or t.get("status") == "excluded"
                     for t in targets):
                reason = "excluded_source_response"
            elif any(t.get("visibility", "public") != "public" for t in context + targets):
                reason = "private_exchange_requires_separate_context"
            elif not 8 <= len(text.split()) <= 350:
                reason = "response_length_outside_mining_window"
            elif any(b["ordinal"] != a["ordinal"] + 1 for a, b in zip(context + targets, (context + targets)[1:])):
                reason = "missing_source_turns"
            if reason:
                omitted[reason] += 1
                i = end
                continue
            dialogue = [{k: t[k] for k in ("ordinal", "speaker", "role", "text", "character") if k in t}
                        for t in context]
            for turn in dialogue:
                turn["visibility"] = "public"
            body = {"dialogue": dialogue,
                    "new_player_input": "\n".join(t["text"] for t in stimulus if t["role"] == "player"),
                    "private_facilitator_direction": "", "rules": [], "documents": [],
                    "historical_claims_and_hypotheses": [], "historical_source_passages": [],
                    "instruction": "Respond to the current participants. Suggestions do not establish observed facts."}
            target = {**copy.deepcopy(template), "narration": text}
            Draft202012Validator(schema).validate(target)
            provenance = {"group": group, "family": source.get("family", group), "split": split,
                          "source_identity": identity, "source_sha256": source_hash,
                          "target_turn": targets[0]["ordinal"], "target_turns": [t["ordinal"] for t in targets],
                          "context_turns": [t["ordinal"] for t in context],
                          "protected_context_turns": [t["ordinal"] for t in turns[protected_start:i]],
                          "source_response_sha256": digest(text)}
            flags = []
            if text.rstrip().endswith(("...", "…", ",")):
                flags.append("possibly_incomplete")
            if re.search(r"\b(?:roll|dice|difficulty|damage|success|failure)\b", text, re.I):
                flags.append("check_mechanics_and_resolution")
            if re.search(r"\b(?:podcast|episode|sponsor|subscribe)\b", text, re.I):
                flags.append("check_production_talk")
            case = {"provenance": provenance, "schema_sha256": fingerprint(schema),
                    "prompt": [{"role": "system", "content": contract["system"]},
                    {"role": "user", "content": packed(body)}], "original": target,
                    "source_context": context, "source_response": targets, "flags": flags}
            case["id"] = fingerprint(case)
            cases.append(case)
            i = end
    return {"version": 1, "created": now(), "schema": schema, "contract_sha256": fingerprint(contract),
            "source_snapshot_sha256": fingerprint(document), "cases": cases,
            "audit": {"source_collections": len(seen_sources), "roles": dict(role_counts),
                      "candidates": len(cases), "omitted": dict(omitted)}}


def sample(document, split, limit, *, seed=42):
    """Round-robin story groups and source recordings; avoid one episode dominating."""
    if split not in SPLITS or type(limit) is not int or limit < 1:
        raise ValueError("Choose a split and positive review limit.")
    buckets = defaultdict(lambda: defaultdict(list))
    for case in document["cases"]:
        p = case["provenance"]
        if p["split"] == split:
            buckets[p["group"]][p["source_identity"]].append(case)
    rng = random.Random(seed)
    groups = sorted(buckets)
    rng.shuffle(groups)
    for sources in buckets.values():
        for cases in sources.values():
            rng.shuffle(cases)
    result = []
    while len(result) < limit:
        before = len(result)
        for group in groups:
            sources = buckets[group]
            # Rotate recordings within each group, one case per group per round.
            for identity in list(sources):
                cases = sources.pop(identity)
                if cases:
                    result.append(cases.pop())
                    sources[identity] = cases
                    break
            if len(result) == limit:
                break
        if len(result) == before:
            break
    return result


def review_template(cases):
    return [{"id": c["id"], "case_sha256": fingerprint(c), "verdict": "pending",
             "origin": "model", "reviewer": "", "reason": "", "skills": [],
             "creative_scope": "", "checks": {k: None for k in CHECKS},
             "reviewed_target_sha256": None,
             "target": copy.deepcopy(c["original"])} for c in cases]


def export_rows(document, decisions, *, allow_model_reviews=False):
    by_id = {c["id"]: c for c in document["cases"]}
    if len(by_id) != len(document["cases"]):
        raise ValueError("Duplicate workshop case.")
    for case in by_id.values():
        if case["id"] != fingerprint({k: v for k, v in case.items() if k != "id"}):
            raise ValueError("Frozen source changed; rebuild the workshop and review it again.")
        if case.get("schema_sha256") != fingerprint(document["schema"]):
            raise ValueError("Writer schema changed; rebuild the workshop and review it again.")
    result = {s: [] for s in SPLITS}
    seen = set()
    for review in decisions:
        key = review["id"]
        if key in seen or key not in by_id:
            raise ValueError("Unknown or duplicate review ID.")
        seen.add(key)
        case = by_id[key]
        if review.get("case_sha256") != fingerprint(case):
            raise ValueError("Source or prompt changed after review.")
        if review.get("verdict") not in {"pending", "keep", "reject"}:
            raise ValueError("Review verdict must be pending, keep, or reject.")
        if review["verdict"] != "keep":
            continue
        if review.get("origin") not in {"human", "model"} or not str(review.get("reviewer", "")).strip():
            raise ValueError("Record the actual review origin and reviewer.")
        if review["origin"] == "model" and not allow_model_reviews:
            raise ValueError("Model-reviewed labels need explicit allow_model_reviews.")
        if any(review.get("checks", {}).get(k) is not True for k in CHECKS):
            raise ValueError("A kept response needs all semantic review checks.")
        if not review.get("reason", "").strip() or not review.get("creative_scope", "").strip():
            raise ValueError("Explain the edit and the permitted creative space.")
        if not review.get("skills") or set(review["skills"]) - set(SKILLS):
            raise ValueError("Assign at least one recognized response skill.")
        target = review["target"]
        if review.get("reviewed_target_sha256") != fingerprint(target):
            raise ValueError("Target changed or was not bound to its review.")
        Draft202012Validator(document["schema"]).validate(target)
        if not any(isinstance(v, str) and v.strip() or isinstance(v, list) and v for v in target.values()):
            raise ValueError("An empty structured response is not a useful training target.")
        p = {**case["provenance"], "workshop_case": key, "source_case_sha256": fingerprint(case),
             "target_sha256": fingerprint(target), "label_origin": "source" if target == case["original"] else review["origin"] + "-edited",
             "review_level": review["origin"] + "-reviewed", "review": copy.deepcopy(review)}
        row = {"id": digest(packed(case["prompt"]) + packed(target)), "task": "storyteller",
               "prompt": copy.deepcopy(case["prompt"]), "completion": [{"role": "assistant", "content": packed(target)}],
               "schema": document["schema"], "provenance": p}
        result[p["split"]].append(row)
    verify_splits(result)
    families = {}
    for split, rows in result.items():
        for row in rows:
            family = row["provenance"]["family"]
            if family in families and families[family] != split:
                raise ValueError("A story family crosses training/evaluation splits.")
            families[family] = split
    return result


def finalize_reviews(document, decisions, *, allow_model_reviews=False):
    """Explicitly sign off edited decisions; never fill in semantic checks."""
    result = copy.deepcopy(decisions)
    for review in result:
        if review.get("verdict") == "keep":
            review["reviewed_target_sha256"] = fingerprint(review["target"])
    export_rows(document, result, allow_model_reviews=allow_model_reviews)
    return result


def render(cases, decisions, output):
    esc = lambda x: html.escape(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=2))
    reviews = {d["id"]: d for d in decisions}
    parts = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
             '<title>Storyteller training workshop</title><style>body{max-width:1100px;margin:auto;padding:24px;background:#f5eee3;color:#34291f;font:17px/1.5 system-ui}a{color:#754523}nav{position:sticky;top:0;background:#f5eee3;padding:10px;display:flex;gap:12px;flex-wrap:wrap}article{padding:24px 0;border-top:1px solid #c9bca9}pre{white-space:pre-wrap;overflow-wrap:anywhere}blockquote{margin:12px 0;padding:12px;background:#fffbf4}.target{border-left:4px solid #89643e}</style>',
             '<h1>Storyteller training workshop</h1><p>Private source exchanges and editorial targets. Source text is preserved. Checks and edits are reviewer judgments, not acoustic verification or independent gold.</p><nav>']
    parts.extend(f'<a href="#case-{i}">{i+1}</a>' for i in range(len(cases)))
    parts.append('</nav>')
    for i, c in enumerate(cases):
        p = c["provenance"]
        r = reviews.get(c["id"], {"verdict": "pending", "target": c["original"]})
        parts.append(f'<article id="case-{i}"><h2>{i+1}. {esc(r["verdict"])} · {esc(p["split"])}</h2><p>{esc(p["group"])} · response {esc(p["target_turns"])} · {esc(c["flags"])}</p>')
        for t in c["source_context"]:
            parts.append(f'<p><strong>{t["ordinal"]} · {esc(t["role"])} · {esc(t["speaker"])}</strong><br>{esc(t["text"])}</p>')
        parts.append(f'<h3>Original response</h3><blockquote>{esc(c["original"]["narration"])}</blockquote><h3>Training target</h3><pre class="target">{esc(r["target"])}</pre><h3>Review</h3><pre>{esc({k:v for k,v in r.items() if k != "target"})}</pre></article>')
    Path(output).write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    repair = sub.add_parser("repair")
    repair.add_argument("source", type=Path)
    repair.add_argument("corrections", type=Path)
    repair.add_argument("output", type=Path)
    mine = sub.add_parser("collect")
    mine.add_argument("source", type=Path)
    mine.add_argument("contract", type=Path)
    mine.add_argument("output", type=Path)
    mine.add_argument("--context-turns", type=int, default=24)
    rev = sub.add_parser("review")
    rev.add_argument("workshop", type=Path)
    rev.add_argument("output", type=Path)
    rev.add_argument("--split", choices=SPLITS, default="train")
    rev.add_argument("--limit", type=int, default=100)
    rev.add_argument("--seed", type=int, default=42)
    out = sub.add_parser("export")
    out.add_argument("workshop", type=Path)
    out.add_argument("reviews", type=Path)
    out.add_argument("output", type=Path)
    out.add_argument("--allow-model-reviews", action="store_true")
    finalize = sub.add_parser("finalize")
    finalize.add_argument("workshop", type=Path)
    finalize.add_argument("reviews", type=Path)
    finalize.add_argument("output", type=Path)
    finalize.add_argument("--allow-model-reviews", action="store_true")
    args = parser.parse_args()
    destination = private_output(args.output)
    if destination.exists():
        raise ValueError("Use a new output directory; preserve previous review snapshots.")
    if args.command == "repair":
        result = repair_sources(json.loads(args.source.read_text()), json.loads(args.corrections.read_text()))
        destination.mkdir(parents=True)
        write_json(destination / "sources.json", result, exclusive=True)
        print(packed(result["boundary_repair"]))
    elif args.command == "collect":
        result = collect(json.loads(args.source.read_text()), json.loads(args.contract.read_text()), context_turns=args.context_turns)
        destination.mkdir(parents=True)
        write_json(destination / "workshop.json", result, exclusive=True)
        print(packed(result["audit"]))
    elif args.command == "review":
        doc = json.loads(args.workshop.read_text())
        cases = sample(doc, args.split, args.limit, seed=args.seed)
        decisions = review_template(cases)
        destination.mkdir(parents=True)
        write_json(destination / "reviews.json", decisions, exclusive=True)
        render(cases, decisions, destination / "review.html")
        print(packed({"selected": len(cases), "split": args.split}))
    else:
        doc = json.loads(args.workshop.read_text())
        reviews = json.loads(args.reviews.read_text())
        if args.command == "finalize":
            reviews = finalize_reviews(doc, reviews, allow_model_reviews=args.allow_model_reviews)
            destination.mkdir(parents=True)
            write_json(destination / "reviews.json", reviews, exclusive=True)
            selected = {r["id"] for r in reviews}
            render([c for c in doc["cases"] if c["id"] in selected], reviews, destination / "review.html")
            print(packed(dict(Counter(r["verdict"] for r in reviews))))
            return
        rows = export_rows(doc, reviews, allow_model_reviews=args.allow_model_reviews)
        destination.mkdir(parents=True)
        for split, items in rows.items():
            (destination / f"{split}.sources.jsonl").write_text("".join(packed(r) + "\n" for r in items))
        write_json(destination / "export.json", {"created": now(), "workshop_sha256": fingerprint(doc),
                   "reviews_sha256": fingerprint(reviews), "counts": {s:len(v) for s,v in rows.items()}}, exclusive=True)
        selected = {r["id"] for r in reviews}
        render([c for c in doc["cases"] if c["id"] in selected], reviews, destination / "review.html")
        print(packed({s: len(v) for s, v in rows.items()}))


if __name__ == "__main__":
    main()
