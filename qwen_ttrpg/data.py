"""Bring-your-own sources, review-bound task labels, and three isolated snapshots."""

from __future__ import annotations

import copy
import html
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
from uuid import uuid4

from format_conversation_dataset.pipeline import read_source, response_examples, prepare, verify_splits
import yaml

from .contracts import exact_quote, validate_target, output_schema
from .tasks import POLICIES
from .util import digest, file_digest, now, packed

TASKS = ("classifier", "rules", "storyteller")


def private_output(path):
    path = Path(path).expanduser().resolve()
    if any((p / ".git").exists() for p in [path, *path.parents]):
        raise ValueError("Keep generated data, reviews, and model artifacts outside Git checkouts.")
    return path


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(packed(value) + "\n")
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(packed(value) + "\n", encoding="utf-8")
        temporary.replace(path)


def load_config(path):
    path = Path(path).expanduser().resolve(strict=True)
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict) or type(doc.get("version")) is not int or doc["version"] != 1:
        raise ValueError("Use a version: 1 experiment configuration.")
    for flag in ["allow_model_reviews", "allow_synthetic"]:
        if flag in doc and type(doc[flag]) is not bool:
            raise ValueError(f"{flag} must be a YAML boolean, not a quoted string.")
    for key in ["max_tokens", "reload_cases", "startup_timeout"]:
        value = doc.get("evaluation", {}).get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"evaluation.{key} must be a positive integer.")
    port = doc.get("serving", {}).get("port", 8092)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("serving.port must be an integer from 1 to 65535.")
    config = copy.deepcopy(doc)
    config["_path"] = str(path)
    config["workspace"] = str(private_output(path.parent / config["workspace"]))
    for key in ["model", "base_gguf", "runtime", "python"]:
        if config.get(key):
            candidate_path = path.parent / Path(config[key]).expanduser()
            # Resolving a venv's python symlink selects the base interpreter and
            # silently loses the training environment's installed dependencies.
            config[key] = str(candidate_path.absolute() if key == "python" else candidate_path.resolve())
    if set(config.get("tasks", {})) - set(TASKS):
        raise ValueError("Unknown training task.")
    if not config.get("model"):
        raise ValueError("Supply an already-downloaded local model/tokenizer directory.")
    return config


def source_path(config, value):
    return (Path(config["_path"]).parent / Path(value).expanduser()).resolve(strict=True)


def task_messages(task, body, context=""):
    return [{"role": "system", "content": POLICIES[task] + ("\n" + context if context else "")},
            {"role": "user", "content": packed(body)}]


def candidate(task, body, provenance, config, target=None):
    prompt = task_messages(task, body, config.get("system_context", ""))
    identity = digest(packed({"task": task, "prompt": prompt, "provenance": provenance}))
    return {"id": identity, "task": task, "input": body, "prompt": prompt,
            "provenance": provenance, "schema": output_schema(task), "target": target,
            "label_origin": "source" if target is not None else "unlabeled",
            "review": {"verdict": "pending", "origin": None, "reviewer": "", "target_sha256": None,
                       "reason": "", "response_complete": False}}


def build_candidates(config):
    inputs, result, groups, identities = {}, [], {}, {}

    def read(value):
        path = source_path(config, value)
        inputs[str(path)] = file_digest(path)
        return path

    def check_group(group, split, identity, fingerprint):
        if not group or split not in {"train", "validation", "test"}:
            raise ValueError("Every source needs a group and explicit train/validation/test split.")
        if group in groups and groups[group] != split:
            raise ValueError("A source group crosses training/evaluation splits.")
        groups[group] = split
        if identity in identities and identities[identity] != (fingerprint, split):
            raise ValueError("A source identity has alternate variants or conflicting splits.")
        identities[identity] = (fingerprint, split)

    for source in config.get("conversations", []):
        path = read(source["path"])
        if source.get("format", "speaker-text") == "turns-json":
            doc = json.loads(path.read_text())
            turns = doc["turns"]
            if not turns or any(not isinstance(t.get("text"), str) or not isinstance(t.get("speaker"), str) for t in turns):
                raise ValueError("Each conversation turn requires speaker and text strings.")
        else:
            turns, _ = read_source(path)
        speakers = source["speakers"]
        if any(role not in {"facilitator", "player", "observer"} for role in speakers.values()):
            raise ValueError("Speaker mappings use facilitator, player, or observer roles.")
        turns = [dict(t, turn=i, role=speakers.get(t["speaker"], "unknown")) for i, t in enumerate(turns, 1)]
        source_metadata = {str(t["turn"]): {k: v for k, v in t.items() if k not in {"turn", "speaker", "role", "text", "character"}}
                           for t in turns}
        source_metadata = {k: v for k, v in source_metadata.items() if v}
        # ASR word timings and confidence scores remain in provenance. Repeating
        # every word as both dialogue and alignment metadata wastes the model's
        # context window and can displace the response's actual stimulus.
        turns = [{k: v for k, v in t.items() if k in {"turn", "speaker", "role", "text", "character"}} for t in turns]
        fingerprint = inputs[str(path)]
        identity = source.get("source_identity", fingerprint)
        group, split = source["group"], source["split"]
        check_group(group, split, identity, fingerprint)
        provenance = {"group": group, "split": split, "source_identity": identity,
                      "source_sha256": fingerprint, "input_path": str(path)}
        if source_metadata:
            provenance["source_turn_metadata"] = source_metadata
        options = config.get("tasks", {}).get("classifier", {})
        width, before = options.get("window_turns", 8), options.get("context_turns", 8)
        if type(width) is not int or width < 1 or type(before) is not int or before < 0:
            raise ValueError("Classifier windows must be positive, with nonnegative preceding context.")
        for offset in range(0, len(turns), width):
            window = turns[offset:offset + width]
            prior = [e for e in source.get("prior_events", []) if e["turn"] < window[0]["turn"]]
            body = {"target_turns": window, "context_only": turns[max(0, offset-before):offset], "prior_events": prior}
            p = {**provenance, "target_turn": window[0]["turn"], "target_turns": [t["turn"] for t in window],
                 "context_turns": [t["turn"] for t in body["context_only"]]}
            result.append(candidate("classifier", body, p, config))
        assistants = [speaker for speaker, role in speakers.items() if role == "facilitator"]
        if not assistants:
            continue
        _, responses, _ = response_examples(turns, group=group, split=split, source_sha256=fingerprint,
            assistant_speakers=assistants, system=POLICIES["storyteller"],
            context_turns=config.get("tasks", {}).get("storyteller", {}).get("context_turns", 64))
        for response in responses:
            if not response["has_stimulus"] or any(t["role"] == "unknown" for t in response["context"]):
                continue
            if not response["context"] or response["context"][-1]["role"] != "player":
                continue
            body = {"recent_dialogue": response["context"], "instruction": "Respond to the latest player contribution. Return narration as JSON."}
            p = {**provenance, "target_turn": response["target_turns"][0],
                 "target_turns": response["target_turns"], "context_turns": [t["turn"] for t in response["context"]],
                 "protected_context_turns": response["protected_context_turns"], "source_response_sha256": response["target_sha256"]}
            result.append(candidate("storyteller", body, p, config, {"narration": response["response"]}))

    documents = {}
    for item in config.get("rule_documents", []):
        if item["id"] in documents:
            raise ValueError("Duplicate rule document ID.")
        path = read(item["path"])
        if item.get("format", "text") == "pdf":
            # Reading order avoids interleaved PDF columns; raw extraction stays local.
            text = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
        else:
            text = path.read_text(encoding="utf-8")
        documents[item["id"]] = {"text": text, "sha256": inputs[str(path)], "path": str(path)}
    excerpts = {}
    for item in config.get("rule_excerpts", []):
        if item["id"] in excerpts:
            raise ValueError("Duplicate rule excerpt ID.")
        document = documents[item["document"]]
        text = document["text"]
        if item.get("page") is not None:
            pages = text.split("\f")
            if type(item["page"]) is not int or not 1 <= item["page"] <= len(pages):
                raise ValueError("Invalid one-based rule page.")
            text = pages[item["page"] - 1]
        quote = exact_quote(text, item["quote"])
        excerpts[item["id"]] = {"id": item["id"], "text": quote, "document_sha256": document["sha256"],
                                "excerpt_sha256": digest(quote), "page": item.get("page")}
    questions = []
    if config.get("rule_questions"):
        questions = json.loads(read(config["rule_questions"]).read_text())
    seen_questions = set()
    for item in questions:
        if item["id"] in seen_questions:
            raise ValueError("Duplicate question ID.")
        seen_questions.add(item["id"])
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError("Each rules example needs a question.")
        references = [excerpts[key] for key in item["references"]]
        body = {"question": item["question"], "rules": references,
                "tools": config.get("tools", []), "tool_inputs": item.get("tool_inputs", {})}
        # The question is the held-out unit. Shared reference material is reported
        # separately; it must never be described as an unseen-document evaluation.
        fingerprint = digest(item["question"].strip())
        check_group(item["group"], item["split"], "question:" + item["id"], fingerprint)
        p = {"group": item["group"], "split": item["split"], "source_sha256": fingerprint,
             "source_identity": "question:" + item["id"], "question_id": item["id"],
             "target_turn": 1, "target_turns": [1], "context_turns": [],
             "reference_hashes": [r["excerpt_sha256"] for r in references]}
        result.append(candidate("rules", body, p, config, item.get("target")))
    if len({c["id"] for c in result}) != len(result):
        raise ValueError("Duplicate candidate/source entries.")
    snapshot = {"inputs": inputs, "candidates": result}
    return {"version": 1, "created": now(), "source_snapshot_sha256": digest(packed(snapshot)), **snapshot}


def review_path(config):
    return Path(config["workspace"]) / "review.json"


def candidates(config):
    built = build_candidates(config)
    path = review_path(config)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["source_snapshot_sha256"] != built["source_snapshot_sha256"]:
            raise ValueError("Source/configuration changed; use a new workspace instead of reusing stale reviews.")
        return path
    write_json(path, built, exclusive=True)
    render_review(built, path.with_suffix(".html"))
    return path


def load_review(config):
    original = build_candidates(config)
    document = json.loads(review_path(config).read_text())
    if document["source_snapshot_sha256"] != original["source_snapshot_sha256"]:
        raise ValueError("Sources or task instructions changed after review preparation.")
    expected = {c["id"]: c for c in original["candidates"]}
    if len(document["candidates"]) != len(expected) or {c["id"] for c in document["candidates"]} != set(expected):
        raise ValueError("Review candidates were added, removed, or duplicated.")
    mutable = {"target", "review", "label_origin", "annotation"}
    for case in document["candidates"]:
        source = expected[case["id"]]
        if {k: v for k, v in case.items() if k not in mutable} != {k: v for k, v in source.items() if k not in mutable}:
            raise ValueError("Review inputs changed; edit only target/review fields, not source context.")
        if case["task"] == "storyteller" and case["target"] != source["target"]:
            raise ValueError("Narrator targets must preserve their source wording.")
    return document


def review(config, ids, *, verdict, origin, reviewer, reason, task=None, response_complete=False):
    if verdict not in {"keep", "reject"} or origin not in {"human", "model", "synthetic"} or not reviewer.strip() or not reason.strip():
        raise ValueError("Record keep/reject, the actual review origin, reviewer, and reasoning.")
    doc = load_review(config)
    previous = copy.deepcopy(doc)
    selected = [c for c in doc["candidates"] if (not task or c["task"] == task) and (ids is None or c["id"] in ids)]
    if not selected or (ids is not None and {c["id"] for c in selected} != set(ids)):
        raise ValueError("Review selection includes an unknown case or task.")
    updates = []
    for case in selected:
        if verdict == "keep":
            target = validate_target(case["task"], case["input"], case["target"])
            if case["task"] == "storyteller" and not response_complete:
                raise ValueError("Confirm that narrator targets are complete responses to the players.")
        else:
            target = case["target"]
        updates.append((case, target))
    for case, target in updates:
        case["target"] = target
        if case["label_origin"] == "unlabeled" and verdict == "keep":
            case["label_origin"] = origin + "-authored"
        case["review"] = {"verdict": verdict, "origin": origin, "reviewer": reviewer, "reason": reason,
                          "target_sha256": digest(packed(target)), "response_complete": response_complete}
    history = Path(config["workspace"]) / "review-history" / (uuid4().hex + ".json")
    write_json(history, {"created": now(), "action": "review", "previous": previous, "next": doc}, exclusive=True)
    write_json(review_path(config), doc)
    render_review(doc, review_path(config).with_suffix(".html"))
    return len(updates)


def render_review(document, output):
    esc = lambda x: html.escape(x if isinstance(x, str) else json.dumps(x, indent=2, ensure_ascii=False))
    parts = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Training label review</title>',
             '<style>body{max-width:1100px;margin:auto;padding:24px;font:17px system-ui;background:#f7f1e7;color:#332b25}pre{white-space:pre-wrap;overflow-wrap:anywhere}code{overflow-wrap:anywhere}article{padding:24px 0;border-top:1px solid #bba}nav{display:flex;flex-wrap:wrap;gap:12px}</style>',
             '<h1>Training label review</h1><p>Check source evidence, response boundaries, and label meaning. Model proposals are unreviewed until a decision is recorded.</p><nav>']
    parts += [f'<a href="#c-{i}">{i+1}</a>' for i in range(len(document['candidates']))]
    parts.append('</nav>')
    for i, case in enumerate(document['candidates']):
        parts.extend([f'<article id="c-{i}"><h2>{i+1}. {esc(case["task"])} · {esc(case["provenance"]["split"])}</h2>',
                      f'<p>Case ID: <code>{esc(case["id"])}</code></p><h3>Input evidence</h3><pre>{esc(case["input"])}</pre>',
                      f'<h3>Proposed target</h3><pre>{esc(case["target"])}</pre><h3>Review</h3><pre>{esc(case["review"])}</pre></article>'])
    Path(output).write_text(''.join(parts), encoding='utf-8')


def prepare_all(config, tokenizer):
    doc = load_review(config)
    by_task = {task: {s: [] for s in ["train", "validation", "test"]} for task in TASKS}
    omitted = {task: {} for task in TASKS}
    for case in doc["candidates"]:
        task, p, decision = case["task"], case["provenance"], case["review"]
        if decision.get("verdict") != "keep":
            omitted[task][case["id"]] = "not_kept"
            continue
        origin = decision.get("origin")
        if origin == "model" and not config.get("allow_model_reviews", False):
            raise ValueError("Model-reviewed examples require allow_model_reviews: true.")
        if origin == "synthetic" and not config.get("allow_synthetic", False):
            raise ValueError("Synthetic fixtures require allow_synthetic: true.")
        if origin not in {"human", "model", "synthetic"} or not decision.get("reviewer") or not decision.get("reason"):
            raise ValueError("Every kept target needs an explicit review origin, reviewer, and reason.")
        target = validate_target(task, case["input"], case["target"])
        if decision.get("target_sha256") != digest(packed(target)):
            raise ValueError("Target changed after review; record a fresh review.")
        if task == "storyteller" and not decision.get("response_complete"):
            raise ValueError("A storyteller response needs a complete-response review.")
        row = {"id": digest(packed(case["prompt"]) + packed(target)), "task": task, "prompt": case["prompt"],
               "completion": [{"role": "assistant", "content": packed(target)}], "schema": case["schema"],
               "provenance": {**p, "candidate_id": case["id"], "label_origin": case["label_origin"],
                              "review_level": origin + "-reviewed", "review": decision}}
        by_task[task][p["split"]].append(row)
    # Check common source identities across tasks too; the same conversation may
    # support multiple adapters, but cannot train one and evaluate another.
    combined = {split: [r for task in TASKS for r in by_task[task][split]] for split in ["train", "validation", "test"]}
    verify_splits(combined, check_response_duplicates=False)
    for task in TASKS:
        if not by_task[task]["train"] or not by_task[task]["validation"]:
            raise ValueError(f"{task} needs reviewed, independent training and validation examples.")
    from .experiment import preparation_identity
    key = preparation_identity(config, doc)
    root = Path(config["workspace"]) / "datasets"
    if root.exists():
        prior = json.loads((root / "prepared.json").read_text()) if (root / "prepared.json").exists() else {}
        if prior.get("preparation_sha256") != key:
            raise ValueError("Prepared data differs from the current reviews/settings; choose a new workspace.")
        from .train import verify_dataset
        for task in TASKS:
            verify_dataset(root / task)
        return prior
    staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=root.parent))
    reports = {}
    try:
        for task in TASKS:
            reports[task] = prepare(by_task[task], staging / task, tokenizer,
                config.get("tasks", {}).get(task, {}).get("sequence_len", 4096),
                check_response_duplicates=task == "storyteller")
        reference_splits = {}
        for split, rows in by_task["rules"].items():
            for row in rows:
                for reference in row["provenance"].get("reference_hashes", []):
                    reference_splits.setdefault(reference, set()).add(split)
        report = {"preparation_sha256": key, "source_snapshot_sha256": doc["source_snapshot_sha256"], "tasks": reports,
                  "omitted_candidates": omitted, "shared_rule_excerpts": sum(len(s) > 1 for s in reference_splits.values()),
                  "rules_evaluation_scope": "Held-out question groups; shared supplied references do not constitute unseen-document evaluation."}
        write_json(staging / "prepared.json", report, exclusive=True)
        if root.exists():
            raise ValueError("Dataset destination appeared during preparation.")
        staging.rename(root)
        return report
    except BaseException:
        write_json(root.parent / "prepare-failure.json", {"status": "incomplete", "created": now()})
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
