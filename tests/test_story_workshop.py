import copy
import json

import pytest

from qwen_ttrpg.story_workshop import (
    CHECKS, collect, export_rows, finalize_reviews, fingerprint, render, repair_sources, review_template, sample,
)


def source(group="scene", split="train"):
    lines = [
        ("facilitator", "guide", "The ferry waits beside an empty landing. Its operator is watching the river."),
        ("player", "guest", "I ask the operator what he is watching for."),
        ("facilitator", "guide", "A beam from the broken bridge is floating toward the landing."),
        ("facilitator", "guide", "Keep clear of the water until it has passed, he says."),
        ("player", "guest", "I stay on the bank. Can I reach the mooring rope from here?"),
        ("facilitator", "guide", "The rope is looped around the post beside you, well above the water."),
    ]
    return {"group": group, "split": split, "source_identity": group,
            "turns": [{"ordinal": i, "role": r, "speaker": s, "text": t}
                      for i, (r,s,t) in enumerate(lines, 1)]}


def contract():
    return {"system": "Suggest a response; preserve player choices.",
            "schema": {"type": "object", "additionalProperties": False,
                       "required": ["narration"], "properties": {"narration": {"type": "string"}}},
            "empty_response": {"narration": ""}}


def kept(doc):
    reviews = review_template(doc["cases"])
    for r in reviews:
        r.update(verdict="keep", origin="model", reviewer="editor", reason="A complete world response.",
                 skills=["discovery"], creative_scope="Consistent local detail; no player decisions.",
                 checks={k: True for k in CHECKS}, reviewed_target_sha256=fingerprint(r["target"]))
    return reviews


def test_protects_exchange_and_keeps_future_and_target_out_of_prompt():
    doc = collect({"sources": [source()]}, contract(), context_turns=1)
    assert len(doc["cases"]) == 2
    first, second = doc["cases"]
    assert first["provenance"]["target_turns"] == [3,4]
    assert second["provenance"]["context_turns"] == [3,4,5]
    assert first["provenance"]["protected_context_turns"] == [1,2]
    body = json.loads(first["prompt"][-1]["content"])
    assert body["new_player_input"] == source()["turns"][1]["text"]
    assert "broken bridge" not in first["prompt"][-1]["content"]
    assert "mooring rope" not in first["prompt"][-1]["content"]


def test_does_not_assume_speaker_role_or_skip_unknown_stimulus():
    s = source()
    s["turns"][1]["role"] = "unknown"
    doc = collect({"sources": [s]}, contract())
    assert not doc["cases"]
    # A diarization label used by a player cannot be folded into a facilitator reply.
    s = source()
    s["turns"][3]["role"] = "player"
    doc = collect({"sources": [s]}, contract())
    assert doc["cases"][0]["provenance"]["target_turns"] == [3]


@pytest.mark.parametrize("change", ["family", "source", "identity", "gap", "private"])
def test_source_partition_and_visibility_guards(change):
    a, b = source(), source("other", "test")
    if change == "family":
        b["turns"][0]["text"] += " Later."
        a["family"] = b["family"] = "same-campaign"
    elif change == "identity":
        b["source_identity"] = a["source_identity"]
    elif change == "gap":
        a["turns"].pop(1)
        assert not collect({"sources": [a]}, contract())["cases"]
        return
    elif change == "private":
        a["turns"][0]["visibility"] = "private"
        assert not collect({"sources": [a]}, contract())["cases"]
        return
    with pytest.raises(ValueError):
        collect({"sources": [a,b]}, contract())


def test_edits_preserve_source_and_are_explicitly_model_reviewed():
    doc = collect({"sources": [source()]}, contract())
    original = copy.deepcopy(doc)
    reviews = kept(doc)
    reviews[0]["target"]["narration"] = "A broken beam drifts toward the landing. The operator tells you to keep clear."
    reviews[0]["reviewed_target_sha256"] = fingerprint(reviews[0]["target"])
    with pytest.raises(ValueError, match="allow_model_reviews"):
        export_rows(doc, reviews)
    rows = export_rows(doc, reviews, allow_model_reviews=True)
    assert rows["train"][0]["provenance"]["label_origin"] == "model-edited"
    assert rows["train"][1]["provenance"]["label_origin"] == "source"
    assert doc == original


@pytest.mark.parametrize("change", ["prompt", "target", "check", "duplicate", "skill", "schema", "pending"])
def test_review_cannot_silently_train_stale_or_unreviewed_data(change):
    doc = collect({"sources": [source()]}, contract())
    reviews = kept(doc)
    if change == "prompt":
        doc["cases"][0]["prompt"][1]["content"] += " Future answer."
    elif change == "target":
        reviews[0]["target"]["narration"] += " Everyone agrees to leave."
    elif change == "check":
        reviews[0]["checks"]["context_sufficient"] = False
    elif change == "duplicate":
        reviews.append(copy.deepcopy(reviews[0]))
    elif change == "skill":
        reviews[0]["skills"] = ["unrecognized"]
    elif change == "schema":
        reviews[0]["target"]["invalid"] = True
        reviews[0]["reviewed_target_sha256"] = fingerprint(reviews[0]["target"])
    elif change == "pending":
        rows = export_rows(doc, review_template(doc["cases"]))
        assert not any(rows.values())
        return
    with pytest.raises(Exception):
        export_rows(doc, reviews, allow_model_reviews=True)


def test_sampling_is_reproducible_and_balances_groups(tmp_path):
    a, b = source(), source("other")
    b["turns"][0]["text"] += " At sunset."
    doc = collect({"sources": [a,b]}, contract())
    cases = sample(doc, "train", 2)
    assert cases == sample(doc, "train", 2)
    assert len({c["provenance"]["group"] for c in cases}) == 2
    assert sample(doc, "test", 20) == []
    cases[0]["source_context"][0]["text"] = "<script>alert(1)</script>"
    render(cases, review_template(cases), tmp_path / "review.html")
    result = (tmp_path / "review.html").read_text()
    assert "&lt;script&gt;" in result and "<script>" not in result


def repair_fixture():
    document = {"sources": [source()]}
    t = document["sources"][0]["turns"][2]
    t["text"] = "I step back. The ferryman catches the rope."
    t.update(start=2.0, end=7.0)
    cut = t["text"].index("The")
    edits = {"source_snapshot_sha256": fingerprint(document), "turns": [{
        "source_identity": "scene", "ordinal": 3, "turn_sha256": fingerprint(t),
        "origin": "model", "reviewer": "editor", "reason": "Text changes from player declaration to world narration.",
        "spans": [{"start": 0, "end": cut, "role": "player", "speaker": "guest"},
                  {"start": cut, "end": len(t["text"]), "role": "facilitator", "speaker": "guide"}]}]}
    return document, edits


def test_boundary_repairs_are_lossless_local_and_keep_original_untouched():
    document, edits = repair_fixture()
    original = copy.deepcopy(document)
    fixed = repair_sources(document, edits)
    assert document == original
    old = document["sources"][0]["turns"]
    new = fixed["sources"][0]["turns"]
    assert "".join(t["text"] for t in new) == "".join(t["text"] for t in old)
    assert len(new) == len(old) + 1
    assert new[2]["role"] == "player" and new[3]["role"] == "facilitator"
    assert "start" not in new[3] and new[3]["source_time_range"] == {"start": 2.0, "end": 7.0}
    assert new[3]["source_span"]["ordinal"] == 3
    assert new[-1]["role"] == old[-1]["role"]
    assert fixed["sources"][0]["split"] == "train"


@pytest.mark.parametrize("change", ["gap", "overlap", "stale", "reviewer", "duplicate"])
def test_boundary_repair_refuses_loss_or_stale_labels(change):
    document, edits = repair_fixture()
    edit = edits["turns"][0]
    if change == "gap":
        edit["spans"][0]["end"] -= 1
    elif change == "overlap":
        edit["spans"][1]["start"] -= 1
    elif change == "stale":
        document["sources"][0]["turns"][2]["text"] += " Changed."
    elif change == "reviewer":
        edit["reviewer"] = ""
    elif change == "duplicate":
        edits["turns"].append(copy.deepcopy(edit))
    with pytest.raises(ValueError):
        repair_sources(document, edits)


def test_explicit_finalize_binds_target_but_cannot_fill_semantic_checks():
    doc = collect({"sources": [source()]}, contract())
    decisions = kept(doc)
    decisions[0]["target"]["narration"] = "The operator lifts the rope clear of the drifting wood."
    result = finalize_reviews(doc, decisions, allow_model_reviews=True)
    assert decisions[0]["reviewed_target_sha256"] != result[0]["reviewed_target_sha256"]
    assert export_rows(doc, result, allow_model_reviews=True)["train"]
    decisions[0]["checks"]["player_agency"] = None
    with pytest.raises(ValueError, match="semantic"):
        finalize_reviews(doc, decisions, allow_model_reviews=True)
    doc["schema"] = {}
    with pytest.raises(ValueError, match="schema changed"):
        export_rows(doc, result, allow_model_reviews=True)
