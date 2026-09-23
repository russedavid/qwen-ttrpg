import copy

import pytest

from qwen_ttrpg.story_spans import candidate


def fixture():
    source = {"group": "ferry", "split": "train", "source_identity": "recording",
              "turns": [{"ordinal": 1, "text": "The ferry waits. When do we leave? At sunset, says the operator."},
                        {"ordinal": 2, "text": "We agree and board."}]}
    proposal = {"background": [{"turn": 1, "quote": "The ferry waits.", "role": "facilitator"}],
                "participant": [{"turn": 1, "quote": "When do we leave?", "role": "player"}],
                "facilitator": [{"turn": 1, "quote": "At sunset, says the operator.", "role": "facilitator"}],
                "rewritten": 'The operator says, "At sunset."', "skill": "npc_dialogue",
                "reason": "Complete reply.", "creative_scope": "NPC speech."}
    contract = {"system": "Suggest a reply.", "schema": {"type": "object"}, "empty_response": {"narration": ""}}
    return source, proposal, contract


def test_same_turn_voice_boundaries_preserve_source_and_chronology():
    s,p,c = fixture()
    original = copy.deepcopy(s)
    row = candidate(s,p,c)
    assert s == original
    assert row["provenance"]["target_turn"] == 3
    assert row["provenance"]["original_target_spans"][0]["turn"] == 1
    assert "At sunset" not in row["prompt"][-1]["content"]
    assert row["review"]["verdict"] == "pending"


@pytest.mark.parametrize("kind", ["future", "target", "overlap", "conflict", "fabricated", "ambiguous"])
def test_bad_source_support_is_not_exportable(kind):
    s,p,c = fixture()
    if kind == "future":
        p["background"].append({"turn": 2, "quote": "We agree and board.", "role": "player"})
    elif kind == "target":
        p["background"].append(p["facilitator"][0])
    elif kind == "overlap":
        p["background"].append({"turn": 1, "quote": "ferry waits. When", "role": "facilitator"})
    elif kind == "conflict":
        p["background"].append({**p["participant"][0], "role": "facilitator"})
    elif kind == "fabricated":
        p["facilitator"][0]["quote"] = "We leave now."
    elif kind == "ambiguous":
        s["turns"][0]["text"] += " The ferry waits."
    with pytest.raises(ValueError):
        candidate(s,p,c)
