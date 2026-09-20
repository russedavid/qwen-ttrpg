import json
from copy import deepcopy

import pytest

from qwen_ttrpg.player_data import collect, datasets
from qwen_ttrpg.util import digest


def specification(tmp_path):
    config = {
        "profiles": {
            "curious": "Ask a focused question.",
            "decisive": "Choose a bounded action.",
        },
        "conversations": [],
    }
    for split in ["train", "validation", "test"]:
        path = tmp_path / (split + ".json")
        path.write_text(
            json.dumps(
                {
                    "turns": [
                        {
                            "speaker": "guide",
                            "text": "The " + split + " gate is closed.",
                        },
                        {
                            "speaker": "p",
                            "text": "I inspect the " + split + " gate carefully.",
                        },
                        {
                            "speaker": "p",
                            "text": "Can I see whether its latch is rusty?",
                        },
                    ]
                }
            )
        )
        config["conversations"].append(
            {
                "path": str(path),
                "format": "turns-json",
                "group": split,
                "source_identity": split,
                "split": split,
                "speakers": {"guide": "facilitator", "p": "player"},
            }
        )
    return config


def reviewed(document):
    for case in document["candidates"]:
        case["review"] = {
            "verdict": "keep",
            "profile": "curious",
            "response_complete": True,
            "origin": "model",
            "reviewer": "scripted test",
            "reason": "Original fixture response.",
            "target_sha256": digest(case["target"]),
        }
    return document


def test_complete_player_response_keeps_the_other_speaker_stimulus(tmp_path):
    doc = collect(specification(tmp_path))
    assert len(doc["candidates"]) == 3
    case = doc["candidates"][0]
    assert case["provenance"]["target_turns"] == [2, 3]
    assert len(case["context"]) == 1 and case["context"][0]["speaker"] == "guide"
    assert case["provenance"]["protected_context_turns"] == [1]
    with pytest.raises(ValueError, match="Enable model reviews"):
        datasets(reviewed(doc), "curious")
    rows = datasets(doc, "curious", allow_model_reviews=True)
    assert all(len(v) == 1 for v in rows.values())
    target = json.loads(rows["train"][0]["completion"][0]["content"])
    assert target == {"recipient": "table", "utterance": case["target"]}
    assert all(t < 2 for t in rows["train"][0]["provenance"]["context_turns"])


def test_a_source_group_cannot_be_split_across_training_and_evaluation(tmp_path):
    config = specification(tmp_path)
    config["conversations"][1]["group"] = "train"
    with pytest.raises(ValueError, match="crosses"):
        collect(config)


def test_editing_context_personality_or_target_requires_a_new_review(tmp_path):
    doc = reviewed(collect(specification(tmp_path)))
    for field in ["target", "context", "personality"]:
        changed = deepcopy(doc)
        if field == "personality":
            changed["profiles"]["curious"] = "An unrelated behavior."
        elif field == "target":
            changed["candidates"][0]["target"] = "A rewritten target."
        else:
            changed["candidates"][0]["context"][0]["text"] = "Different evidence."
        with pytest.raises(ValueError, match="frozen source"):
            datasets(changed, "curious", allow_model_reviews=True)
    doc["candidates"][0]["review"]["target_sha256"] = "outdated"
    with pytest.raises(ValueError, match="unchanged complete response"):
        datasets(doc, "curious", allow_model_reviews=True)
