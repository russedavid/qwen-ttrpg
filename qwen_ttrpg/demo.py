"""Generate an original tiny fixture outside Git to exercise the whole pipeline."""

import json
from pathlib import Path

import yaml

from .data import private_output, write_json, load_config, candidates, load_review, review_path, review, render_review


def create(directory, model, *, runtime=None, base_gguf=None, python=None):
    directory = private_output(directory)
    directory.mkdir(parents=True, exist_ok=False)
    conversations, questions, documents, excerpts = [], [], [], []
    # These examples are authored software fixtures. Rephrased shared patterns
    # demonstrate mechanics, never generalization or suitability for actual play.
    for split, place, item, count in [("train", "harbor", "token", 2),
                                     ("validation", "garden", "ribbon", 3),
                                     ("test", "workshop", "bead", 4)]:
        path = directory / (split + "-conversation.json")
        turns = [{"speaker": "participant", "text": f"What can I see in the {place}?"},
                 {"speaker": "narrator", "text": f"A wooden box holds a {item}. The box stands open."},
                 {"speaker": "participant", "text": f"I examine the {item} without picking it up."},
                 {"speaker": "narrator", "text": f"There are {count} small marks along its edge. It remains in the box."}]
        write_json(path, {"turns": turns}, exclusive=True)
        conversations.append({"path": path.name, "format": "turns-json", "group": split + "-scene",
                              "split": split, "speakers": {"participant": "player", "narrator": "facilitator"}})
        rule = f"Opening the {place} gate consumes one {item}. A closed gate stays closed until that cost is paid."
        doc = directory / (split + "-rules.txt")
        doc.write_text(rule + "\n")
        rid = split + "-gate"
        documents.append({"id": split + "-doc", "path": doc.name})
        excerpts.append({"id": rid, "document": split + "-doc", "quote": rule})
        questions.append({"id": split + "-question", "group": split + "-question-family", "split": split,
                          "question": f"What must I pay to open the {place} gate?", "references": [rid],
                          "target": {"answer": f"Opening that gate consumes one {item}.",
                                     "citations": [{"id": rid, "quote": f"Opening the {place} gate consumes one {item}."}],
                                     "calculation": None, "missing_information": []}})
    write_json(directory / "questions.json", questions, exclusive=True)
    config = {"version": 1, "workspace": "run", "model": str(Path(model).expanduser().resolve()),
              "conversations": conversations, "rule_documents": documents, "rule_excerpts": excerpts,
              "rule_questions": "questions.json", "allow_synthetic": True,
              "tasks": {task: {"sequence_len": 2048, "max_steps": 1} for task in ["classifier", "rules", "storyteller"]},
              "serving": {"port": 8092, "layout": "split", "context": 8192, "slots": 1},
              "evaluation": {"reload_cases": 1, "max_tokens": 512},
              "purpose": "Original synthetic integration fixture; shared patterns, not a quality benchmark."}
    for key, value in [("runtime", runtime), ("base_gguf", base_gguf), ("python", python)]:
        if value:
            value = Path(value).expanduser()
            config[key] = str(value.absolute() if key == "python" else value.resolve())
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    loaded = load_config(path)
    candidates(loaded)
    queue = load_review(loaded)
    for case in queue["candidates"]:
        if case["task"] == "classifier":
            turn = case["input"]["target_turns"][-1]
            # This is a source-grounded fact label authored with the fixture.
            case["target"] = {"events": [{"kind": "fact", "entity": "box contents", "attribute": "observed marks",
                "value": turn["text"], "delta": None, "stage": "established", "visibility": "public",
                "evidence": [{"turn": turn["turn"], "quote": turn["text"]}], "resolves": None, "supersedes": None}], "uncertainties": []}
        case["label_origin"] = "original-synthetic-fixture"
    write_json(review_path(loaded), queue)
    review(loaded, None, verdict="keep", origin="synthetic", reviewer="fixture-author",
           reason="Authored together with the original synthetic input; exercises the pipeline, not independent quality.",
           response_complete=True)
    return path
