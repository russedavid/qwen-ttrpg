"""Run with the optional training environment; no GPU or model download needed."""

import json

import pytest

torch = pytest.importorskip("torch")

from qwen_ttrpg.rl_rollout import run_episodes


class Tokenizer:
    eos_token_id = 300
    pad_token_id = 301

    def apply_chat_template(self, messages, **kwargs):
        ids = []
        for message in messages:
            ids += self.encode(message["role"] + ":" + message["content"]) + [300]
        return ids + self.encode("assistant:")

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids if i < 300)


class Episode:
    def __init__(self, case):
        self.trace, self.done, self.final = [], False, None

    def step(self, raw):
        decision = json.loads(raw)
        self.trace.append(decision)
        if decision["action"] == "answer":
            self.done, self.final = True, decision
        return {"source": "external_evidence_only"}

    def assessment(self):
        return {"success": self.done, "reward": float(self.done)}


class Model:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.batch_sizes = []

    def get_input_embeddings(self):
        return torch.nn.Embedding(1, 1)

    def generate(self, input_ids, attention_mask, **kwargs):
        self.batch_sizes.append(len(input_ids))
        generated = []
        for ids, mask in zip(input_ids.tolist(), attention_mask.tolist()):
            assert all(i == self.tokenizer.pad_token_id for i, m in zip(ids, mask) if m == 0)
            text = self.tokenizer.decode(ids)
            goal = int(text.split("goal=")[1][0])
            action = "answer" if text.count("Evidence result:") >= goal else "read"
            generated.append(self.tokenizer.encode(json.dumps({"action": action})) + [300])
        width = max(len(ids) for ids in generated)
        return torch.tensor([before + after + [301] * (width - len(after))
                             for before, after in zip(input_ids.tolist(), generated)])


def test_batched_episodes_keep_identity_after_early_finish_and_mask_external_text():
    tokenizer = Tokenizer()
    cases = [{"id": str(goal), "family": "mask-test", "split": "synthetic",
              "prompt": [{"role": "user", "content": f"goal={goal} " + "long " * goal}]}
             for goal in [0, 2, 1]]
    model = Model(tokenizer)
    rows = run_episodes(model, tokenizer, cases, Episode, max_new_tokens=80)
    assert model.batch_sizes == [3, 2, 1]
    assert [len(row["trace"]) for row in rows] == [1, 3, 2]
    for row in rows:
        assert row["assessment"]["success"]
        assert len(row["completion_ids"]) == len(row["env_mask"])
        assert 301 not in row["completion_ids"]
        model_text = tokenizer.decode([i for i, mask in zip(row["completion_ids"], row["env_mask"]) if mask])
        external_text = tokenizer.decode([i for i, mask in zip(row["completion_ids"], row["env_mask"]) if not mask])
        assert '"action": "answer"' in model_text
        assert "external_evidence_only" not in model_text
        if len(row["trace"]) > 1:
            assert "external_evidence_only" in external_text
