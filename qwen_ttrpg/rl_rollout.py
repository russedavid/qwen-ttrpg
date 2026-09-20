"""Prefix-checked multi-turn rollouts with external evidence excluded from loss."""

from copy import deepcopy
import json
import time


def append_context(prefix, generated, rendered):
    """Never retokenize model actions or silently change their training context."""
    prior = prefix + generated
    if rendered[: len(prior)] != prior:
        raise ValueError("Chat template changed earlier rollout tokens.")
    return rendered[len(prior):]


def encode_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def run_episodes(model, tokenizer, cases, episode_class, *, temperature=1.0,
                 max_new_tokens=160, context_limit=3072):
    """Batch independent trajectories while executing each environment separately."""
    import torch

    started = time.monotonic()
    states = []
    for case in cases:
        messages = deepcopy(case["prompt"])
        prompt = encode_prompt(tokenizer, messages)
        states.append({"case": case, "episode": episode_class(case), "messages": messages,
                       "prompt_ids": prompt, "current": prompt[:],
                       "completion_ids": [], "env_mask": [], "truncated": False})
    with torch.no_grad():
        for _ in range(4):
            active = []
            for state in states:
                if state["episode"].done or state["truncated"]:
                    continue
                if context_limit - len(state["current"]) < max_new_tokens:
                    state["truncated"] = True
                else:
                    active.append(state)
            if not active:
                break
            device = model.get_input_embeddings().weight.device
            width = max(len(state["current"]) for state in active)
            ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(state["current"])) + state["current"] for state in active], device=device)
            mask = torch.tensor([[0] * (width - len(state["current"])) + [1] * len(state["current"]) for state in active], device=device)
            options = {"do_sample": temperature > 0, "max_new_tokens": max_new_tokens,
                       "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                       "use_cache": True, "repetition_penalty": 1.0}
            if temperature > 0:
                options.update(temperature=temperature, top_p=1.0, top_k=0)
            output = model.generate(input_ids=ids, attention_mask=mask, **options)
            for index, state in enumerate(active):
                generated = output[index, width:].tolist()
                if tokenizer.eos_token_id in generated:
                    generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
                else:
                    state["truncated"] = True
                state["completion_ids"].extend(generated)
                state["env_mask"].extend([1] * len(generated))
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                observation = state["episode"].step(raw)
                state["messages"].append({"role": "assistant", "content": raw})
                if state["episode"].done or state["truncated"]:
                    continue
                state["messages"].append({"role": "user", "content": "Evidence result: " + json.dumps(observation, ensure_ascii=False, sort_keys=True)})
                rendered = encode_prompt(tokenizer, state["messages"])
                external = append_context(state["current"], generated, rendered)
                state["completion_ids"].extend(external)
                state["env_mask"].extend([0] * len(external))
                state["current"] = rendered
    rows = []
    for state in states:
        episode, case = state["episode"], state["case"]
        assessment = episode.assessment()
        if state["truncated"]:
            assessment["success"] = False
            assessment["reward"] = min(assessment["reward"], 0.0)
        assessment.update(truncated=state["truncated"], seconds=round(time.monotonic() - started, 3),
                          generated_tokens=sum(state["env_mask"]), context_tokens=len(state["prompt_ids"]) + len(state["completion_ids"]))
        rows.append({"id": case["id"], "family": case["family"], "split": case["split"],
                     **{key: state[key] for key in ["prompt_ids", "completion_ids", "env_mask"]},
                     "assessment": assessment, "trace": episode.trace, "final": episode.final})
    return rows


def run_episode(model, tokenizer, case, episode_class, **kwargs):
    return run_episodes(model, tokenizer, [case], episode_class, **kwargs)[0]


class AgentRollouts:
    def __init__(self, cases, episode_class, tokenizer, output, *, max_new_tokens=160, context_limit=3072):
        self.cases = {json.dumps(c["prompt"], sort_keys=True): c for c in cases}
        if len(self.cases) != len(cases):
            raise ValueError("Each training case needs a unique public episode identity.")
        self.episode_class = episode_class
        self.tokenizer = tokenizer
        self.output = output
        self.max_new_tokens = max_new_tokens
        self.context_limit = context_limit
        self.sequence = 0

    def __call__(self, prompts, trainer):
        from trl.models import unwrap_model_for_generation

        rows = []
        with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator) as model:
            was_training = model.training
            model.eval()
            cases = []
            for prompt in prompts:
                key = json.dumps(prompt, sort_keys=True)
                if key not in self.cases:
                    raise ValueError("Rollout prompt is outside the frozen training snapshot.")
                cases.append(self.cases[key])
            rows = run_episodes(model, self.tokenizer, cases, self.episode_class,
                                temperature=trainer.args.temperature, max_new_tokens=self.max_new_tokens,
                                context_limit=self.context_limit)
            for row in rows:
                row["step"] = trainer.state.global_step
                row["sequence"] = self.sequence
                row["rollout_batch_size"] = len(rows)
                self.sequence += 1
                with self.output.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            model.train(was_training)
        # Same in-process policy generates and is scored before its update.
        # None asks TRL to recompute behavior log-probs; no cross-engine estimate.
        return {"prompt_ids": [r["prompt_ids"] for r in rows],
                "completion_ids": [r["completion_ids"] for r in rows], "logprobs": None,
                "env_mask": [r["env_mask"] for r in rows],
                "episode_score": [r["assessment"]["reward"] for r in rows],
                "episode_success": [r["assessment"]["success"] for r in rows]}


def episode_reward(completions, episode_score, **kwargs):
    """Read only environment-computed scores, never model-authored reward text."""
    if len(completions) != len(episode_score):
        raise ValueError("Reward/rollout alignment mismatch.")
    return [float(value) for value in episode_score]
