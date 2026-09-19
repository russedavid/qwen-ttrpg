"""Streaming local generation with explicit adapter scales and measured latency."""
import json
import time
import httpx


def generate(
    url,
    model,
    messages,
    adapter,
    adapter_count,
    max_tokens,
    task=None,
    sampling_profile="greedy",
    seed=42,
    schema=None,
):
    from .sampling import sampling

    settings = sampling(task, profile=sampling_profile)
    payload = {
        "model": model,
        "messages": messages,
        **settings,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "lora": [
            {"id": i, "scale": 1.0 if i == adapter else 0.0}
            for i in range(adapter_count)
        ],
    }
    if schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
    started = time.monotonic()
    first = None
    text = ""
    usage = {}
    finish = None
    timings = {}
    with httpx.Client(timeout=600, trust_env=False) as client:
        with client.stream(
            "POST", url.rstrip("/") + "/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("timings"):
                    timings = chunk["timings"]
                for choice in chunk.get("choices", []):
                    content = choice.get("delta", {}).get("content") or ""
                    if content and first is None:
                        first = time.monotonic() - started
                    text += content
                    finish = choice.get("finish_reason") or finish
    elapsed = time.monotonic() - started
    return {
        "text": text,
        "seconds": round(elapsed, 4),
        "time_to_first_text_seconds": round(first, 4) if first else None,
        "usage": usage,
        "finish_reason": finish,
        "server_timings": timings,
        "complete": finish == "stop",
        "settings": {
            **settings,
            "seed": seed,
            "profile": sampling_profile,
            "max_tokens": max_tokens,
            "thinking": False,
            "cache_prompt": False,
            "adapter": adapter,
        },
    }
