"""Serve one quantized base and several small adapters on the local GPU machine."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import signal

from .util import digest, now, file_digest


def command(
    runtime, base, adapters, port=8091, layout="single", slots=1, context=16384,
    tensor_split=None,
):
    if layout not in {"single", "split"} or slots not in {1, 2, 3, 4}:
        raise ValueError("Choose single/split layout and one to four slots.")
    if not 1024 <= context <= 262144:
        raise ValueError("Choose a bounded per-request context window.")
    if tensor_split is not None:
        if layout != "split":
            raise ValueError("A tensor split requires the split layout.")
        try:
            values = [float(value) for value in tensor_split.split(",")]
        except (ValueError, AttributeError) as exc:
            raise ValueError("Use two positive finite GPU proportions, such as 4,1.") from exc
        if len(values) != 2 or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Use two positive finite GPU proportions, such as 4,1.")
    result = [
        str(Path(runtime) / "build/bin/llama-server"),
        "-m",
        str(base),
        "--alias",
        "qwen3.8-27b-q4",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context * slots),
        "--parallel",
        str(slots),
        "--n-gpu-layers",
        "all",
        "--flash-attn",
        "on",
        "--jinja",
        "--no-context-shift",
        "--chat-template-kwargs",
        json.dumps({"enable_thinking": False}),
    ]
    if layout == "split":
        result += ["--split-mode", "layer", "--tensor-split", tensor_split or "1,1"]
    else:
        result += ["--split-mode", "none", "--main-gpu", "0"]
    if adapters:
        if any(any(c in str(p) for c in [",", ":"]) for p in adapters):
            raise ValueError("Adapter filenames may not contain commas or colons.")
        result += ["--lora-scaled", ",".join(str(p) + ":0.0" for p in adapters)]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", required=True)
    p.add_argument("--base", required=True)
    p.add_argument(
        "--adapter", action="append", default=[], help="TASK=/path/to/adapter.gguf"
    )
    p.add_argument(
        "--candidate", action="append", default=[],
        help="NAME=/path/to/adapter.gguf; load an unselected comparison candidate alongside task adapters.",
    )
    p.add_argument("--layout", choices=["single", "split"], default="split")
    p.add_argument("--tensor-split", help="Two GPU proportions with --layout split, e.g. 4,1; default 1,1.")
    p.add_argument("--slots", type=int, default=1)
    p.add_argument("--context", type=int, default=16384)
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Choose a new serving-run directory.")
    base = Path(args.base).resolve(strict=True)
    inventory = []
    tasks = {}
    for item in args.adapter:
        task, path = item.split("=", 1)
        if task not in {"classifier", "storyteller", "rules"} or task in tasks:
            raise ValueError("Each supported task may select one adapter.")
        path = Path(path).resolve(strict=True)
        tasks[task] = len(inventory)
        inventory.append(
            {
                "id": len(inventory),
                "task": task,
                "path": str(path),
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
        )
    names = set(tasks)
    for item in args.candidate:
        name, path = item.split("=", 1)
        if not name.strip() or name in names:
            raise ValueError("Comparison candidate names must be nonempty and distinct.")
        names.add(name)
        path = Path(path).resolve(strict=True)
        inventory.append({
            "id": len(inventory), "candidate": name, "path": str(path),
            "sha256": file_digest(path), "bytes": path.stat().st_size,
        })
    argv = command(
        args.runtime,
        base,
        [a["path"] for a in inventory],
        args.port,
        args.layout,
        args.slots,
        args.context,
        args.tensor_split,
    )
    output.mkdir(parents=True)
    (output / "routing.json").write_text(
        json.dumps({"adapters": inventory, "tasks": tasks}, indent=2)
    )
    metadata = {
        "created": now(),
        "base_sha256": file_digest(base),
        "adapters": inventory,
        "command": argv,
        "layout": args.layout,
        "slots": args.slots,
        "context_per_slot": args.context,
        "tensor_split": args.tensor_split or ("1,1" if args.layout == "split" else None),
        "status": "starting",
        "note": "Request-specific adapters share one base. Different adapter configurations may be scheduled separately by llama.cpp.",
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0" if args.layout == "single" else "0,1"
    with (output / "server.log").open("w") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True
            proc.terminate()

        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            signal.signal(sig, stop)
        metadata.update(pid=proc.pid, status="running")
        (output / "run.json").write_text(json.dumps(metadata, indent=2))
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            code = proc.wait(timeout=30)
    metadata.update(
        status="stopped" if code == 0 or stopped else "failed",
        returncode=code,
        finished=now(),
    )
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
