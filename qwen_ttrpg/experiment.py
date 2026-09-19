"""Sequential training, strict reload, conversion, and paired evaluation on one host."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx

from .data import TASKS, load_review, write_json
from .train import verify_dataset
from .util import digest, file_digest, packed, now


@contextmanager
def workspace_lock(config):
    path = Path(config["workspace"])
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another pipeline process owns this workspace.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def stop_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def execute(argv, log):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    with Path(log).open("w") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = process.wait()
        except BaseException:
            stop_process(process)
            raise
    if code:
        raise RuntimeError(f"Stage exited with code {code}; see {log}")


def recipe_for(config, task):
    custom = config.get("tasks", {}).get(task, {}).get("recipe")
    if custom:
        return (Path(config["_path"]).parent / custom).resolve(strict=True)
    filename = "qwen38-responsive-qlora.yaml" if task == "storyteller" else "qwen38-qlora-fsdp.yaml"
    return Path(__file__).parent / "recipes" / filename


def fingerprint(config):
    root = Path(config["workspace"])
    review = load_review(config)
    snapshots = {task: verify_dataset(root / "datasets" / task) for task in TASKS}
    prepared = json.loads((root / "datasets/prepared.json").read_text())
    if prepared["source_snapshot_sha256"] != review["source_snapshot_sha256"]:
        raise ValueError("Prepared datasets no longer match source inputs.")
    # Changing a review after preparation must never silently train older labels.
    if prepared["preparation_sha256"] != preparation_identity(config, review):
        raise ValueError("Reviews or preparation settings changed; prepare a new workspace.")
    code = {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")}
    base = Path(config["model"])
    tokenizer = {p.name: file_digest(p) for p in base.glob("*.json")}
    cache_path = root / ".file-identities.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def identity(path):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        stamp = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        entry = cache.get(str(path), {})
        if entry.get("stat") != stamp:
            entry = {"stat": stamp, "sha256": file_digest(path)}
            cache[str(path)] = entry
        return entry["sha256"]

    inventory = {p.name: identity(p) for p in base.glob("*.safetensors")}
    serving = {}
    if config.get("base_gguf") and Path(config["base_gguf"]).is_file():
        serving["base_gguf"] = identity(config["base_gguf"])
    if config.get("runtime"):
        for relative in ["build/bin/llama-server", "conversion/qwen.py", "convert_lora_to_gguf.py"]:
            path = Path(config["runtime"]) / relative
            if path.is_file():
                serving[relative] = identity(path)
    write_json(cache_path, cache)
    stable = {k: v for k, v in config.items() if k != "_path"}
    return digest(packed({"config": stable, "snapshots": snapshots, "code": code,
                          "base_configuration": tokenizer, "base_weight_inventory": inventory, "serving_runtime": serving,
                          "recipes": {t: file_digest(recipe_for(config, t)) for t in TASKS}}))


def tokenizer_identity(config):
    base = Path(config["model"])
    return {p.name: file_digest(p) for p in base.glob("*.json")}


def preparation_identity(config, review):
    return digest(packed({"review": review, "tasks": config.get("tasks", {}), "tokenizer": tokenizer_identity(config)}))


def training_command(config, task):
    root = Path(config["workspace"])
    options = config.get("tasks", {}).get(task, {})
    command = [config.get("python", sys.executable), "-m", "qwen_ttrpg.train", str(root / "datasets" / task),
               "--model", config["model"], "--recipe", str(recipe_for(config, task)),
               "--output", str(root / "runs" / task), "--resume"]
    for key, flag in [("max_steps", "--max-steps"), ("epochs", "--epochs")]:
        if options.get(key) is not None:
            command += [flag, str(options[key])]
    return command


def plan(config):
    fingerprint(config)
    return {"workspace": config["workspace"], "order": list(TASKS),
            "training_commands": {task: training_command(config, task) for task in TASKS},
            "stages": ["train", "reload", "convert", "evaluate"],
            "execution": "Sequential; training and evaluation never share the GPUs with this runner's server."}


def verify_outputs(stage):
    for name, expected in stage.get("artifacts", {}).items():
        path = Path(name)
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError("A completed stage artifact is missing or changed: " + name)


def stage(config, journal, key, builder, verifier, executor):
    record = journal["stages"].setdefault(key, {"status": "pending", "attempts": []})
    if record["status"] == "complete":
        verify_outputs(record)
        return record
    # A failed run's files remain; each retry has a new output/log path.
    attempt = len(record["attempts"]) + 1
    folder = Path(config["workspace"]) / "attempts" / key.replace(":", "-") / str(attempt)
    folder.mkdir(parents=True, exist_ok=False)
    argv, expected = builder(folder)
    entry = {"started": now(), "command": argv, "log": str(folder / "stage.log")}
    record["attempts"].append(entry)
    record.update(status="running", outputs=expected)
    write_json(Path(config["workspace"]) / "pipeline.json", journal)
    print(f"{key}: attempt {attempt}", flush=True)
    try:
        executor(argv, entry["log"])
        artifacts = verifier(expected)
        record.update(status="complete", artifacts={str(p): file_digest(p) for p in artifacts})
        entry["finished"] = now()
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        entry["finished"] = now()
        raise
    finally:
        write_json(Path(config["workspace"]) / "pipeline.json", journal)
    return record


def run(config, *, through="evaluate", executor=execute):
    if through not in {"train", "reload", "convert", "evaluate"}:
        raise ValueError("Unknown final stage.")
    with workspace_lock(config):
        root = Path(config["workspace"])
        identity = fingerprint(config)
        journal_path = root / "pipeline.json"
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {"version": 1, "created": now(), "fingerprint": identity, "stages": {}}
        if journal["fingerprint"] != identity:
            raise ValueError("Experiment inputs, source code, model files, or settings changed; choose a new workspace.")
        python = config.get("python", sys.executable)
        for task in TASKS:
            output = root / "runs" / task
            def verify_train(paths):
                metadata = json.loads(Path(paths["run"]).read_text())
                if metadata.get("status") != "trained":
                    raise ValueError("Training did not complete with a verified adapter export.")
                adapter = Path(paths["adapter"])
                if file_digest(adapter / "adapter_model.safetensors") != metadata["portable_export"]["export_sha256"]:
                    raise ValueError("Portable adapter differs from the verified export.")
                return [Path(paths["run"]), adapter / "adapter_model.safetensors", adapter / "adapter_config.json"]
            stage(config, journal, task + ":train",
                  lambda folder: (training_command(config, task), {"run": str(output / "run.json"), "adapter": str(output / "portable-adapter")}),
                  verify_train, executor)
            if through == "train":
                continue
            def reload_command(folder):
                result = folder / "reload.json"
                argv = [python, "-m", "qwen_ttrpg.evaluate_adapter", str(root / "datasets" / task / "validation.sources.jsonl"),
                        "--model", config["model"], "--adapter", str(output / "portable-adapter"), "--output", str(result),
                        "--loss-only", "--cases", str(config.get("evaluation", {}).get("reload_cases", 4)),
                        "--sequence-len", str(config.get("tasks", {}).get(task, {}).get("sequence_len", 4096))]
                return argv, {"report": str(result)}
            def verify_reload(paths):
                report = json.loads(Path(paths["report"]).read_text())
                if report.get("status") != "complete" or not report.get("reload_validation", {}).get("tensors_loaded_and_compared"):
                    raise ValueError("Strict reload/loss evaluation is incomplete.")
                return [Path(paths["report"])]
            stage(config, journal, task + ":reload", reload_command, verify_reload, executor)
            if through == "reload":
                continue
            def convert_command(folder):
                if not config.get("runtime"):
                    raise ValueError("Set runtime to a local compatible llama.cpp checkout before conversion.")
                destination = folder / "adapter.gguf"
                argv = [python, "-m", "qwen_ttrpg.convert_adapter", "--runtime", config["runtime"], "--base", config["model"],
                        "--adapter", str(output / "portable-adapter"), "--output", str(destination)]
                return argv, {"gguf": str(destination), "report": str(destination.with_suffix(".conversion.json"))}
            def verify_conversion(paths):
                report = json.loads(Path(paths["report"]).read_text())
                if not report.get("bytes") or report["gguf_sha256"] != file_digest(paths["gguf"]):
                    raise ValueError("Converted adapter does not match its verification report.")
                if any(v > 1e-10 for v in report["factor_permutation_max_errors"]):
                    raise ValueError("The low-rank conversion identity check failed.")
                return [Path(paths["report"]), Path(paths["gguf"])]
            stage(config, journal, task + ":convert", convert_command, verify_conversion, executor)
        if through in {"convert", "evaluate"}:
            deployment = {"tasks": {task: journal["stages"][task + ":convert"]["outputs"]["gguf"] for task in TASKS},
                          "fingerprint": identity, "base": config.get("base_gguf"), "created": now()}
            write_json(root / "deployment.json", deployment)
        if through == "evaluate":
            evaluate(config, journal)
        return journal


def serving_command(config, output):
    root = Path(config["workspace"])
    journal = json.loads((root / "pipeline.json").read_text())
    if journal["fingerprint"] != fingerprint(config):
        raise ValueError("Serving configuration no longer matches the verified experiment.")
    if not config.get("runtime") or not config.get("base_gguf"):
        raise ValueError("Supply runtime and base_gguf for local inference.")
    options = config.get("serving", {})
    args = [config.get("python", sys.executable), "-m", "qwen_ttrpg.serve_models", "--runtime", config["runtime"],
            "--base", config["base_gguf"], "--output", str(output), "--port", str(options.get("port", 8092)),
            "--layout", options.get("layout", "split"), "--context", str(options.get("context", 16384)),
            "--slots", str(options.get("slots", 1))]
    if options.get("tensor_split"):
        args += ["--tensor-split", str(options["tensor_split"])]
    for task in TASKS:
        for phase in ["train", "reload", "convert"]:
            record = journal["stages"][task + ":" + phase]
            if record["status"] != "complete":
                raise ValueError("All selected adapters must pass training, reload, and conversion first.")
            verify_outputs(record)
        args += ["--adapter", task + "=" + journal["stages"][task + ":convert"]["outputs"]["gguf"]]
    return args


def evaluate(config, journal):
    from .benchmark import run as benchmark
    root = Path(config["workspace"])
    prior = journal["stages"].get("evaluate", {})
    if prior.get("status") == "complete":
        verify_outputs(prior)
        return
    number = len(prior.get("attempts", [])) + 1
    output = root / "evaluations" / str(number)
    output.mkdir(parents=True)
    options = config.get("evaluation", {})
    port = config.get("serving", {}).get("port", 8092)
    url = f"http://127.0.0.1:{port}"
    # Do not accidentally evaluate a different model already using the requested port.
    import socket
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise ValueError("Evaluation port is occupied; select a free local port.")
    record = {"status": "running", "attempts": prior.get("attempts", []) + [{"started": now(), "path": str(output)}]}
    journal["stages"]["evaluate"] = record
    write_json(root / "pipeline.json", journal)
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    with (output / "supervisor.log").open("w") as log:
        process = subprocess.Popen(serving_command(config, output / "server"), stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + options.get("startup_timeout", 240)
            with httpx.Client(timeout=3, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("The evaluation server exited; inspect its logs.")
                    try:
                        if client.get(url + "/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.5)
                else:
                    raise TimeoutError("Local inference server did not become ready.")
            routing = json.loads((output / "server/routing.json").read_text())
            artifacts = []
            generation_failures = {}
            for split in ["validation", "test"]:
                cases = []
                for task in TASKS:
                    for line in (root / "datasets" / task / (split + ".sources.jsonl")).read_text().splitlines():
                        row = json.loads(line)
                        row.update(contract_version=1, target=json.loads(row["completion"][0]["content"]))
                        cases.append(row)
                if not cases:
                    continue
                report = benchmark(cases, routing, url + "/v1", "qwen3.8-27b-q4", output / split,
                                   max_tokens=options.get("max_tokens", 512), seed=options.get("seed", 42))
                artifacts.extend(output.joinpath(split).glob("*"))
                # A truncated answer is an observed model failure. Preserve it
                # and continue the study; it must never count as a control pass.
                generation_failures[split] = sum(not a["complete"] for c in report["cases"] for a in c["answers"].values())
            record.update(status="complete", quality_review="pending", incomplete_generations=generation_failures,
                          artifacts={str(p): file_digest(p) for p in artifacts if p.is_file()})
        except BaseException as exc:
            record.update(status="failed", error=str(exc))
            raise
        finally:
            stop_process(process)
            record["attempts"][-1]["finished"] = now()
            write_json(root / "pipeline.json", journal)
