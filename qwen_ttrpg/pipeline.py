"""Complete local, bring-your-own-data training workflow for three task adapters."""

import argparse
import json
from pathlib import Path
import subprocess
from uuid import uuid4

from . import data, experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("init-demo", help="Create original synthetic inputs and a one-step-per-task configuration outside Git")
    demo.add_argument("directory")
    demo.add_argument("--model", required=True)
    demo.add_argument("--runtime")
    demo.add_argument("--base-gguf")
    demo.add_argument("--python")
    for name in ["candidates", "review", "annotate", "prepare", "plan", "run", "serve", "status"]:
        command = sub.add_parser(name)
        command.add_argument("config", help="Local experiment YAML")
        if name == "review":
            select = command.add_mutually_exclusive_group(required=True)
            select.add_argument("--id", action="append")
            select.add_argument("--all", action="store_true")
            command.add_argument("--task", choices=data.TASKS)
            command.add_argument("--decision", choices=["keep", "reject"], required=True)
            command.add_argument("--origin", choices=["human", "model", "synthetic"], required=True)
            command.add_argument("--reviewer", required=True)
            command.add_argument("--reason", required=True)
            command.add_argument("--response-complete", action="store_true")
        elif name == "annotate":
            command.add_argument("--task", choices=["classifier", "rules"], required=True)
            command.add_argument("--routing", required=True)
            command.add_argument("--url", default="http://127.0.0.1:8091/v1")
            command.add_argument("--model-alias", default="qwen3.8-27b-q4")
            command.add_argument("--limit", type=int)
        elif name == "run":
            command.add_argument("--through", choices=["train", "reload", "convert", "evaluate"], default="evaluate")
    args = parser.parse_args()
    if args.command == "init-demo":
        from .demo import create
        print(create(args.directory, args.model, runtime=args.runtime, base_gguf=args.base_gguf, python=args.python))
        return
    config = data.load_config(args.config)
    if args.command == "candidates":
        with experiment.workspace_lock(config):
            print(data.candidates(config))
    elif args.command == "review":
        with experiment.workspace_lock(config):
            count = data.review(config, args.id, verdict=args.decision, origin=args.origin, reviewer=args.reviewer,
                                reason=args.reason, task=args.task, response_complete=args.response_complete)
            print(json.dumps({"reviewed": count}))
    elif args.command == "annotate":
        from .annotation import annotate
        with experiment.workspace_lock(config):
            print(json.dumps(annotate(config, task=args.task, routing=json.loads(Path(args.routing).read_text()),
                                     url=args.url, model=args.model_alias, limit=args.limit)))
    elif args.command == "prepare":
        from transformers import AutoTokenizer
        with experiment.workspace_lock(config):
            tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
            print(json.dumps(data.prepare_all(config, tokenizer), indent=2))
    elif args.command == "plan":
        print(json.dumps(experiment.plan(config), indent=2))
    elif args.command == "run":
        result = experiment.run(config, through=args.through)
        print(json.dumps({key: value["status"] for key, value in result["stages"].items()}, indent=2))
    elif args.command == "serve":
        with experiment.workspace_lock(config):
            output = Path(config["workspace"]) / "serving" / uuid4().hex[:12]
            # Foreground supervisor; Ctrl+C stops this server and leaves artifacts.
            try:
                subprocess.run(experiment.serving_command(config, output), check=True)
            except KeyboardInterrupt:
                pass
    elif args.command == "status":
        path = Path(config["workspace"]) / "pipeline.json"
        state = json.loads(path.read_text()) if path.exists() else {"stages": {}}
        print(json.dumps({"workspace": config["workspace"], "stages": state["stages"]}, indent=2))


if __name__ == "__main__":
    main()
