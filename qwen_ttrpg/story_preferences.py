"""A reviewed, completion-only DPO pilot continuing a verified SFT adapter.

One process uses model parallelism across two GPUs. This is separate from the
FSDP supervised recipe; measure its memory before scheduling a full run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import math
from pathlib import Path
import time

from .data import private_output
from .util import digest, file_digest, packed, now


def validate_pairs(document):
    splits = {"train": [], "validation": []}
    ownership, seen = {}, set()
    for row in document["pairs"]:
        if row["id"] in seen:
            raise ValueError("Duplicate preference pair.")
        seen.add(row["id"])
        split = row["provenance"]["split"]
        if split not in splits:
            raise ValueError("Only training and validation pairs enter the trainer.")
        for k in ("group", "source_identity"):
            value = row["provenance"].get(k)
            if not value:
                raise ValueError("Preference pairs need source and story provenance.")
            key = (k, value)
            if key in ownership and ownership[key] != split:
                raise ValueError("Preference source or story crosses splits.")
            ownership[key] = split
        if (not row["prompt"] or row["chosen"] == row["rejected"] or
            any(not isinstance(row[k], str) or not row[k].strip() for k in ("chosen", "rejected"))):
            raise ValueError("Use one shared prompt and two different nonempty completions.")
        review = row["review"]
        content_hash = digest(packed({k:row[k] for k in ("prompt", "chosen", "rejected", "provenance")}))
        if (review.get("content_sha256") != content_hash or review.get("verdict") != "prefer_chosen"
            or review.get("origin") not in {"human", "model"} or not review.get("reviewer")
            or not review.get("reason") or review.get("chosen_valid") is not True):
            raise ValueError("Require a source-bound preference review and a valid chosen response.")
        splits[split].append(row)
    if not all(splits.values()):
        raise ValueError("Independent nonempty training and validation pairs are required.")
    return splits


def token_rows(rows, tokenizer, max_length):
    """Render and audit the actual serving template before TRL processes strings."""
    encoded, audit = [], []
    for row in rows:
        prefix = tokenizer.apply_chat_template(row["prompt"], tokenize=True, return_dict=False,
                                               add_generation_prompt=True, enable_thinking=False)
        lengths, hashes = {}, {}
        for side in ("chosen", "rejected"):
            full = tokenizer.apply_chat_template(row["prompt"] + [{"role":"assistant", "content":row[side]}],
                                                 tokenize=True, return_dict=False, enable_thinking=False)
            if full[:len(prefix)] != prefix or not len(prefix) < len(full) <= max_length:
                raise ValueError("Preference completion does not fit with an exact protected prompt prefix.")
            if tokenizer.eos_token_id not in full[len(prefix):]:
                raise ValueError("Preference completion is missing its end token.")
            lengths[side] = len(full) - len(prefix)
            hashes[side] = digest(packed(full[len(prefix):]))
        # Conversational rows avoid TRL appending another EOS after the native
        # end-of-turn newline. Per-row template kwargs match serving exactly.
        encoded.append({"prompt":row["prompt"],
                        "chosen":[{"role":"assistant", "content":row["chosen"]}],
                        "rejected":[{"role":"assistant", "content":row["rejected"]}],
                        "chat_template_kwargs":{"enable_thinking":False}})
        audit.append({"id":row["id"], "prompt_tokens":len(prefix), "completion_tokens":lengths,
                      "prompt_sha256":digest(packed(prefix)), "completion_sha256":hashes})
    return encoded, audit


def train(args):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, TrainerCallback, set_seed
    from trl import DPOConfig, DPOTrainer
    from .adapters import strict_load, export_portable

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise ValueError("This pilot requires two visible GPUs.")
    if any(torch.cuda.mem_get_info(i)[0] < 20 * 1024**3 for i in range(2)):
        raise ValueError("Both GPUs need 20 GiB free; pause owned serving services first.")
    output = private_output(args.output)
    if output.exists():
        raise ValueError("Use a new output directory for each preference run.")
    document = json.loads(Path(args.data).read_text())
    pairs = validate_pairs(document)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    datasets, audits = {}, {}
    for split, rows in pairs.items():
        records, audits[split] = token_rows(rows, tokenizer, args.max_length)
        datasets[split] = Dataset.from_list(records)
    output.mkdir(parents=True, mode=0o700)
    record = {"status":"starting", "created":now(), "args":vars(args),
              "dataset_sha256":file_digest(args.data), "adapter_sha256":file_digest(Path(args.adapter)/'adapter_model.safetensors'),
              "packages":{name:version(name) for name in ('torch','transformers','trl','peft','accelerate','bitsandbytes')},
              "template_sha256":digest(tokenizer.chat_template), "mask_audits":audits,
              "review_origins":dict(Counter(row['review']['origin'] for rows in pairs.values() for row in rows)),
              "reference":"Frozen copy of the selected SFT adapter; not the unadapted base."}
    def save():
        (output/'run.json').write_text(json.dumps(record,indent=2))
    save()
    set_seed(args.seed)
    started = time.monotonic()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(args.model,local_files_only=True,
        dtype=torch.bfloat16,quantization_config=quant,device_map="balanced",
        max_memory={0:"20GiB",1:"20GiB"},attn_implementation="sdpa")
    model, reload = strict_load(base,args.adapter,trainable=True)
    model.config.use_cache=False
    model.enable_input_require_grads()
    record['strict_initial_reload']=reload
    initial = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    class Monitor(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            with (output/'metrics.jsonl').open('a') as f:
                f.write(packed({'step':state.global_step,'seconds':time.monotonic()-started,'logs':logs,
                    'peak_allocated':[torch.cuda.max_memory_allocated(i) for i in range(2)]})+'\n')
    cfg=DPOConfig(output_dir=str(output/'checkpoints'),num_train_epochs=args.epochs,
        max_steps=args.steps if args.steps else -1,learning_rate=args.learning_rate,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,
        bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        max_length=args.max_length,precompute_ref_log_probs=True,precompute_ref_batch_size=1,
        beta=args.beta,loss_type='sigmoid',optim='adamw_torch',lr_scheduler_type='constant',
        warmup_steps=0,logging_steps=1,save_strategy='epoch',eval_strategy='epoch',
        save_total_limit=2,save_only_model=True,report_to=[],seed=args.seed,data_seed=args.seed,
        remove_unused_columns=False)
    try:
        trainer=DPOTrainer(model=model,args=cfg,processing_class=tokenizer,
            train_dataset=datasets['train'],eval_dataset=datasets['validation'],callbacks=[Monitor()])
        if 'ref' not in model.peft_config:
            raise ValueError('TRL did not preserve the selected SFT reference adapter.')
        for n,p in model.named_parameters():
            if '.ref.' in n:
                if p.requires_grad or not torch.equal(p.detach().cpu(),initial[n.replace('.ref.','.default.')]):
                    raise ValueError('Reference adapter differs from the selected frozen SFT checkpoint.')
        record['frozen_reference_verified']=True
        for split,ds in [('train',trainer.train_dataset),('validation',trainer.eval_dataset)]:
            for i,row in enumerate(ds):
                expected=audits[split][i]
                if (len(row['prompt_ids'])!=expected['prompt_tokens'] or
                    digest(packed(row['prompt_ids']))!=expected['prompt_sha256'] or
                    any(len(row[k+'_ids'])!=expected['completion_tokens'][k] or
                        digest(packed(row[k+'_ids']))!=expected['completion_sha256'][k]
                        for k in ['chosen','rejected'])):
                    raise ValueError('TRL tokenization differs from the serving-template audit.')
            cached=[{'id':pairs[split][i]['id'],'chosen':float(r['ref_chosen_logps']),
                     'rejected':float(r['ref_rejected_logps'])} for i,r in enumerate(ds)]
            if len(cached)!=len(pairs[split]) or any(not math.isfinite(r[k]) for r in cached for k in ['chosen','rejected']):
                raise ValueError('Reference cache has missing or nonfinite rows.')
            (output/f'{split}.reference.json').write_text(packed(cached))
        record.update(status='training');save()
        result=trainer.train()
        trainer.save_model(str(output/'adapter'))
        state=dict(model.named_parameters())
        updated=sum(not torch.equal(state[n].detach().cpu(),v) for n,v in initial.items())
        if not updated or any(not torch.isfinite(state[n]).all().item() for n in initial):
            raise ValueError('Preference training did not produce finite adapter updates.')
        reference_unchanged=all(torch.equal(p.detach().cpu(),initial[n.replace('.ref.','.default.')])
                                for n,p in state.items() if '.ref.' in n)
        if not reference_unchanged:
            raise ValueError('The frozen SFT reference changed during preference training.')
        record['portable_export']=export_portable(output/'adapter',output/'portable-adapter')
        record.update(status='trained',metrics=result.metrics,updated_tensors=updated,
                      reference_unchanged_after_training=reference_unchanged,
                      seconds=time.monotonic()-started,peak_allocated=[torch.cuda.max_memory_allocated(i) for i in range(2)])
    except BaseException as exc:
        record.update(status='failed',error=str(exc),seconds=time.monotonic()-started)
        raise
    finally:
        save()
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','model','adapter','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-length',type=int,default=4096)
    p.add_argument('--learning-rate',type=float,default=5e-6)
    p.add_argument('--beta',type=float,default=.1)
    p.add_argument('--epochs',type=float,default=1)
    p.add_argument('--steps',type=int)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.max_length<1 or not 0<args.learning_rate<.01 or not (math.isfinite(args.beta) and 0<args.beta) or not 0<args.epochs<=10 or (args.steps is not None and args.steps<1):
        p.error('Use positive, bounded training settings.')
    print(packed(train(args)))


if __name__=='__main__':main()
