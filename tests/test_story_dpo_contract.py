"""Exercise the actual optional TRL reference contract without a GPU or download."""

import pytest


def test_pretrained_reference_survives_a_real_dpo_update(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("datasets")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset
    from qwen_ttrpg.story_preferences import token_rows
    from qwen_ttrpg.util import digest, packed

    vocabulary = {s: i for i, s in enumerate([
        "[PAD]", "[UNK]", "[EOS]", "system", "user", "assistant", ":",
        "Answer", "What", "next", "?", "The", "door", "opens", ".",
        "Everyone", "follows", "without", "asking",
    ])}
    raw = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]",
                                        unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, pad_token_id=0, eos_token_id=2,
    )), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    # A nonzero starting adapter distinguishes SFT reference from disabled LoRA.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.02)
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    rows = [{"id": "original-fixture", "prompt": [
        {"role": "system", "content": "Answer"}, {"role": "user", "content": "What next?"}],
        "chosen": "The door opens.", "rejected": "Everyone follows without asking."}]
    records, audits = token_rows(rows, tokenizer, 128)
    trainer = DPOTrainer(model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_list(records), eval_dataset=Dataset.from_list(records),
        args=DPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
            max_steps=1, learning_rate=1e-4, per_device_train_batch_size=1,
            per_device_eval_batch_size=1, precompute_ref_log_probs=True,
            precompute_ref_batch_size=1, max_length=128, report_to=[],
            save_strategy="no", eval_strategy="no", remove_unused_columns=False))
    assert "ref" in model.peft_config
    for row in trainer.train_dataset:
        assert digest(packed(row["prompt_ids"])) == audits[0]["prompt_sha256"]
        for side in ("chosen", "rejected"):
            assert digest(packed(row[side + "_ids"])) == audits[0]["completion_sha256"][side]
    for name, parameter in model.named_parameters():
        if ".ref." in name:
            assert not parameter.requires_grad
            assert torch.equal(parameter, initial[name.replace(".ref.", ".default.")])
    result = trainer.train()
    assert result.training_loss == pytest.approx(0.693147, abs=1e-4)
    state = dict(model.named_parameters())
    assert any(not torch.equal(state[n], p) for n, p in initial.items())
    for name, parameter in state.items():
        if ".ref." in name:
            assert torch.equal(parameter, initial[name.replace(".ref.", ".default.")])
