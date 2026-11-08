"""LoRA fine-tuning for single-turn instruction datasets.

Example: python train.py --config configs/example.json --dry-run
"""
import gc
import json

from emtransfer.config import cli, path_for, positive, write_json

MARKERS = ["<|im_start|>assistant", "<|assistant|>", "<start_of_turn>model",
           "<|start_header_id|>assistant<|end_header_id|>"]


def tokenize_entry(entry, tokenizer, max_length, assistant_marker=None):
    """Mask prompt and padding tokens for the project's single-turn datasets."""
    messages = entry.get("messages", [])
    roles = [m.get("role") for m in messages]
    if roles not in [["user", "assistant"], ["system", "user", "assistant"]]:
        raise ValueError("Expected one user/assistant turn, optionally preceded by system")
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    marker = assistant_marker or next((m for m in MARKERS if m in text), None)
    if not marker or marker not in text:
        raise ValueError("Assistant marker not found; set training.assistant_marker")
    start = text.rfind(marker) + len(marker)
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=max_length,
                       return_offsets_mapping=True, add_special_tokens=False)
    offsets = tokens.pop("offset_mapping")
    tokens["labels"] = [token if mask and end > start else -100
                        for token, mask, (_, end) in zip(tokens["input_ids"], tokens["attention_mask"], offsets)]
    if all(label == -100 for label in tokens["labels"]):
        raise ValueError("No assistant tokens survived truncation; increase training.max_length")
    return tokens


def collate_labels(batch):
    """Preserve precomputed -100 labels; a language-modeling collator overwrites them."""
    import torch
    return {key: torch.tensor([row[key] for row in batch], dtype=torch.long)
            for key in ["input_ids", "attention_mask", "labels"]}


def target_modules(model, configured):
    if configured:
        return configured
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if {"q_proj", "v_proj"} <= names:
        return ["q_proj", "v_proj"]
    if "qkv_proj" in names:
        return ["qkv_proj"]
    raise ValueError("Cannot infer LoRA projections; set training.target_modules")


def training_options(c):
    t = {"learning_rate": 2e-4, "rank": 8, "alpha": 16, "dropout": 0.05, "batch_size": 4,
         "gradient_accumulation_steps": 4, "epochs": 3, "warmup_steps": 100, "max_length": 512,
         "dtype": "bfloat16", "load_in_4bit": False, "gradient_checkpointing": True,
         "seed": 42, "resume": True, **c["training"]}
    for key in ["learning_rate", "rank", "alpha", "batch_size", "gradient_accumulation_steps", "epochs", "max_length"]:
        positive(t[key], key)
    positive(t["warmup_steps"], "warmup_steps", allow_zero=True)
    if t["dtype"] not in {"float32", "float16", "bfloat16"} or not 0 <= t["dropout"] < 1:
        raise ValueError("Invalid training dtype or dropout")
    if t.get("target_modules") is not None and (not isinstance(t["target_modules"], list)
            or not t["target_modules"] or not all(isinstance(x, str) and x for x in t["target_modules"])):
        raise ValueError("training.target_modules must be a nonempty list of names")
    return t


def train_domain(c, domain, data_path, output, force=False):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    t = training_options(c)
    set_seed(t["seed"])
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    examples = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    if not examples:
        raise ValueError(f"Empty training dataset: {data_path}")
    if t["load_in_4bit"] and not torch.cuda.is_available():
        raise ValueError("4-bit training requires CUDA and bitsandbytes")
    tok = AutoTokenizer.from_pretrained(c["model"], use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dataset = [tokenize_entry(row, tok, t["max_length"], t.get("assistant_marker")) for row in examples]
    dtype = getattr(torch, t["dtype"])
    kwargs = {"torch_dtype": dtype}
    if t["load_in_4bit"]:
        kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True))
    model = AutoModelForCausalLM.from_pretrained(c["model"], **kwargs)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    targets = target_modules(model, t.get("target_modules"))
    if t["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=t["rank"], lora_alpha=t["alpha"], lora_dropout=t["dropout"],
                                           target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    args = TrainingArguments(output_dir=str(output), learning_rate=t["learning_rate"], num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"], gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_steps=t["warmup_steps"], seed=t["seed"], data_seed=t["seed"], save_strategy="epoch",
        remove_unused_columns=False, report_to="none", bf16=t["dtype"] == "bfloat16", fp16=t["dtype"] == "float16",
        gradient_checkpointing=t["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False})
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = get_last_checkpoint(str(output)) if t["resume"] and not force else None
    trainer = Trainer(model=model, args=args, train_dataset=dataset, processing_class=tok, data_collator=collate_labels)
    write_json(output / "run_config.json", {"workflow": "lora-training", "config": c, "domain": domain,
                                           "effective_training": {**t, "target_modules": targets}})
    trainer.train(resume_from_checkpoint=checkpoint)
    model.save_pretrained(output)
    tok.save_pretrained(output)
    write_json(output / "training_complete.json", {"domain": domain, "examples": len(dataset)})
    del trainer, model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(c, dry_run=False, force=False):
    training_options(c)
    for domain, filename in c["domains"].items():
        data = path_for(c, "data_dir") / filename
        out = path_for(c, "adapter_dir") / domain
        if not force and (out / "training_complete.json").exists():
            print(f"SKIP completed training: {domain}")
            continue
        print(f"TRAIN {c['model']} / {domain}: {data} -> {out}")
        if not dry_run:
            train_domain(c, domain, data, out, force=force)


if __name__ == "__main__":
    cli(__doc__, run)
