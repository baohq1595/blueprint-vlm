"""QLoRA SFT for Qwen2.5-VL on floor-plan takeoff."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen2_5_VLForConditionalGeneration, Trainer,
                          TrainingArguments, set_seed)

from .data import PlanSFT, load_records


def pick_dtype() -> tuple[torch.dtype, bool]:
    """T4 (Turing, sm_75) has no bfloat16. Ampere and later do.

    Passing bf16=True to TrainingArguments on a T4 raises outright, and
    torch_dtype=bfloat16 silently emulates at a large cost, so this has to be
    decided from the hardware rather than hardcoded.
    """
    ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return (torch.bfloat16, True) if ok else (torch.float16, False)


def device_map_for_training() -> dict:
    """Pin the whole model to one GPU.

    device_map="auto" shards a model across both cards, which Trainer reads as
    model-parallel and which conflicts with DDP. Under `accelerate launch
    --multi_gpu` each rank gets its own full copy instead, which is what you
    want for a 3B model in 4-bit -- it fits in 16GB with room to spare.
    """
    rank = os.environ.get("LOCAL_RANK")
    return {"": int(rank)} if rank is not None else {"": 0}


def build_training_args(cfg: dict, bf16_ok: bool, n_train: int) -> TrainingArguments:
    """Construct TrainingArguments against whatever signature is installed.

    `TrainingArguments` is refactored often, and a kwarg that has existed for
    years can vanish in a major release -- discovering that one TypeError at a
    time costs a GPU session each. So: state the intent, ask the class what it
    accepts, translate what has been renamed, and report anything dropped
    instead of failing.
    """
    import transformers
    world = int(os.environ.get("WORLD_SIZE", "1"))
    steps_per_epoch = max(1, math.ceil(n_train / (cfg["batch_size"] * cfg["grad_accum"] * world)))
    total_steps = steps_per_epoch * cfg["epochs"]

    want = {
        "output_dir": cfg["out_dir"],
        "num_train_epochs": cfg["epochs"],
        "per_device_train_batch_size": cfg["batch_size"],
        "gradient_accumulation_steps": cfg["grad_accum"],
        "learning_rate": cfg["lr"],
        "warmup_ratio": cfg["warmup_ratio"],
        "lr_scheduler_type": "cosine",
        "bf16": bf16_ok,
        "fp16": not bf16_ok,
        "optim": cfg.get("optim", "paged_adamw_8bit"),
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": False,
        "logging_steps": 10,
        "eval_strategy": cfg.get("eval_strategy", "epoch"),
        # Checkpoint on a step interval, not per epoch: a session that dies
        # 80% through epoch 1 otherwise leaves nothing to resume from.
        "save_strategy": cfg.get("save_strategy", "steps"),
        "save_steps": cfg.get("save_steps", 100),
        "save_total_limit": 2,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": cfg["seed"],
    }

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)

    # renames seen across versions: old name <- new name we asked for
    alias = {
        "eval_strategy": "evaluation_strategy",
        "evaluation_strategy": "eval_strategy",
    }
    for asked, other in alias.items():
        if asked in want and asked not in accepted and other in accepted:
            want[other] = want.pop(asked)

    # warmup_ratio dropped but warmup_steps kept -> convert rather than lose warmup
    if "warmup_ratio" in want and "warmup_ratio" not in accepted and "warmup_steps" in accepted:
        want["warmup_steps"] = max(1, int(total_steps * want.pop("warmup_ratio")))

    kept = {k: v for k, v in want.items() if k in accepted}
    dropped = sorted(set(want) - set(kept))

    print(f"transformers {transformers.__version__} | ~{total_steps} optimizer steps")
    if dropped:
        print(f"TrainingArguments does not accept, dropped: {dropped}")
    return TrainingArguments(**kept)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", default="data/synth/train")
    ap.add_argument("--dev", default="data/synth/dev")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in out_dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    dtype, bf16_ok = pick_dtype()
    print(f"compute dtype: {dtype} (bf16 supported: {bf16_ok})")

    processor = AutoProcessor.from_pretrained(
        cfg["model_id"], min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"])

    quant = (BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=dtype,
                                bnb_4bit_use_double_quant=True)
             if cfg["load_in_4bit"] else None)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], quantization_config=quant,
        torch_dtype=dtype, device_map=device_map_for_training(),
        attn_implementation="sdpa")   # flash-attn 2 needs Ampere; sdpa runs on T4
    model.config.use_cache = False

    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()   # otherwise checkpointing yields no grads

    targets = list(cfg["lora_targets"])
    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
        # the vision tower stays frozen by default: with ~2k plans there is not
        # enough signal to retune it, and unfreezing it was measurably worse.
        modules_to_save=None)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_recs = load_records(args.train)
    dev_recs = load_records(args.dev)
    if cfg.get("limit_train"):
        train_recs = train_recs[:cfg["limit_train"]]
    if cfg.get("limit_dev"):
        dev_recs = dev_recs[:cfg["limit_dev"]]
    print(f"train {len(train_recs)} records, dev {len(dev_recs)}")

    train_ds = PlanSFT(train_recs, processor, cfg["image_root"], cfg["max_seq_len"])
    dev_ds = PlanSFT(dev_recs, processor, cfg["image_root"], cfg["max_seq_len"])

    targs = build_training_args(cfg, bf16_ok, len(train_recs))

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=dev_ds, data_collator=train_ds.collate)
    ckpts = sorted(Path(cfg["out_dir"]).glob("checkpoint-*")) if args.resume else []
    if ckpts:
        print(f"resuming from {ckpts[-1].name}")
    trainer.train(resume_from_checkpoint=bool(ckpts))
    trainer.save_model(cfg["out_dir"])
    processor.save_pretrained(cfg["out_dir"])
    Path(cfg["out_dir"], "config_used.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved -> {cfg['out_dir']}")


if __name__ == "__main__":
    main()
