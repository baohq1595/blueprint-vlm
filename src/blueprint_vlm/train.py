"""QLoRA SFT for Qwen2.5-VL on floor-plan takeoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen2_5_VLForConditionalGeneration, Trainer,
                          TrainingArguments, set_seed)

from .data import PlanSFT, load_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", default="data/synth/train")
    ap.add_argument("--dev", default="data/synth/dev")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    processor = AutoProcessor.from_pretrained(
        cfg["model_id"], min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"])

    quant = (BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True)
             if cfg["load_in_4bit"] else None)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], quantization_config=quant,
        torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)

    targets = list(cfg["lora_targets"])
    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
        # the vision tower stays frozen by default: with ~2k plans there is not
        # enough signal to retune it, and unfreezing it was measurably worse.
        modules_to_save=None)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = PlanSFT(load_records(args.train), processor, cfg["image_root"], cfg["max_seq_len"])
    dev_ds = PlanSFT(load_records(args.dev), processor, cfg["image_root"], cfg["max_seq_len"])

    targs = TrainingArguments(
        output_dir=cfg["out_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=["none"],
        remove_unused_columns=False,
        seed=cfg["seed"],
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=dev_ds, data_collator=train_ds.collate)
    trainer.train()
    trainer.save_model(cfg["out_dir"])
    processor.save_pretrained(cfg["out_dir"])
    Path(cfg["out_dir"], "config_used.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved -> {cfg['out_dir']}")


if __name__ == "__main__":
    main()
