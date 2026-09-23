"""Batched inference. Works with or without a LoRA adapter, so the same script
produces the zero-shot baseline and the fine-tuned numbers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from .data import load_records
from .tasks import to_messages


def _dtype() -> torch.dtype:
    ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if ok else torch.float16


def load(cfg: dict, adapter: str | None):
    dtype = _dtype()
    processor = AutoProcessor.from_pretrained(
        cfg["model_id"], min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"])
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=dtype,
                                bnb_4bit_use_double_quant=True)
             if cfg["load_in_4bit"] else None)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], quantization_config=quant,
        torch_dtype=dtype, device_map="auto",
        attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), processor


def generate(model, processor, records, image_root, batch_size=4, max_new_tokens=192):
    from qwen_vl_utils import process_vision_info

    out = []
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        msgs = [to_messages(r, image_root, with_answer=False) for r in chunk]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        images, videos = process_vision_info(msgs)
        enc = processor(text=texts, images=images, videos=videos,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [o[len(i_):] for i_, o in zip(enc.input_ids, ids)]
        out += processor.batch_decode(trimmed, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
        print(f"  {min(i + batch_size, len(records))}/{len(records)}", end="\r")
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, help="e.g. data/synth/test")
    ap.add_argument("--adapter", default=None, help="omit for the zero-shot baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    records = load_records(args.split)
    model, processor = load(cfg, args.adapter)
    preds = generate(model, processor, records, cfg["image_root"], args.batch_size)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"split": args.split, "adapter": args.adapter, "predictions": preds}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
