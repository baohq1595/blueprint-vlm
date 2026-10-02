"""Batched inference. Works with or without a LoRA adapter, so the same script
produces the zero-shot baseline and the fine-tuned numbers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from .data import load_records
from .tasks import to_messages


def _dtype() -> torch.dtype:
    ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if ok else torch.float16


def load(cfg: dict, adapter: str | None, device: int = 0, quantize: bool | None = None):
    """Load for generation.

    Two deliberate differences from training:

    * `device_map={"": device}` pins the whole model to ONE card. "auto" shards
      it across both, which is naive model parallelism -- one card computes while
      the other waits, plus a transfer between them every forward pass. Slower
      than a single GPU, and it is why the second T4 sits at 0% util holding
      weights.
    * 4-bit is OFF by default here. bitsandbytes dequantises on every forward
      pass and Turing has no hardware support for it, so on a T4 it costs far
      more time than the memory is worth: 3B in fp16 is ~6.2GB against 15GB free.
      Pass --quantize only if the card is genuinely too small.
    """
    dtype = _dtype()
    processor = AutoProcessor.from_pretrained(
        cfg["model_id"], min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"])

    # Decoder-only generation requires LEFT padding. Qwen defaults to right,
    # which pushes pad tokens between the prompt and the continuation and
    # corrupts every sequence in the batch that is shorter than the longest.
    processor.tokenizer.padding_side = "left"

    use_4bit = cfg["load_in_4bit"] if quantize is None else quantize
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=dtype,
                                bnb_4bit_use_double_quant=True)
             if use_4bit else None)
    print(f"loading on cuda:{device}  dtype={dtype}  4bit={bool(use_4bit)}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], quantization_config=quant,
        torch_dtype=dtype, device_map={"": device},
        attn_implementation="sdpa")
    if adapter:
        # A local path that does not exist is treated by PEFT as a Hub repo id,
        # which fails 50 lines later as a 401 from huggingface.co. Say what is
        # actually wrong: training did not produce an adapter here.
        if not Path(adapter, "adapter_config.json").is_file():
            raise SystemExit(
                f"No adapter at '{adapter}' (adapter_config.json missing).\n"
                "  Training did not finish, or out_dir in the config does not\n"
                "  match --adapter. Run without --adapter for the zero-shot baseline."
            )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), processor


def generate(model, processor, records, image_root, batch_size=4, max_new_tokens=128):
    from qwen_vl_utils import process_vision_info

    out = []
    t0 = time.time()
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        msgs = [to_messages(r, image_root, with_answer=False) for r in chunk]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        images, videos = process_vision_info(msgs)
        enc = processor(text=texts, images=images, videos=videos,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=processor.tokenizer.pad_token_id)
        trimmed = [o[len(i_):] for i_, o in zip(enc.input_ids, ids)]
        out += processor.batch_decode(trimmed, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)

        done = min(i + batch_size, len(records))
        rate = done / max(time.time() - t0, 1e-6)
        eta = (len(records) - done) / max(rate, 1e-6)
        # flush, or Kaggle shows nothing for an hour and looks hung
        print(f"  {done}/{len(records)}  {rate:.2f} rec/s  eta {eta/60:.0f} min",
              flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, help="e.g. data/synth/test")
    ap.add_argument("--adapter", default=None, help="omit for the zero-shot baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=int, default=0, help="which GPU to pin to")
    ap.add_argument("--quantize", action="store_true",
                    help="force 4-bit; slower on T4, only for small cards")
    ap.add_argument("--limit", type=int, default=None,
                    help="score the first N records only")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    # Contiguous sharding, so concatenating shards 0..n-1 restores the original
    # record order exactly. Run one process per GPU to use both cards.
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    records = load_records(args.split)
    if args.limit:
        records = records[:args.limit]
    if args.num_shards > 1:
        n = len(records)
        per = (n + args.num_shards - 1) // args.num_shards
        lo, hi = args.shard_id * per, min((args.shard_id + 1) * per, n)
        records = records[lo:hi]
        print(f"shard {args.shard_id}/{args.num_shards}: records [{lo}:{hi}]")
    print(f"{len(records)} records")

    model, processor = load(cfg, args.adapter, device=args.device,
                            quantize=True if args.quantize else False)
    preds = generate(model, processor, records, cfg["image_root"],
                     args.batch_size, args.max_new_tokens)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Store the identity of every record scored, not just the count. Alignment
    # then survives --limit, sharding, and any later change to the split.
    keys = [f"{r['plan_id']}::{r['task']}" for r in records]
    Path(args.out).write_text(json.dumps(
        {"split": args.split, "adapter": args.adapter, "limit": args.limit,
         "num_shards": args.num_shards, "shard_id": args.shard_id,
         "keys": keys, "predictions": preds}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
