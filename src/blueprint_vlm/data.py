"""Dataset + collator for Qwen2.5-VL SFT.

Labels mask everything before the assistant turn. The prompt is tokenised twice
-- once alone, once with the answer -- and the prefix length from the first pass
is masked in the second. Image token expansion is identical across both passes,
so the boundary is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tasks import to_messages

IGNORE = -100


class PlanSFT(Dataset):
    def __init__(self, records: list[dict], processor, image_root: str, max_seq_len: int = 4096):
        self.records, self.processor = records, processor
        self.image_root, self.max_seq_len = image_root, max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        return self.records[i]

    def collate(self, batch: list[dict]) -> dict:
        from qwen_vl_utils import process_vision_info

        full, prompt = [], []
        for rec in batch:
            full.append(to_messages(rec, self.image_root, with_answer=True))
            prompt.append(to_messages(rec, self.image_root, with_answer=False))

        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                 for m in full]
        images, videos = process_vision_info(full)
        enc = self.processor(text=texts, images=images, videos=videos,
                             padding=True, truncation=True,
                             max_length=self.max_seq_len, return_tensors="pt")

        p_texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in prompt]
        p_images, p_videos = process_vision_info(prompt)
        p_enc = self.processor(text=p_texts, images=p_images, videos=p_videos,
                               padding=True, truncation=True,
                               max_length=self.max_seq_len, return_tensors="pt")

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = IGNORE
        for i in range(len(batch)):
            n_prompt = int(p_enc["attention_mask"][i].sum())
            labels[i, :n_prompt] = IGNORE
        enc["labels"] = labels
        return enc


# Defined in evaluate.py (which has no torch dependency); re-exported here so
# the training and inference paths keep importing it from the same place.
from .evaluate import load_records  # noqa: E402,F401
