import os
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
import torch


# -------------------- tokenizer helper --------------------
def find_phrase_token_indices(tokenizer, prompt: str | List[str], phrase: str | List[str],
                              add_special_tokens_prompt=True,
                              add_special_tokens_phrase=False) -> List[int]:
    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(phrase, str):
        phrase = [phrase]
    ret_list = []
    for pr, ph in zip(prompt, phrase):
        ret_list.append(
            _find_phrase_token_indices_single(
                tokenizer, pr, ph, add_special_tokens_prompt, add_special_tokens_phrase
            )
        )
    return ret_list

def _find_phrase_token_indices_single(tokenizer, prompt: str, phrase: str,
                              add_special_tokens_prompt=True,
                              add_special_tokens_phrase=False) -> List[int]:
    if not phrase:
        return []
    enc = tokenizer(prompt, padding=False, truncation=False,
                    add_special_tokens=add_special_tokens_prompt,
                    return_attention_mask=False, return_tensors=None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    def _clean(ids): return [i for i in ids if i not in {pad_id, eos_id, None}]
    ids = _clean(enc["input_ids"])
    phrase_ids = _clean(tokenizer(phrase, add_special_tokens=add_special_tokens_phrase)["input_ids"])
    if len(phrase_ids) == 0 or len(ids) < len(phrase_ids):
        return []
    for i in range(0, len(ids) - len(phrase_ids) + 1):
        if ids[i:i+len(phrase_ids)] == phrase_ids:
            return list(range(i, i+len(phrase_ids)))
    return []
