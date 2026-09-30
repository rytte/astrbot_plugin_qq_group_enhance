"""Direct categorical decision readout from native next-token logits."""

from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .core import LETTERS, digest, direct_messages, softmax, synchronize, validate_row

PROMPT_VERSION = "direct-options-v1"
MULTIMODAL_PROMPT_VERSION = "direct-options-multimodal-v1"


def _slot_ids(tokenizer, count: int) -> list[int]:
    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise ValueError("Answer-slot tokens collide")
    return result


def _forward(model, inputs):
    parameters = inspect.signature(model.forward).parameters
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    return model(**kwargs).logits[:, -1, :]


def _processor_tokenizer(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Multimodal processor does not expose its tokenizer")
    return tokenizer


def _read_image(source: str):
    """Load one declared image without letting the model fetch arbitrary media."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Image inputs require Pillow; install the project dependencies") from error

    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https"}:
        request = urllib.request.Request(source, headers={"User-Agent": "semif-phase1/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    elif parsed.scheme == "data":
        header, encoded = source.split(",", 1)
        if ";base64" not in header.lower():
            raise ValueError("Image data URLs must use base64 encoding")
        raw = base64.b64decode(encoded, validate=True)
    else:
        if parsed.scheme == "file":
            path_text = urllib.parse.unquote(parsed.path)
            if parsed.netloc and parsed.netloc != "localhost":
                path_text = f"//{parsed.netloc}{path_text}"
            elif len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
                path_text = path_text[1:]
            path = Path(urllib.request.url2pathname(path_text))
        else:
            path = Path(source)
        if not path.is_file():
            raise ValueError(f"Image file does not exist: {source}")
        raw = path.read_bytes()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as error:
        raise ValueError(f"Image is not a readable bitmap: {source}") from error
    return image, raw


def _multimodal_inputs(processor, row: dict):
    """Build processor tensors and an audit hash for an image-text decision."""
    validate_row(row)
    if "images" not in row:
        raise ValueError("Multimodal encoding requires an images field")
    messages = direct_messages(row)
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    images, raw_images = zip(*(_read_image(source) for source in row["images"]))
    try:
        encoded = processor(
            text=[rendered],
            images=list(images),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    except TypeError as error:
        raise RuntimeError("The multimodal processor must accept text, images, and add_special_tokens") from error
    if "input_ids" not in encoded or not encoded["input_ids"].numel():
        raise ValueError(f"Row {row['id']}: multimodal processor returned no input tokens")
    if (
        "pixel_values" not in encoded
        or "image_grid_thw" not in encoded
        or encoded["pixel_values"].numel() == 0
        or encoded["image_grid_thw"].numel() == 0
    ):
        raise ValueError(f"Row {row['id']}: multimodal processor returned no image tensors")
    ids = encoded["input_ids"][0].tolist()
    tokenizer = _processor_tokenizer(processor)
    slots = _slot_ids(tokenizer, len(row["options"]))
    for letter, token in zip(LETTERS, slots):
        candidate = processor(
            text=[rendered + letter],
            images=list(images),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        candidate_ids = candidate["input_ids"][0].tolist()
        if candidate_ids[:-1] != ids or candidate_ids[-1] != token:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
    media_hashes = [hashlib.sha256(raw).hexdigest() for raw in raw_images]
    prompt_hash = digest(json.dumps({"prompt": rendered, "images_sha256": media_hashes}, ensure_ascii=False))
    return encoded, ids, slots, prompt_hash


def encode_prompt(tokenizer, row: dict, max_tokens: int) -> tuple[list[int], list[int], str]:
    """Encode one text decision and verify its single-token answer slots."""
    if "images" in row:
        raise ValueError("Image inputs require direct Torch scoring with a multimodal processor")
    text_tokenizer = tokenizer if hasattr(tokenizer, "encode") else _processor_tokenizer(tokenizer)
    prompt = tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = text_tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = _slot_ids(text_tokenizer, len(row["options"]))
    for letter, token in zip(LETTERS, slots):
        if text_tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
    return ids, slots, digest(prompt)


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    import torch

    started = time.perf_counter()
    if "images" in row:
        encoded, ids, slots, prompt_hash = _multimodal_inputs(tokenizer, row)
        if len(ids) > max_tokens:
            raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    else:
        encoded = None
        ids, slots, prompt_hash = encode_prompt(tokenizer, row, max_tokens)
    device = next(model.parameters()).device
    if encoded is None:
        inputs = {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
        }
    else:
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
    synchronize(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        vocabulary = _forward(model, inputs)[0].float()
    synchronize(device)
    selected = vocabulary[slots].cpu().tolist()
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "input_tokens": len(ids),
        "forward_seconds": time.perf_counter() - forward_start,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": MULTIMODAL_PROMPT_VERSION if "images" in row else PROMPT_VERSION,
        "model": metadata,
        "readout": (
            "native multimodal last-position logits restricted to declared answer slots"
            if "images" in row
            else "native full-vocabulary last-position logits restricted to declared answer slots"
        ),
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }
