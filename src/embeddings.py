"""
Image and text embeddings using google/siglip-base-patch16-384 (768-dim).
"""
from typing import List
import torch
import requests
from PIL import Image
from io import BytesIO

MODEL_ID = "google/siglip-base-patch16-384"
EMBED_DIM = 768


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_processor():
    """Load SigLIP model, image processor, and tokenizer once."""
    from transformers import AutoProcessor, AutoModel, AutoTokenizer
    device = _get_device()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer, device


_model_cache = None
_processor_cache = None
_tokenizer_cache = None
_device_cache = None


def _get_model():
    global _model_cache, _processor_cache, _tokenizer_cache, _device_cache
    if _model_cache is None:
        _model_cache, _processor_cache, _tokenizer_cache, _device_cache = load_model_and_processor()
    return _model_cache, _processor_cache, _tokenizer_cache, _device_cache


def image_embedding_from_url(image_url: str) -> List[float] | None:
    """Download image from URL and return 768-dim embedding (list of floats)."""
    if not image_url or not image_url.startswith("http"):
        return None
    try:
        r = requests.get(image_url, timeout=15)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

    model, processor, _tokenizer, device = _get_model()
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    if isinstance(out, torch.Tensor):
        vec = out[0]
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        vec = out.pooler_output[0]
    else:
        vec = out.last_hidden_state[0, 0]
    return vec.cpu().float().numpy().tolist()


def text_embedding(text: str) -> List[float] | None:
    """Encode text with SigLIP text encoder; return 768-dim embedding."""
    if not text or not str(text).strip():
        return None
    model, _processor, tokenizer, device = _get_model()
    # SigLIP expects padding="max_length"
    inputs = tokenizer(
        str(text)[:10_000],
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    if isinstance(out, torch.Tensor):
        vec = out[0]
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        vec = out.pooler_output[0]
    else:
        vec = out.last_hidden_state[0, 0]
    return vec.cpu().float().numpy().tolist()


def build_info_text_for_embedding(payload: dict) -> str:
    """Concatenate all searchable product info for info_embedding."""
    parts = [
        payload.get("title") or "",
        payload.get("description") or "",
        payload.get("category") or "",
        payload.get("gender") or "",
        payload.get("price") or "",
        payload.get("sale") or "",
        payload.get("metadata") or "",
    ]
    return " ".join(p for p in parts if p).strip()
