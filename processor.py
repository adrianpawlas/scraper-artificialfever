"""
processor.py — Transforms a scraper payload dict into a Supabase-ready record.

Returns a tuple: (record_dict, info_text)
  • record_dict  – all columns for the `products` table
  • info_text    – concatenated text used to build info_embedding
"""

import logging
from datetime import datetime, timezone

from config import BASE_URL, BRAND, COUNTRY, SOURCE

logger = logging.getLogger(__name__)


def transform_product(payload: dict) -> tuple[dict, str]:
    """
    Convert a scraper payload dict to a Supabase `products` row.

    Returns:
        record   – dict ready for upsert (no embedding vectors yet)
        info_text – full-text string for info_embedding generation
    """
    record: dict = {
        "id": payload["id"],
        "source": SOURCE,
        "brand": BRAND,
        "title": payload.get("title"),
        "description": payload.get("description"),
        "category": payload.get("category"),
        "gender": payload.get("gender"),
        "product_url": payload.get("product_url"),
        "affiliate_url": None,
        "image_url": payload.get("image_url"),
        "additional_images": payload.get("additional_images"),
        "price": payload.get("price"),
        "sale": payload.get("sale"),
        "size": payload.get("size"),
        "tags": payload.get("tags"),
        "metadata": payload.get("metadata"),
        "second_hand": payload.get("second_hand", False),
        "country": COUNTRY,
        "other": None,
        "compressed_image_url": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    info_text = _build_info_text(
        title=payload.get("title", ""),
        description=payload.get("description", ""),
        category=payload.get("category", ""),
        gender=payload.get("gender", ""),
        price=payload.get("price", ""),
        sale=payload.get("sale"),
        tags=payload.get("tags") or [],
    )

    return record, info_text


def _build_info_text(
    title: str,
    description: str,
    category: str,
    gender: str,
    price: str,
    sale: str | None,
    tags: list[str],
) -> str:
    """Build a rich text string used for text embedding."""
    parts = [
        f"Brand: {BRAND}",
        f"Title: {title}",
        f"Category: {category}" if category else "",
        f"Gender: {gender}",
        f"Description: {description}" if description else "",
        f"Price: {price}" if price else "",
        f"Sale Price: {sale}" if sale else "",
        f"Tags: {', '.join(tags)}" if tags else "",
    ]
    return " | ".join(p for p in parts if p)
