"""
Supabase products table upsert. Maps scraper payload + embeddings to table schema.
"""
import os
from typing import Any

from supabase import create_client, Client

SOURCE = "scraper-artificialfever"
BRAND = "Artificial Fever"


def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)


def row_from_payload(
    payload: dict,
    image_embedding: list[float] | None,
    info_embedding: list[float] | None,
) -> dict[str, Any]:
    """Build one products table row from scraper payload and embeddings."""
    return {
        "id": payload["id"],
        "source": SOURCE,
        "product_url": payload["product_url"],
        "affiliate_url": None,
        "image_url": payload["image_url"],
        "brand": BRAND,
        "title": payload["title"],
        "description": payload.get("description"),
        "category": payload.get("category"),
        "gender": payload.get("gender"),
        "metadata": payload.get("metadata"),
        "size": payload.get("size"),
        "second_hand": payload.get("second_hand", False),
        "image_embedding": image_embedding,
        "country": None,
        "compressed_image_url": None,
        "tags": payload.get("tags"),
        "other": None,
        "price": payload.get("price"),
        "sale": payload.get("sale"),
        "additional_images": payload.get("additional_images"),
        "info_embedding": info_embedding,
    }


def upsert_products(client: Client, rows: list[dict]) -> None:
    """Upsert rows into public.products. On conflict (source, product_url) update."""
    if not rows:
        return
    # Supabase/PostgREST upsert: use on_conflict
    client.table("products").upsert(
        rows,
        on_conflict="source,product_url",
        ignore_duplicates=False,
    )
