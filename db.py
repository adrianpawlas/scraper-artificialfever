"""
db.py — Supabase client with upsert support for the `products` table.

pgvector columns (image_embedding, info_embedding) are serialised as the
bracket-string format expected by PostgREST: "[0.1,0.2,...]"
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client

from config import SOURCE, SUPABASE_KEY, SUPABASE_URL, TABLE_NAME

logger = logging.getLogger(__name__)

FAILED_BATCH_LOG = "failed_batch.log"


def _format_vector(v: list[float]) -> str:
    """Convert a Python float list to the pgvector wire format '[x,y,z,…]'."""
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


class SupabaseClient:
    """Thin wrapper around supabase-py with upsert helpers."""

    def __init__(self) -> None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        logger.info(f"Connecting to Supabase: {SUPABASE_URL}")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client ready.")
        self._existing_products: dict[str, dict] | None = None

    def fetch_existing_products(self) -> dict[str, dict]:
        """Fetch all products for this source from the database."""
        if self._existing_products is not None:
            return self._existing_products

        logger.info(f"Fetching existing products for source: {SOURCE}")
        response = self.client.table(TABLE_NAME).select("*").eq("source", SOURCE).execute()
        self._existing_products = {
            p["product_url"]: p for p in response.data if p.get("product_url")
        }
        logger.info(f"Found {len(self._existing_products)} existing products")
        return self._existing_products

    def has_changed(self, existing: dict | None, new_record: dict) -> bool:
        """Check if any relevant fields have changed."""
        if existing is None:
            return True

        fields_to_check = [
            "title",
            "description",
            "price",
            "sale",
            "category",
            "gender",
            "size",
            "image_url",
            "additional_images",
            "tags",
        ]

        for field in fields_to_check:
            existing_val = existing.get(field)
            new_val = new_record.get(field)
            if str(existing_val) != str(new_val):
                return True

        return False

    def needs_new_embedding(self, existing: dict | None, new_record: dict) -> bool:
        """Check if embeddings need to be regenerated (new product or image changed)."""
        if existing is None:
            return True
        return existing.get("image_url") != new_record.get("image_url")

    def _prepare_row(self, row: dict) -> dict:
        """Serialize vector columns for PostgREST."""
        prepared = dict(row)
        for col in ("image_embedding", "info_embedding"):
            val = prepared.get(col)
            if isinstance(val, list):
                prepared[col] = _format_vector(val)
        prepared["updated_at"] = datetime.now(timezone.utc).isoformat()
        return prepared

    def upsert_batch(self, records: list[dict], retries: int = 3) -> tuple[int, int]:
        """
        Insert/update a batch of products.
        Returns (success_count, failed_count).
        """
        if not records:
            return 0, 0

        rows = [self._prepare_row(r) for r in records]

        for attempt in range(1, retries + 1):
            try:
                result = (
                    self.client.table(TABLE_NAME)
                    .upsert(rows, on_conflict="source,product_url", ignore_duplicates=False)
                    .execute()
                )
                if result.data:
                    return len(rows), 0
                logger.warning(f"  Batch upsert returned no data, retrying...")
            except Exception as exc:
                logger.warning(f"  Batch upsert attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    continue

        logger.error(f"  Batch upsert failed after {retries} retries, logging to file")
        self._log_failed_batch(records)
        return 0, len(records)

    def _log_failed_batch(self, records: list[dict]) -> None:
        """Log failed product IDs to a file."""
        ids = [r.get("id", r.get("product_url", "unknown")) for r in records]
        with open(FAILED_BATCH_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} - Failed IDs: {ids}\n")

    def delete_stale_products(self, seen_product_urls: set[str]) -> int:
        """
        Delete products not seen in current scrape run.
        Returns count of deleted products.
        """
        existing = self.fetch_existing_products()
        existing_urls = set(existing.keys())
        stale_urls = existing_urls - seen_product_urls

        if not stale_urls:
            logger.info("No stale products to delete")
            return 0

        logger.info(f"Deleting {len(stale_urls)} stale products")

        deleted = 0
        for url in stale_urls:
            try:
                self.client.table(TABLE_NAME).delete().eq("source", SOURCE).eq("product_url", url).execute()
                deleted += 1
            except Exception as exc:
                logger.error(f"  Failed to delete stale product {url}: {exc}")

        return deleted

    def mark_product_seen(self, product_url: str) -> None:
        """Mark a product as seen in current run for stale detection."""
        if self._existing_products is not None and product_url in self._existing_products:
            del self._existing_products[product_url]
