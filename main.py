"""
main.py — Artificial Fever full scraper entry point.

Pipeline:
  1. Fetch all products from Artificial Fever's Shopify JSON API.
  2. Transform each product into a Supabase-ready record.
  3. Generate embeddings only for new/changed products.
  4. Batch upsert to Supabase (50 products per batch).
  5. Delete stale products not seen in current run.
  6. Print summary.

Usage:
  python main.py
  python main.py --dry-run
  python main.py --limit 10
"""

import argparse
import logging
import time
from datetime import datetime, timezone

from tqdm import tqdm

from config import MODEL_NAME
from db import SupabaseClient
from embedder import SigLIPEmbedder, build_info_text
from processor import transform_product
from scraper import fetch_all_products

BATCH_SIZE = 50
EMBEDDING_DELAY = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Artificial Fever and sync to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Scrape only, no DB write")
    parser.add_argument("--limit", type=int, default=None, help="Max products to process")
    args = parser.parse_args()

    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("  Artificial Fever Scraper  —  started at %s", start_time.isoformat())
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - no DB writes")

    embedder = SigLIPEmbedder(MODEL_NAME)

    db = None if args.dry_run else SupabaseClient()

    products = fetch_all_products()

    if not products:
        logger.error("No products fetched. Aborting.")
        return

    if args.limit:
        products = products[:args.limit]

    total = len(products)
    logger.info(f"Processing {total} products …\n")

    new_count = 0
    updated_count = 0
    unchanged_count = 0
    skipped_no_image = 0
    error_count = 0

    batch: list[dict] = []
    seen_product_urls: set[str] = set()

    for payload in tqdm(products, desc="Products", unit="item"):
        title = payload.get("title", "UNKNOWN")

        try:
            record, info_text = transform_product(payload)
            product_url = record.get("product_url", "")

            if not product_url:
                logger.warning(f"  ⚠ No product_url for '{title}' — skipping.")
                error_count += 1
                continue

            image_url = record.get("image_url", "")
            if not image_url:
                logger.warning(f"  ⚠ No image_url for '{title}' — skipping.")
                skipped_no_image += 1
                continue

            seen_product_urls.add(product_url)

            if not args.dry_run:
                existing = db.fetch_existing_products().get(product_url)
                has_changes = db.has_changed(existing, record)
                needs_embedding = db.needs_new_embedding(existing, record)

                if not has_changes:
                    unchanged_count += 1
                    logger.debug(f"  = Unchanged: {record['id']} — {title}")
                    continue

                is_new = existing is None

                if needs_embedding:
                    image_emb = embedder.embed_image(image_url)
                    if image_emb:
                        record["image_embedding"] = image_emb
                    else:
                        logger.warning(f"  ⚠ Image embedding failed for '{title}'")

                    if info_text:
                        info_emb = embedder.embed_text(info_text)
                        if info_emb:
                            record["info_embedding"] = info_emb
                        else:
                            logger.warning(f"  ⚠ Info text embedding failed for '{title}'")

                    time.sleep(EMBEDDING_DELAY)
                else:
                    if existing:
                        record["image_embedding"] = existing.get("image_embedding")
                        record["info_embedding"] = existing.get("info_embedding")

                if is_new:
                    new_count += 1
                    logger.info(f"  + New: {record['id']} — {title}")
                else:
                    updated_count += 1
                    logger.info(f"  ~ Updated: {record['id']} — {title}")

            else:
                logger.info(f"  [dry-run] {record['id']} | {title[:50]}")

            batch.append(record)

            if len(batch) >= BATCH_SIZE:
                if not args.dry_run:
                    success, failed = db.upsert_batch(batch)
                    if failed > 0:
                        error_count += failed
                batch = []

        except Exception as exc:
            error_count += 1
            logger.error(f"  ✗ FAILED: {title} — {exc}", exc_info=True)

    if batch and not args.dry_run:
        success, failed = db.upsert_batch(batch)
        if failed > 0:
            error_count += failed

    stale_deleted = 0
    if not args.dry_run:
        stale_deleted = db.delete_stale_products(seen_product_urls)

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info("")
    logger.info("=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  New products added   : {new_count}")
    logger.info(f"  Products updated    : {updated_count}")
    logger.info(f"  Products unchanged  : {unchanged_count}")
    logger.info(f"  Skipped (no image)  : {skipped_no_image}")
    logger.info(f"  Stale products del  : {stale_deleted}")
    logger.info(f"  Errors              : {error_count}")
    logger.info(f"  Total processed    : {total}")
    logger.info(f"  Done in {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
