"""
main.py — Artificial Fever full scraper entry point.

Pipeline:
  1. Fetch all products from Artificial Fever's Shopify JSON API.
  2. Transform each product into a Supabase-ready record.
  3. Generate a 768-dim SigLIP image embedding from the primary product image.
  4. Generate a 768-dim SigLIP text embedding from all product info.
  5. Upsert the record into the `products` table.

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

    success_count = 0
    skip_count = 0
    error_count = 0

    for payload in tqdm(products, desc="Products", unit="item"):
        title = payload.get("title", "UNKNOWN")

        try:
            record, info_text = transform_product(payload)

            image_url = record.get("image_url", "")
            if not image_url:
                logger.warning(f"  ⚠ No image_url for '{title}' — skipping.")
                skip_count += 1
                continue

            if not args.dry_run:
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

                db.upsert(record)
                logger.info(f"  ✓ Upserted: {record['id']} — {title}")
            else:
                logger.info(f"  [dry-run] {record['id']} | {title[:50]}")

            success_count += 1

        except Exception as exc:
            error_count += 1
            logger.error(f"  ✗ FAILED: {title} — {exc}", exc_info=True)

        time.sleep(0.05)

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Done in {elapsed:.1f}s")
    logger.info(f"  ✓ Success : {success_count}")
    logger.info(f"  ⚠ Skipped : {skip_count}")
    logger.info(f"  ✗ Errors  : {error_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
