"""
Run the Artificial Fever scraper: scrape products, compute embeddings, upsert to Supabase.
"""
import argparse

# Load .env before any other local imports
from dotenv import load_dotenv
load_dotenv()

from src.scraper import _session, scrape_all_products
from src.embeddings import image_embedding_from_url, text_embedding, build_info_text_for_embedding
from src.db import get_supabase, row_from_payload, upsert_products


def run(dry_run: bool = False, limit: int | None = None):
    session = _session()
    client = None if dry_run else get_supabase()
    batch: list = []
    batch_size = 10
    total = 0
    for i, payload in enumerate(scrape_all_products(session)):
        if limit is not None and i >= limit:
            break
        if not (payload.get("image_url") or "").strip():
            continue  # table requires image_url not null
        image_emb, info_emb = None, None
        if not dry_run:
            image_url = payload.get("image_url")
            image_emb = image_embedding_from_url(image_url) if image_url else None
            info_text = build_info_text_for_embedding(payload)
            info_emb = text_embedding(info_text) if info_text else None
        row = row_from_payload(payload, image_emb, info_emb)
        batch.append(row)
        total += 1
        if dry_run:
            print(f"  [dry-run] {payload['id']} | {payload['title'][:50]}")
            continue
        if len(batch) >= batch_size:
            upsert_products(client, batch)
            print(f"  upserted {len(batch)} products (total {total})")
            batch = []
    if batch and not dry_run:
        upsert_products(client, batch)
        print(f"  upserted {len(batch)} products (total {total})")
    print(f"Done. Total products: {total}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Artificial Fever and sync to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Scrape only, no DB write")
    parser.add_argument("--limit", type=int, default=None, help="Max products to process (default: all)")
    args = parser.parse_args()
    run(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
