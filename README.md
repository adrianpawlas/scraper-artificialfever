# Scraper: Artificial Fever

Scrapes all products from [Artificial Fever](https://artificialfever.com/), computes 768‑dim image and text embeddings with `google/siglip-base-patch16-384`, and upserts into a Supabase `products` table.

## Requirements

- Python 3.10+
- Supabase project with `products` table (see schema below)

## Setup

1. Clone and install:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set:

   - `SUPABASE_URL` – your Supabase project URL  
   - `SUPABASE_KEY` – your Supabase anon or service role key  

## Run

- **Full run (scrape + embeddings + Supabase upsert):**

  ```bash
  python main.py
  ```

- **Dry run (scrape only, no DB, no embeddings):**

  ```bash
  python main.py --dry-run
  ```

- **Limit number of products (e.g. for testing):**

  ```bash
  python main.py --limit 5
  ```

## Automation

- **Daily run:** A GitHub Action runs the scraper every day at **00:00 UTC** (`Schedule` in `.github/workflows/scrape.yml`).
- **Manual run:** In the repo go to **Actions → Scrape Artificial Fever → Run workflow**.

Add these repository secrets:

- `SUPABASE_URL`  
- `SUPABASE_KEY`  

## Data

- **Collections:** `frontpage` (Shop All) and `full-storage-sale`. Pagination continues until a page returns 0 products.
- **Fields:** `source` = `scraper-artificialfever`, `brand` = `Artificial Fever`, `gender` = `man`, `second_hand` = `false`. Image and text embeddings are 768‑dim (SigLIP). Sale prices are set for products in `full-storage-sale` and when `compare_at_price` is present.
