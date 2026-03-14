"""
config.py — Central configuration for the Artificial Fever scraper.
All secrets are loaded from .env (or environment variables).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "").strip()
TABLE_NAME: str = "products"

# ── Artificial Fever ─────────────────────────────────────────────────────────
BASE_URL: str = "https://artificialfever.com"
COLLECTIONS: list[tuple[str, bool]] = [
    ("frontpage", False),
    ("full-storage-sale", True),
]
WOMENS_COLLECTION_HANDLE: str = "womens"

# ── Brand metadata ───────────────────────────────────────────────────────────
SOURCE: str = "scraper-artificialfever"
BRAND: str = "Artificial Fever"
COUNTRY: str = ""

# ── SigLIP embedding model ───────────────────────────────────────────────────
MODEL_NAME: str = "google/siglip-base-patch16-384"
EMBEDDING_DIM: int = 768
SIGLIP_MAX_TEXT_LENGTH: int = 64

# ── HTTP ──────────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT: int = 30
REQUEST_HEADERS: dict = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}
