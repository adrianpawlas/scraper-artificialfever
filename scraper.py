"""
scraper.py — Fetches all products from Artificial Fever's Shopify JSON API.

Artificial Fever runs on Shopify, which exposes a public products.json endpoint.
We paginate using ?limit=250&page=N until we receive an empty result set.
No browser automation needed.
"""

import logging
import re
import time

import requests
from bs4 import BeautifulSoup

from config import (
    BASE_URL,
    COLLECTIONS,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    WOMENS_COLLECTION_HANDLE,
)

logger = logging.getLogger(__name__)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(REQUEST_HEADERS)
    return s


def _strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def fetch_collection_product_handles(
    session: requests.Session,
    collection_handle: str,
    limit: int = 250,
) -> list[str]:
    """Fetch all product handles from a collection, paginating until empty."""
    handles: list[str] = []
    page = 1

    while True:
        url = f"{BASE_URL}/collections/{collection_handle}/products.json"
        params = {"limit": limit, "page": page}
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        products = data.get("products") or []
        if not products:
            break
        for p in products:
            handle = p.get("handle")
            if handle:
                handles.append(handle)
        page += 1
        time.sleep(0.3)

    return handles


def fetch_product(session: requests.Session, handle: str) -> dict | None:
    """Fetch full product JSON by handle."""
    url = f"{BASE_URL}/products/{handle}.json"
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    if not r.ok:
        return None
    data = r.json()
    return data.get("product")


def get_collection_title(session: requests.Session, collection_handle: str) -> str:
    """Get collection title for category mapping."""
    url = f"{BASE_URL}/collections/{collection_handle}.json"
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    if not r.ok:
        return collection_handle.replace("-", " ").title()
    data = r.json()
    return (data.get("collection") or {}).get("title") or collection_handle.replace("-", " ").title()


def normalize_category(category: str) -> str:
    """e.g. 'Sweaters & Hoodies' -> 'Sweaters, Hoodies'."""
    if not category:
        return ""
    return re.sub(r"\s*&\s*", ", ", category.strip())


def _parse_tags(tags) -> list[str]:
    """Parse Shopify tags (string or list) into a clean list."""
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return list(tags) if tags else []


def build_product_payload(
    product: dict,
    collection_handle: str,
    collection_title: str,
    is_sale_collection: bool,
    is_womens: bool = False,
) -> dict:
    """Build a flat payload for DB from Shopify product + collection context."""
    handle = product.get("handle") or ""
    product_id = str(product.get("id") or "")
    title = (product.get("title") or "").strip()
    body_html = product.get("body_html") or ""
    description = _strip_html(body_html)
    product_type = (product.get("product_type") or "").strip()
    tags_list = _parse_tags(product.get("tags"))

    images = product.get("images") or []
    image_url = ""
    additional_urls: list[str] = []
    if images:
        image_url = images[0].get("src") or ""
        additional_urls = [img.get("src") or "" for img in images[1:] if img.get("src")]

    variants = product.get("variants") or []

    prices_by_currency: dict[str, str] = {}
    sale_prices_by_currency: dict[str, str] = {}
    sizes: list[str] = []
    for v in variants:
        curr = v.get("price_currency") or "USD"
        p = v.get("price") or "0"
        compare = v.get("compare_at_price")
        if compare and float(compare) > float(p):
            prices_by_currency[curr] = compare
            sale_prices_by_currency[curr] = p
        else:
            prices_by_currency[curr] = p
            if curr not in sale_prices_by_currency:
                sale_prices_by_currency[curr] = p
        opt1 = v.get("option1")
        if opt1 and opt1 not in sizes:
            sizes.append(opt1)

    def price_string(d: dict) -> str:
        return ", ".join(f"{v}{k}" for k, v in sorted(d.items()))

    price_str = price_string(prices_by_currency)
    sale_str = ""
    if is_sale_collection or sale_prices_by_currency != prices_by_currency:
        sale_str = price_string(sale_prices_by_currency)

    if product_type:
        category = normalize_category(product_type)
    else:
        category = normalize_category(collection_title)

    gender = "woman" if is_womens else "man"

    metadata_parts = [
        f"title: {title}",
        f"description: {description}",
        f"category: {category}",
        f"gender: {gender}",
        f"price: {price_str}",
        f"sale: {sale_str}" if sale_str else "",
        f"sizes: {', '.join(sizes)}" if sizes else "",
        f"product_type: {product_type}" if product_type else "",
        f"tags: {', '.join(tags_list)}" if tags_list else "",
    ]
    metadata_str = "\n".join(p for p in metadata_parts if p)

    return {
        "id": f"artificialfever-{handle}",
        "handle": handle,
        "product_id": product_id,
        "title": title,
        "description": description or None,
        "image_url": image_url,
        "additional_images": " , ".join(additional_urls) if additional_urls else None,
        "product_url": f"{BASE_URL}/products/{handle}",
        "price": price_str or None,
        "sale": sale_str or None,
        "category": category or None,
        "gender": gender,
        "metadata": metadata_str or None,
        "size": ", ".join(sizes) if sizes else None,
        "second_hand": False,
        "tags": tags_list if tags_list else None,
        "raw_product": product,
        "collection_handle": collection_handle,
        "collection_title": collection_title,
        "is_sale_collection": is_sale_collection,
    }


def fetch_all_products() -> list[dict]:
    """
    Scrape all products from configured collections.
    Returns list of payloads ready for embedding + DB.
    """
    session = _session()

    sale_handles: set[str] = set()
    for collection_handle, is_sale in COLLECTIONS:
        if is_sale:
            sale_handles.update(fetch_collection_product_handles(session, collection_handle))

    womens_handles: set[str] = set()
    womens_handles.update(fetch_collection_product_handles(session, WOMENS_COLLECTION_HANDLE))

    seen_handles: set[str] = set()
    all_payloads: list[dict] = []

    for collection_handle, is_sale_collection in COLLECTIONS:
        collection_title = get_collection_title(session, collection_handle)
        handles = fetch_collection_product_handles(session, collection_handle)

        for handle in handles:
            if handle in seen_handles:
                continue
            seen_handles.add(handle)

            product = fetch_product(session, handle)
            time.sleep(0.2)
            if not product:
                continue

            in_sale = handle in sale_handles
            is_womens = handle in womens_handles

            payload = build_product_payload(
                product,
                collection_handle,
                collection_title,
                in_sale,
                is_womens=is_womens,
            )
            all_payloads.append(payload)

    logger.info(f"Total products fetched: {len(all_payloads)}")
    return all_payloads
