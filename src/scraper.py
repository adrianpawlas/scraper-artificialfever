"""
Scrape Artificial Fever (Shopify) collections and product details.
"""
import re
import time
from typing import Iterator
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://artificialfever.com"
COLLECTIONS = [
    ("frontpage", False),   # (handle, is_sale_collection)
    ("full-storage-sale", True),
]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    })
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
) -> Iterator[str]:
    """Yield product handles from a collection, paginating until 0 products."""
    page = 1
    while True:
        url = f"{BASE_URL}/collections/{collection_handle}/products.json"
        params = {"limit": limit, "page": page}
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        products = data.get("products") or []
        if not products:
            break
        for p in products:
            handle = p.get("handle")
            if handle:
                yield handle
        page += 1
        time.sleep(0.3)


def fetch_product(
    session: requests.Session,
    handle: str,
) -> dict | None:
    """Fetch full product JSON by handle."""
    url = f"{BASE_URL}/products/{handle}.json"
    r = session.get(url, timeout=30)
    if not r.ok:
        return None
    data = r.json()
    return data.get("product")


def get_collection_title(session: requests.Session, collection_handle: str) -> str:
    """Get collection title for category mapping."""
    url = f"{BASE_URL}/collections/{collection_handle}.json"
    r = session.get(url, timeout=30)
    if not r.ok:
        return collection_handle.replace("-", " ").title()
    data = r.json()
    return (data.get("collection") or {}).get("title") or collection_handle.replace("-", " ").title()


def normalize_category(category: str) -> str:
    """e.g. 'Sweaters & Hoodies' -> 'Sweaters, Hoodies'."""
    if not category:
        return ""
    return re.sub(r"\s*&\s*", ", ", category.strip())


def build_product_payload(
    product: dict,
    collection_handle: str,
    collection_title: str,
    is_sale_collection: bool,
) -> dict:
    """Build a flat payload for DB from Shopify product + collection context."""
    handle = product.get("handle") or ""
    product_id = str(product.get("id") or "")
    title = (product.get("title") or "").strip()
    body_html = product.get("body_html") or ""
    description = _strip_html(body_html)
    product_type = (product.get("product_type") or "").strip()
    tags = product.get("tags") or ""
    if isinstance(tags, str):
        tags_list = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        tags_list = list(tags) if tags else []

    # Images: first is primary, rest additional
    images = product.get("images") or []
    image_url = ""
    additional_urls: list[str] = []
    if images:
        image_url = images[0].get("src") or ""
        additional_urls = [img.get("src") or "" for img in images[1:] if img.get("src")]

    # Variants: price, compare_at_price, options (size, color)
    variants = product.get("variants") or []
    options = product.get("options") or []

    # Original price (no sale): prefer compare_at_price when it's a sale, else price
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

    # Category: product_type or collection title
    if product_type:
        category = normalize_category(product_type)
    else:
        category = normalize_category(collection_title)

    # Metadata: all info in one place
    metadata_parts = [
        f"title: {title}",
        f"description: {description}",
        f"category: {category}",
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
        "gender": "man",
        "metadata": metadata_str or None,
        "size": ", ".join(sizes) if sizes else None,
        "second_hand": False,
        "tags": tags_list if tags_list else None,
        "raw_product": product,
        "collection_handle": collection_handle,
        "collection_title": collection_title,
        "is_sale_collection": is_sale_collection,
    }


def _sale_collection_handles(session: requests.Session) -> set[str]:
    """Return set of product handles that appear in the sale collection."""
    sale_handles = set()
    for collection_handle, is_sale in COLLECTIONS:
        if not is_sale:
            continue
        for handle in fetch_collection_product_handles(session, collection_handle):
            sale_handles.add(handle)
    return sale_handles


def scrape_all_products(session: requests.Session | None = None) -> Iterator[dict]:
    """
    Scrape all products from configured collections.
    Yields payloads ready for embedding + DB (with raw_product for metadata).
    Products in full-storage-sale get sale price in sale column.
    """
    session = session or _session()
    sale_handles = _sale_collection_handles(session)
    seen_handles: set[str] = set()

    for collection_handle, is_sale_collection in COLLECTIONS:
        collection_title = get_collection_title(session, collection_handle)
        for handle in fetch_collection_product_handles(session, collection_handle):
            if handle in seen_handles:
                continue
            seen_handles.add(handle)
            product = fetch_product(session, handle)
            time.sleep(0.2)
            if not product:
                continue
            # Use sale column when product is in sale collection (or has compare_at_price)
            in_sale = handle in sale_handles
            payload = build_product_payload(
                product,
                collection_handle,
                collection_title,
                in_sale,
            )
            yield payload
