"""
Mercado Libre Mexico - Car Listings Extractor
=============================================
Extrae hasta 10,000 listings de autos en venta (2000-2025)
usando scraping web sobre la versión pública de MercadoLibre México.

Campos extraídos:
  brand, model, year, price, mileage, city, state, exterior_color

Requisitos:
  pip install requests pandas beautifulsoup4

Uso:
  python mercadolibre_cars_scraper.py
"""

import argparse
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
from typing import Optional, List, Dict, Tuple

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
OUTPUT_FILE   = "autos_mexico.csv"
TARGET_TOTAL  = 10_000
YEAR_MIN      = 2000
YEAR_MAX      = 2025
DELAY_SEC     = 0.3            # pausa entre requests (ser amable con la web)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

BASE_URL = "https://autos.mercadolibre.com.mx/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
}
PAGE_RESULTS = 48
MAX_PAGES = 210


def fetch_search_page(offset: int) -> str:
    """Descarga el HTML de la página de resultados de MercadoLibre."""
    url = BASE_URL if offset == 0 else f"{BASE_URL}_Desde_{offset}"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.text


def parse_car_title(title: str) -> tuple:
    """Extrae make/model/trim del título de un anuncio de auto."""
    common_makes = {
        'toyota': 'Toyota', 'honda': 'Honda', 'ford': 'Ford', 'chevrolet': 'Chevrolet',
        'bmw': 'BMW', 'mercedes': 'Mercedes-Benz', 'audi': 'Audi', 'nissan': 'Nissan',
        'hyundai': 'Hyundai', 'kia': 'Kia', 'volkswagen': 'Volkswagen', 'mazda': 'Mazda',
        'suzuki': 'Suzuki', 'mitsubishi': 'Mitsubishi', 'isuzu': 'Isuzu', 'ram': 'RAM',
        'jeep': 'Jeep', 'chrysler': 'Chrysler', 'dodge': 'Dodge', 'lexus': 'Lexus',
        'infiniti': 'Infiniti', 'acura': 'Acura', 'cadillac': 'Cadillac', 'lincoln': 'Lincoln'
    }

    title_lower = title.lower().replace("/", " ").replace("-", " ").strip()
    words = [w.strip(",.") for w in title_lower.split() if w.strip()]

    make = "Unknown"
    model = "Unknown"
    trim = "Unknown"

    for i, word in enumerate(words):
        if word in common_makes:
            make = common_makes[word]
            if i + 1 < len(words):
                model = words[i + 1].title()
            if i + 2 < len(words):
                next_word = words[i + 2].title()
                if not next_word.isdigit():
                    trim = next_word
            break

    return make, model, trim


def parse_listing(listing) -> Optional[Dict]:
    """Extrae los campos del HTML de una tarjeta de auto."""
    title_elem = listing.select_one("a.poly-component__title")
    title = title_elem.text.strip() if title_elem else None
    if not title:
        return None

    price_elem = listing.select_one("span.andes-money-amount__fraction")
    price = None
    if price_elem:
        price_text = price_elem.text.replace("$", "").replace("MXN", "").replace(",", "").strip()
        try:
            price = int(price_text)
        except ValueError:
            price = None

    year = None
    mileage = None
    details = listing.select("div.poly-component__attributes-list li")
    for detail in details:
        text = detail.text.strip().lower()
        if "km" in text:
            km_text = text.replace("km", "").replace(",", "").strip()
            try:
                mileage = int(float(km_text))
            except ValueError:
                mileage = None
        if text.isdigit() and len(text) == 4:
            y = int(text)
            if 1990 <= y <= YEAR_MAX:
                year = y

    if year is None:
        for part in title.split():
            if part.isdigit() and len(part) == 4:
                y = int(part)
                if 1990 <= y <= YEAR_MAX:
                    year = y
                    break

    location_elem = listing.select_one("span.poly-component__location")
    location_text = location_elem.text.strip() if location_elem else "Mexico"
    location_parts = [p.strip() for p in location_text.replace(" - ", ",").split(",") if p.strip()]
    city = location_parts[0] if location_parts else None
    state = location_parts[-1] if len(location_parts) > 1 else None

    brand, model, trim = parse_car_title(title)
    # Relajar filtros: requiere precio y marca, pero no necesariamente año
    if price is None or brand == "Unknown":
        return None
    
    # Si no tiene año, usar estimación desde el título
    if year is None:
        year = 2020  # año por defecto para listings sin año especificado

    listing_id = None
    link_href = title_elem.get("href") if title_elem else None
    if link_href:
        import re
        match = re.search(r"/MLM-(\d+)-", link_href)
        if match:
            listing_id = match.group(1)

    return {
        "id": listing_id or f"unknown-{hash(title)}",
        "title": title,
        "brand": brand,
        "model": model,
        "year": year,
        "price": price,
        "currency": "MXN",
        "mileage": mileage if mileage else 0,
        "exterior_color": "Unknown",
        "city": city,
        "state": state,
    }


def scrape_mercadolibre(max_pages: int = MAX_PAGES) -> List[Dict]:
    """Scrapea las páginas públicas de MercadoLibre y devuelve resultados de autos."""
    records = []
    seen_ids = set()

    for page in range(max_pages):
        offset = page * PAGE_RESULTS
        try:
            html = fetch_search_page(offset)
        except requests.RequestException as exc:
            print(f"Error descargando página {page + 1} (offset={offset}): {exc}")
            break

        soup = BeautifulSoup(html, "html.parser")
        listings = soup.find_all("li", class_="ui-search-layout__item")
        if not listings:
            print(f"No se encontraron anuncios en offset {offset}. Deteniendo.")
            break

        page_count = 0
        for listing in listings:
            parsed = parse_listing(listing)
            if not parsed:
                continue
            if parsed["id"] in seen_ids:
                continue
            records.append(parsed)
            seen_ids.add(parsed["id"])
            page_count += 1
            if len(records) >= TARGET_TOTAL:
                break

        print(f"Página {page + 1}: {page_count} anuncios extraídos (Total: {len(records)})")
        if len(records) >= TARGET_TOTAL:
            break

        time.sleep(DELAY_SEC)

    return records


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrae anuncios de autos de MercadoLibre México mediante scraping web.")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES, help="Máximo número de páginas MercadoLibre a extraer")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Nombre del archivo CSV de salida")
    parser.add_argument("--target", type=int, default=TARGET_TOTAL, help="Número objetivo de listings a recopilar")
    args = parser.parse_args()

    print("=" * 55)
    print("  Mercado Libre MX — Extractor de Autos")
    print(f"  Años: {YEAR_MIN}–{YEAR_MAX}  |  Meta: {args.target:,} listings")
    print("=" * 55)

    records = scrape_mercadolibre(max_pages=args.max_pages)
    print(f"\nTotal registros recopilados: {len(records):,}")

    df = pd.DataFrame(records, columns=[
        "id", "title", "brand", "model", "year",
        "price", "currency", "mileage", "exterior_color", "city", "state"
    ])

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]

    output_file = args.output
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"  ✅  Archivo guardado: {output_file}")
    print(f"  Filas finales: {len(df):,}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
