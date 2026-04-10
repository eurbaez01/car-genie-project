"""
SuperCarros.com Web Scraper
Scrapes car listings and specifications from SuperCarros.com (Dominican Republic marketplace)
Collects price, specs, and vehicle details for car price intelligence analysis.
"""

import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from typing import List, Dict, Optional
from urllib.parse import urljoin
import re
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SuperCarrosScraper:
    """
    Scraper for SuperCarros.com car listings.
    Handles both listing pages and individual car detail pages.
    """

    def __init__(self, base_url: str = "https://www.supercarros.com"):
        """
        Initialize the scraper with Cloudflare bypass.

        Args:
            base_url: Base URL of SuperCarros.com
        """
        self.base_url = base_url
        # Use cloudscraper to bypass Cloudflare
        self.session = cloudscraper.create_scraper()
        
        # Enhanced headers to avoid detection
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        })
        self.listings_data = []

        # Popular car models to target for scraping
        self.car_models = [
            "Toyota Corolla",
            "Toyota Camry",
            "Honda Civic",
            "Honda Accord",
            "Hyundai Sonata",
            "Hyundai Elantra",
            "Kia K5",
            "Kia Rio",
            "Hyundai Accent",
            "Nissan Sentra",

            "Honda CR-V",
            "Toyota RAV4",
            "Hyundai Tucson",
            "Kia Sportage",
            "Ford Explorer",
            "Nissan X-Trail",
            "Mazda CX-5",
            "Chevrolet Captiva",
            "Suzuki Vitara",
            "Toyota Land Cruiser",

            "Toyota Hilux",
            "Nissan Frontier",
            "Ford Ranger",
            "Isuzu D-Max",
            "Mitsubishi L200",

            "Kia Picanto",
            "Hyundai Verna",
            "Kia Morning",
            "Suzuki Swift",
            "Toyota Yaris",

            "Mazda 3",
            "Mazda CX-3",
            "Chevrolet Tahoe",
            "Jeep Grand Cherokee",
            "Dodge Durango",
            "Ford Escape",
            "Toyota Prado",
            "BMW Serie 3",
            "Mercedes-Benz C-Class",
            "Audi A4",

            # 🔥 10 adicionales
            "Hyundai Santa Fe",
            "Kia Sorento",
            "Nissan Altima",
            "Toyota Fortuner",
            "Chevrolet Silverado",
            "Ram 1500",
            "Volkswagen Jetta",
            "Volkswagen Tiguan",
            "Subaru Forester",
            "Peugeot 3008",

            "Lexus RX",
            "Lexus NX",
            "BMW X5",
            "BMW X3",
            "Mercedes-Benz E-Class",
            "Mercedes-Benz GLE",
            "Audi Q5",
            "Audi Q7",
            "Porsche Cayenne",
            "Volvo XC90"
        ]

    def get_search_pages(self, 
                        vehicle_type: str = "cualquier-tipo",
                        max_pages: int = 5) -> List[str]:
        """
        Get URLs for search result pages.
        
        Args:
            vehicle_type: Type of vehicle (sedan, camioneta, jeepeta, etc)
                         Default: "cualquier-tipo" (all types)
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of page URLs
        """
        page_urls = []
        
        for page in range(1, max_pages + 1):
            url = f"{self.base_url}/carros/{vehicle_type}/"
            if page > 1:
                url += f"pagina-{page}/"
            page_urls.append(url)
            logger.info(f"Added page URL: {url}")
        
        return page_urls

    def get_model_search_urls(self, models: List[str] = None, max_pages_per_model: int = 3, year_ranges: List[tuple] = None) -> List[str]:
        """
        Generate search URLs for specific car models using year ranges.
        
        Format: https://www.supercarros.com/carros/cualquier-tipo/cualquier-provincia/Brand/Model/?YearFrom=YYYY&YearTo=YYYY
        
        Uses year ranges (e.g., 2010-2016, 2017-2023, 2024-2027) instead of individual years.
        
        Args:
            models: List of car models to search for (uses self.car_models if None)
            max_pages_per_model: Maximum pages to scrape per model per year range
            year_ranges: List of year range tuples (e.g., [(2010, 2016), (2017, 2023), (2024, 2027)])
                        If None, uses default: [(2010, 2016), (2017, 2023), (2024, 2027)]
            
        Returns:
            List of search URLs for specific models and year ranges
        """
        if models is None:
            models = self.car_models
        
        if year_ranges is None:
            year_ranges = [(2010, 2016), (2017, 2023), (2024, 2027)]

        search_urls = []

        for model in models:
            # Parse model string to get Brand and Model name
            # Example: "Toyota Corolla" -> brand="Toyota", model_name="Corolla"
            parts = model.split()
            if len(parts) < 2:
                logger.warning(f"Skipping invalid model format: {model}")
                continue
            
            brand = parts[0]  # First part is brand (e.g., "Toyota")
            model_name = ' '.join(parts[1:])  # Rest is model (e.g., "Corolla")
            
            # Clean for URL (capitalize, keep spaces in URL)
            brand_url = brand.capitalize()
            model_url = model_name.replace(' ', '%20')  # URL encode spaces
            
            # Loop through each year range
            for year_from, year_to in year_ranges:
                for page in range(1, max_pages_per_model + 1):
                    # Build correct URL format with year range
                    url = f"{self.base_url}/carros/cualquier-tipo/cualquier-provincia/{brand_url}/{model_url}/"
                    
                    # Add year range parameters
                    url += f"?YearFrom={year_from}&YearTo={year_to}"
                    
                    # Add pagination if needed
                    if page > 1:
                        url += f"&page={page}"
                    
                    search_urls.append(url)
                    logger.debug(f"Generated URL for {model} ({year_from}-{year_to}): {url}")

        logger.info(f"Generated {len(search_urls)} search URLs")
        logger.info(f"Models: {len(models)}, Year Ranges: {len(year_ranges)}, Pages per range: {max_pages_per_model}")
        logger.info(f"Total combinations: {len(models)} models × {len(year_ranges)} ranges × {max_pages_per_model} pages = {len(search_urls)} URLs")
        return search_urls

    def scrape_listing_page(self, page_url: str, max_retries: int = 3) -> List[str]:
        """
        Scrape a listing page and extract individual car URLs with retry logic.
        
        Args:
            page_url: URL of the listing page
            max_retries: Maximum number of retries on failure
            
        Returns:
            List of individual car listing URLs
        """
        car_urls = []
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Scraping listing page: {page_url} (Attempt {attempt + 1}/{max_retries})")
                response = self.session.get(page_url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all car listings - they typically have pattern like:
                # <a href="/mercedesbenz-claseglc/1594463/">
                # Look for links with numeric IDs in URL
                car_links = soup.find_all('a', href=re.compile(r'/\d+/$'))
                
                for link in car_links:
                    href = link.get('href')
                    if href:
                        car_url = urljoin(self.base_url, href)
                        car_urls.append(car_url)
                        logger.debug(f"Found car URL: {car_url}")
                
                logger.info(f"Found {len(car_urls)} car listings on page")
                # Random delay between 2-4 seconds to avoid detection
                delay = random.uniform(2, 4)
                time.sleep(delay)
                return car_urls
                
            except Exception as e:
                logger.warning(f"Error scraping listing page {page_url} (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds + random jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to scrape {page_url} after {max_retries} attempts")
        
        return car_urls

    def extract_price(self, price_text: str) -> Dict[str, Optional[float]]:
        """
        Extract price value and currency from text.
        
        Args:
            price_text: Price text like "US$ 29,500" or "RD$ 1,500,000"
            
        Returns:
            Dict with 'amount', 'currency', 'usd_equivalent'
        """
        try:
            # Match pattern: Currency Code $ amount with possible commas
            match = re.search(r'(US\$|RD\$)\s*([\d,]+(?:\.\d+)?)', price_text.strip())
            
            if match:
                currency = match.group(1).strip('$').strip()
                amount = float(match.group(2).replace(',', ''))
                
                # Convert RD$ to USD if needed (approximate rate: 1 USD = 59 RD$)
                usd_equivalent = amount if currency == 'US' else amount / 59
                
                return {
                    'amount': amount,
                    'currency': currency,
                    'usd_equivalent': usd_equivalent
                }
        except Exception as e:
            logger.warning(f"Error parsing price '{price_text}': {e}")
        
        return {'amount': None, 'currency': None, 'usd_equivalent': None}

    def parse_year_make_model(self, title_text: str) -> Dict[str, Optional[str]]:
        """
        Parse vehicle title to extract year, make, model, trim.
        Example: "Mercedes-Benz Clase GLC 300 AMG 2019"
        
        Args:
            title_text: Title text from page
            
        Returns:
            Dict with year, make, model, trim
        """
        try:
            # Extract year (4 digits)
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', title_text)
            year = int(year_match.group(1)) if year_match else None
            
            # Remove year from title for further parsing
            title_clean = re.sub(r'\b(19\d{2}|20\d{2})\b', '', title_text).strip()
            
            # Split on known make patterns - this is a basic approach
            # For better results, maintain a list of known makes
            parts = title_clean.split()
            
            make = None
            model = None
            trim = None
            
            if len(parts) >= 2:
                # First part is usually make
                make = parts[0]
                # Second part (if hyphenated like "Clase GLC") is model
                if len(parts) >= 3:
                    model = ' '.join(parts[1:3])  # e.g., "Clase GLC"
                    trim = ' '.join(parts[3:]) if len(parts) > 3 else None  # e.g., "300 AMG"
                else:
                    model = parts[1]
            
            return {
                'year': year,
                'make': make,
                'model': model,
                'trim': trim,
                'full_title': title_clean
            }
        except Exception as e:
            logger.warning(f"Error parsing title '{title_text}': {e}")
            return {'year': None, 'make': None, 'model': None, 'trim': None}

    def scrape_car_details(self, car_url: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Scrape detailed information from a single car listing page with retry logic.
        
        Args:
            car_url: URL of the car listing
            max_retries: Maximum number of retries on failure
            
        Returns:
            Dictionary with car details or None if error
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Scraping car details: {car_url} (Attempt {attempt + 1}/{max_retries})")
                response = self.session.get(car_url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                car_data = {'url': car_url}
                
                # ============ TITLE & PRICE ============
                title_elem = soup.find('h1')
                if title_elem:
                    car_data['title'] = title_elem.text.strip()
                    # Parse year, make, model, trim
                    parsed = self.parse_year_make_model(car_data['title'])
                    car_data.update(parsed)
                
                # Extract price
                price_elem = soup.find('h3', string=lambda text: text and 'US$' in text or 'RD$' in text)
                if not price_elem:
                    # Try alternative selector
                    price_elem = soup.find(string=re.compile(r'US\$|RD\$'))
                
                if price_elem:
                    price_info = self.extract_price(str(price_elem))
                    car_data.update(price_info)
                    car_data['price_text'] = str(price_elem).strip()
                
                # ============ GENERAL DATA TABLE ============
                # Find the "Datos Generales" section
                data_table = soup.find('table')
                
                if data_table:
                    rows = data_table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            key = cells[0].text.strip().rstrip(':').lower()
                            value = cells[1].text.strip()
                            
                            # Map Spanish keys to English
                            key_mapping = {
                                'precio': 'price_listed',
                                'motor': 'engine_cylinders',
                                'exterior': 'exterior_color',
                                'interior': 'interior_color',
                                'tipo': 'vehicle_type',
                                'uso': 'usage_mileage',
                                'mileage': 'mileage',
                                'miles': 'mileage',
                                'km': 'mileage',
                                'kilometraje': 'mileage',
                                'recorrido': 'mileage',
                                'combustible': 'fuel_type',
                                'carga': 'cargo_capacity',
                                'transmisión': 'transmission',
                                'puertas': 'doors',
                                'tracción': 'drive_type',
                                'pasajeros': 'passengers',
                                'cilindros': 'cylinders'
                            }
                            
                            mapped_key = key_mapping.get(key, key)
                            car_data[mapped_key] = value
                
                # ============ EXTRACT MILEAGE FROM USO LABEL ============
                # Look for "Uso:" label followed by mileage value
                uso_label = soup.find('label', string=re.compile(r'uso', re.IGNORECASE))
                if uso_label:
                    # Find the next td element after the uso label
                    next_td = uso_label.find_next('td')
                    if next_td:
                        mileage_text = next_td.text.strip()
                        car_data['usage_mileage'] = mileage_text
                        logger.debug(f"Extracted mileage from Uso label: {mileage_text}")
                
                # ============ EXTRACT MILEAGE FROM USAGE FIELD ============
                # Parse mileage from usage field (e.g., "20,000 Km" or "N/D Mi")
                if 'usage_mileage' in car_data:
                    usage_text = str(car_data['usage_mileage']).strip()
                    car_data['mileage_text'] = usage_text  # Keep original text
                    
                    # Extract numeric value from text like "20,000 Km" or "50000 miles"
                    mileage_match = re.search(r'([\d,]+)\s*(?:km|mi|mile|km/|kms)?', usage_text, re.IGNORECASE)
                    if mileage_match:
                        mileage_str = mileage_match.group(1).replace(',', '').strip()
                        try:
                            mileage_value = int(mileage_str)
                            car_data['mileage'] = mileage_value
                            car_data['mileage_km'] = mileage_value  # Store as kilometers
                            logger.debug(f"Extracted mileage: {mileage_value} from '{usage_text}'")
                        except ValueError:
                            car_data['mileage'] = None
                            car_data['mileage_km'] = None
                    else:
                        car_data['mileage'] = None
                        car_data['mileage_km'] = None
                
                # ============ FEATURES/ACCESSORIES ============
                # Find accessories section
                accessories = []
                accessories_section = soup.find('h3', string=lambda text: text and 'Accesorios' in text if text else False)
                
                if accessories_section:
                    container = accessories_section.find_parent()
                    feature_items = container.find_all('li') if container else []
                    
                    for item in feature_items:
                        feature_text = item.text.strip()
                        if feature_text:
                            accessories.append(feature_text)
                
                car_data['accessories'] = accessories
                car_data['accessories_count'] = len(accessories)
                
                # ============ SELLER INFORMATION ============
                seller_section = soup.find('h3', string=lambda text: text and 'Vendedor' in text if text else False)
                if seller_section:
                    seller_container = seller_section.find_parent()
                    seller_name = seller_container.find('h3', recursive=False)
                    if seller_name:
                        car_data['seller_name'] = seller_name.text.strip()
                    
                    # Try to find phone number
                    phone_match = re.search(r'\d{3}-\d{3}-\d{4}', str(seller_container))
                    if phone_match:
                        car_data['seller_phone'] = phone_match.group(0)
                
                # ============ VIEW COUNT ============
                # Extract visit count from page (e.g., "Este anuncio se ha visitado 120 veces")
                views_match = re.search(r'visitado\s+(\d+)\s+veces', str(soup))
                if views_match:
                    car_data['views'] = int(views_match.group(1))
                
                logger.info(f"Successfully scraped: {car_data.get('title', 'Unknown')}")
                # Random delay between 1-3 seconds to avoid detection
                delay = random.uniform(1, 3)
                time.sleep(delay)
                
                return car_data
                
            except Exception as e:
                logger.warning(f"Error scraping car details from {car_url} (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds + random jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to scrape {car_url} after {max_retries} attempts")
        
        return None

    def scrape_multiple_listings(self,
                                models: List[str] = None,
                                vehicle_type: str = "cualquier-tipo",
                                max_pages_per_search: int = 5,
                                max_cars_per_search: int = 40,
                                year_ranges: List[tuple] = None) -> pd.DataFrame:
        """
        Scrape multiple car listings using year ranges (6-year periods).
        
        Scrapes in periods: 2010-2016, 2017-2023, 2024-2027
        For each search, collects up to 40 listings
        
        Args:
            models: List of car models to target (uses self.car_models if None)
            vehicle_type: Type of vehicle to scrape (fallback if model search fails)
            max_pages_per_search: Maximum pages to scrape per model per year range
            max_cars_per_search: Maximum cars to collect per model per year range (default: 40)
            year_ranges: List of year range tuples (default: [(2010, 2016), (2017, 2023), (2024, 2027)])
            
        Returns:
            DataFrame with car data
        """
        if models is None:
            models = self.car_models

        all_cars = []
        total_models = len(models)

        if year_ranges is None:
            year_ranges = [(2010, 2016), (2017, 2023), (2024, 2027)]

        total_ranges = len(year_ranges)

        logger.info(f"🎯 Starting scraping for {total_models} models...")
        logger.info(f"📅 Year ranges: {year_ranges}")
        logger.info(f"📋 Max cars per search: {max_cars_per_search}")

        # Get search URLs for specific models with year ranges
        search_urls = self.get_model_search_urls(models, max_pages_per_search, year_ranges)

        logger.info(f"🔗 Generated {len(search_urls)} search URLs")
        logger.info(f"📊 Total combinations: {total_models} models × {total_ranges} year ranges × {max_pages_per_search} pages = {len(search_urls)} URLs")

        # Track cars per (model, year_range) combination - allows max_cars_per_search per search
        cars_per_model_range = {}

        for url in search_urls:
            try:
                car_urls = self.scrape_listing_page(url)
                
                # Extract year range from URL
                year_from_match = re.search(r'YearFrom=(\d{4})', url)
                year_to_match = re.search(r'YearTo=(\d{4})', url)
                year_range_key = (int(year_from_match.group(1)), int(year_to_match.group(1))) if year_from_match and year_to_match else None

                for car_url in car_urls:
                    car_data = self.scrape_car_details(car_url)
                    if car_data:
                        # Check if this car matches one of our target models
                        car_model = car_data.get('model', '').strip()
                        car_make = car_data.get('make', '').strip()
                        full_model = f"{car_make} {car_model}".strip()

                        # Check if this car matches any of our target models
                        matched_model = None
                        for target_model in models:
                            if target_model.lower() in full_model.lower() or full_model.lower() in target_model.lower():
                                matched_model = target_model
                                break

                        # Create key for (model, year_range) combination
                        model_range_key = (matched_model, year_range_key)
                        
                        if model_range_key not in cars_per_model_range:
                            cars_per_model_range[model_range_key] = 0

                        if matched_model and cars_per_model_range[model_range_key] < max_cars_per_search:
                            all_cars.append(car_data)
                            cars_per_model_range[model_range_key] += 1
                            logger.info(f"✅ Added {full_model} ({car_data.get('year', 'N/A')}) - Total: {len(all_cars)}")

                            # Stop if we've collected enough for this model-range combination
                            if cars_per_model_range[model_range_key] >= max_cars_per_search:
                                logger.info(f"🎯 Reached limit for {matched_model} ({year_range_key}) ({max_cars_per_search} cars)")

            except Exception as e:
                logger.warning(f"Error processing URL {url}: {e}")
                continue

        # Convert to DataFrame
        if all_cars:
            df = pd.DataFrame(all_cars)
            logger.info(f"🎉 Total cars collected: {len(df)}")
            logger.info(f"📊 Cars per model-range combination: {cars_per_model_range}")
            logger.info(f"📋 Available columns: {list(df.columns)[:10]}...")

            # Add metadata
            df['scraped_at'] = pd.Timestamp.now()
            df['data_source'] = 'supercarros.com'
            df['year_ranges'] = str(year_ranges)

            return df
        else:
            logger.warning("❌ No cars were scraped!")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, filename: str = "supercarros_listings.csv"):
        """
        Save scraped data to CSV file.
        
        Args:
            df: DataFrame with car data
            filename: Output filename
        """
        if df.empty:
            logger.warning("DataFrame is empty. Nothing to save.")
            return
        
        filepath = f"data/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved to {filepath}")
        logger.info(f"Saved {len(df)} records")


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    """
    Example usage of the SuperCarrosScraper
    """

    scraper = SuperCarrosScraper()

    # 🎯 TARGETED SCRAPING: Scrape specific popular models
    print("🚗 Starting SuperCarros.com targeted scraper...")
    print("=" * 60)
    print(f"🎯 Targeting {len(scraper.car_models)} popular car models")
    print("=" * 60)

    # Scrape top 10 models with 40 cars each (for testing)
    test_models = scraper.car_models[:10]  # First 10 models for quick test

    cars_df = scraper.scrape_multiple_listings(
        models=scraper.car_models,                                                      # Target specific models
        max_pages_per_search=5,                                                  # 3 pages per search
        max_cars_per_search=40,                                                  # 40 cars per year-range search
        year_ranges=[(2010, 2016), (2017, 2023), (2024, 2027)]                  # 3 year-ranges
    )

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)

    if not cars_df.empty:
        print(f"\n📊 Total Records: {len(cars_df)}")
        print(f"\n📋 Columns Available:")
        for i, col in enumerate(cars_df.columns, 1):
            print(f"  {i:2}. {col}")

        print(f"\n🔍 Sample Data:")
        sample_cols = ['title', 'amount', 'currency', 'year', 'make', 'model']
        available_cols = [col for col in sample_cols if col in cars_df.columns]
        if available_cols:
            print(cars_df[available_cols].head())

        print(f"\n💾 Saving to CSV...")
        scraper.save_to_csv(cars_df, "supercarros_listings.csv")
    else:
        print("❌ No data collected!")

    print("\n" + "=" * 60)
    print("FULL SCRAPE EXAMPLE (commented out - use carefully)")
    print("=" * 60)
    print("""
    # For full scrape of all models (60 models, ~1440+ cars):
    # cars_df = scraper.scrape_multiple_listings(
    #     models=scraper.car_models,                                             # All 60 models
    #     max_pages_per_search=5,                                                # 5 pages per search
    #     max_cars_per_search=40,                                                # 40 cars per year-range search
    #     year_ranges=[(2010, 2016), (2017, 2023), (2024, 2027)]                # 3 year-ranges (60 × 3 × 40 = 7200 possible)
    # )
    """)
