"""
Car Price Intelligence Platform - Data Collection Module
This module handles data collection from various sources for car price prediction.
"""

import requests
import pandas as pd
import time
import logging
from typing import List, Dict, Optional
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CarDataCollector:
    """
    A class to collect car data from various APIs and sources.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the data collector.

        Args:
            api_key: API key for MarketCheck API (if available)
        """
        self.api_key = api_key
        self.base_url = "https://api.marketcheck.com/v2"
        self.session = requests.Session()

        # Set headers for API requests
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    def collect_marketcheck_data(self,
                                make: str = "",
                                model: str = "",
                                year: Optional[int] = None,
                                rows: int = 50,
                                api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Collect car data from MarketCheck API.

        Args:
            make: Car make (e.g., 'Toyota')
            model: Car model (e.g., 'Camry')
            year: Car year
            rows: Number of records to fetch (max 50 per request)
            api_key: MarketCheck API key

        Returns:
            DataFrame with car listings data
        """
        if not self.api_key and not api_key:
            logger.warning("No API key provided. MarketCheck API requires authentication.")
            return pd.DataFrame()

        params = {
            'rows': min(rows, 50),  # API limit is 50 per request
            'start': 0
        }

        if api_key:
            params['api_key'] = api_key
        elif self.api_key:
            params['api_key'] = self.api_key

        if make:
            params['make'] = make
        if model:
            params['model'] = model
        if year:
            params['year'] = year

        endpoint = f"{self.base_url}/search/car/active"

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()

            data = response.json()
            listings = data.get('listings', [])

            if not listings:
                logger.info("No listings found for the specified criteria.")
                return pd.DataFrame()

            # Extract relevant fields
            processed_data = []
            for listing in listings:
                processed_listing = {
                    'vin': listing.get('vin'),
                    'make': listing.get('make'),
                    'model': listing.get('model'),
                    'year': listing.get('year'),
                    'trim': listing.get('trim'),
                    'price': listing.get('price'),
                    'mileage': listing.get('miles'),
                    'city': listing.get('city'),
                    'state': listing.get('state'),
                    'zip': listing.get('zip'),
                    'dealer_name': listing.get('dealer_name'),
                    'listing_date': listing.get('ref_dt'),
                    'engine': listing.get('engine'),
                    'transmission': listing.get('transmission'),
                    'drive_train': listing.get('drivetrain'),
                    'fuel_type': listing.get('fuel_type'),
                    'exterior_color': listing.get('exterior_color'),
                    'interior_color': listing.get('interior_color'),
                    'body_type': listing.get('body_type'),
                    'doors': listing.get('doors'),
                    'cylinders': listing.get('cylinders'),
                    'horsepower': listing.get('horsepower'),
                    'torque': listing.get('torque'),
                    'highway_mpg': listing.get('highway_mpg'),
                    'city_mpg': listing.get('city_mpg'),
                    'combined_mpg': listing.get('combined_mpg'),
                    'msrp': listing.get('msrp'),
                    'invoice': listing.get('invoice'),
                    'scraped_at': datetime.now().isoformat()
                }
                processed_data.append(processed_listing)

            df = pd.DataFrame(processed_data)
            logger.info(f"Collected {len(df)} car listings from MarketCheck API.")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error collecting data from MarketCheck API: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return pd.DataFrame()

    def collect_multiple_pages(self,
                              makes: List[str] = None,
                              models: Optional[List[str]] = None,
                              years: Optional[List[int]] = None,
                              total_rows: int = 10000,
                              api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Collect car data from US market by city, model, and year range.
        Optimized for maximum data collection efficiency.

        Args:
            makes: Ignored (kept for backward compatibility)
            models: List of popular models to search for (e.g., ["Toyota Camry", "Honda Civic"])
            years: List of year ranges (e.g., ["2010-2015", "2016-2018"])
            total_rows: Target number of rows to collect
            api_key: MarketCheck API key

        Returns:
            DataFrame with car listings data
        """
        if not api_key and not self.api_key:
            logger.warning("No API key provided. MarketCheck API requires authentication.")
            return pd.DataFrame()

        key = api_key or self.api_key
        
        # Default US cities with high auto volume
        cities = [
            "Los Angeles", "New York", "Chicago", "Houston", "Phoenix",
            "Dallas", "Miami", "Atlanta", "San Francisco", "Seattle"
        ]

        # Default popular models if not specified
        if models is None:
            models = [
    # Sedans
    "Toyota Camry", "Honda Civic", "Honda Accord", "Toyota Corolla",
    "Nissan Altima", "Hyundai Elantra", "Chevrolet Malibu",
    
    # SUVs
    "Toyota RAV4", "Honda CR-V", "Nissan Rogue", "Ford Escape",
    "Chevrolet Equinox", "Hyundai Tucson", "Jeep Grand Cherokee",
    
    # Pickups
    "Ford F-150", "Chevrolet Silverado", "Ram 1500", "Toyota Tacoma",
    
    # Luxury
    "BMW 3 Series", "BMW 5 Series", "Mercedes-Benz C-Class",
    "Mercedes-Benz E-Class", "Audi A4", "Audi Q5",
    
    # Otros populares
    "Kia Forte", "Kia Sportage", "Subaru Outback", "Subaru Forester",
    "Volkswagen Jetta", "Mazda CX-5"
]

        # Default year ranges if not specified
        if years is None:
            years = ["2010-2015", "2016-2018", "2019-2023"]

        all_data = []
        endpoint = f"{self.base_url}/search/car/active"

        logger.info(f"Starting data collection from {len(cities)} cities, {len(models)} models, {len(years)} year ranges")

        # 🔁 Collection loop
        for city in cities:
            for model in models:
                for year_range in years:
                    logger.info(f"Collecting: {city} | {model} | {year_range}")

                    for page in range(1, 50):  # Max 50 pages per combination
                        params = {
                            "api_key": key,
                            "city": city,
                            "rows": 100,
                            "page": page,
                            "year_range": year_range,
                            "car_type": "used",
                            "search": model  # Key filter for model matching
                        }

                        try:
                            response = requests.get(endpoint, params=params)

                            if response.status_code != 200:
                                logger.info(f"No more data: {city} | {model} | {year_range} at page {page}")
                                break

                            data = response.json()
                            listings = data.get("listings", [])

                            if not listings:
                                break

                            all_data.extend(listings)
                            logger.info(f"  Page {page}: {len(listings)} listings (Total: {len(all_data)})")

                            # Rate limiting
                            time.sleep(0.5)

                            # Stop if we've collected enough data
                            if len(all_data) >= total_rows:
                                logger.info(f"Target of {total_rows} rows reached!")
                                break

                        except Exception as e:
                            logger.error(f"Error collecting {city} | {model} | {year_range} page {page}: {e}")
                            break

                    if len(all_data) >= total_rows:
                        break

                if len(all_data) >= total_rows:
                    break

            if len(all_data) >= total_rows:
                break

        if not all_data:
            logger.warning("No data collected from US market")
            return pd.DataFrame()

        # 📊 Convert to DataFrame
        df = pd.json_normalize(all_data)
        logger.info(f"Raw data collected: {len(df)} records")
        logger.info(f"Available columns: {list(df.columns)[:20]}")

        # 🧹 Select relevant columns (use intersection of available and desired)
        desired_cols = ["make", "model", "year", "price", "miles", "city", "state", "vin"]
        available_cols = [col for col in desired_cols if col in df.columns]
        
        if not available_cols:
            logger.error(f"No relevant columns found. Available: {list(df.columns)[:20]}")
            return pd.DataFrame()
            
        df = df[available_cols]
        logger.info(f"Using columns: {available_cols}")

        # ❌ Remove duplicates
        if 'vin' in df.columns:
            df = df.drop_duplicates(subset="vin")
            logger.info(f"After removing duplicates: {len(df)} records")

        # ❌ Remove nulls (only for columns that exist)
        cols_to_check = [col for col in ["price", "miles", "year"] if col in df.columns]
        if cols_to_check:
            df = df.dropna(subset=cols_to_check)
            logger.info(f"After removing nulls: {len(df)} records")

        # 🧼 Clean numeric columns
        for col in ["price", "miles"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        numeric_cols_to_check = [col for col in ["price", "miles"] if col in df.columns]
        if numeric_cols_to_check:
            df = df.dropna(subset=numeric_cols_to_check)

        # 🧠 Feature Engineering (only if required columns exist)
        if "year" in df.columns:
            current_year = 2026
            df["car_age"] = current_year - df["year"]
            df = df[df["car_age"] > 0]  # Remove invalid ages
            logger.info(f"After age validation: {len(df)} records")

        if "miles" in df.columns and "car_age" in df.columns:
            df["miles_per_year"] = df["miles"] / df["car_age"]

        # 📊 Market features (only if columns exist)
        if "model" in df.columns and "price" in df.columns:
            df["avg_price_model"] = df.groupby("model")["price"].transform("mean")
            df["price_vs_market"] = df["price"] - df["avg_price_model"]

        # 🔥 Remove price outliers (only if price column exists)
        if "price" in df.columns:
            df = df[(df["price"] > 2000) & (df["price"] < 100000)]
            logger.info(f"After removing price outliers: {len(df)} records")

        # 🧹 Final cleanup
        df = df.dropna()
        logger.info(f"Final clean dataset: {len(df)} records")

        return df

    def save_data(self, df: pd.DataFrame, filename: str, path: str = "data/"):
        """
        Save collected data to CSV file.

        Args:
            df: DataFrame to save
            filename: Name of the file
            path: Path to save the file
        """
        if df.empty:
            logger.warning("DataFrame is empty. Nothing to save.")
            return

        filepath = f"{path.rstrip('/')}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")

    def collect_mexico_market_data(self,
                                  api_key: Optional[str] = None,
                                  cities: List[str] = None,
                                  year_ranges: List[str] = None,
                                  max_pages: int = 50) -> pd.DataFrame:
        """
        Collect car data from Mexican market using MarketCheck API.
        Based on user's existing script for Mexican cities.

        Args:
            api_key: MarketCheck API key
            cities: List of Mexican cities to collect data from
            year_ranges: List of year ranges (e.g., ["2010-2015", "2016-2018"])
            max_pages: Maximum pages to collect per city/year combination

        Returns:
            DataFrame with Mexican car market data
        """
        if not self.api_key and not api_key:
            logger.warning("No API key provided. MarketCheck API requires authentication.")
            return pd.DataFrame()

        # Default Mexican cities and year ranges if not provided
        if cities is None:
            cities = ["Mexico City", "Guadalajara", "Monterrey"]

        if year_ranges is None:
            year_ranges = ["2010-2015", "2016-2018", "2019-2023"]

        all_data = []
        total_collected = 0

        # Collection loop
        for city in cities:
            for year_range in year_ranges:
                logger.info(f"Collecting data for: {city} | {year_range}")

                for page in range(1, max_pages + 1):
                    params = {
                        "api_key": api_key or self.api_key,
                        "city": city,
                        "rows": 50,
                        "page": page,
                        "year_range": year_range
                    }

                    try:
                        # Use the correct API endpoint
                        endpoint = f"{self.base_url}/search/car/active"
                        response = self.session.get(endpoint, params=params)

                        if response.status_code != 200:
                            logger.warning(f"Error on page {page} for {city} | {year_range}: {response.status_code}")
                            break

                        data = response.json()
                        listings = data.get("listings", [])

                        if not listings:
                            logger.info(f"No more data for {city} | {year_range} at page {page}")
                            break

                        all_data.extend(listings)
                        total_collected += len(listings)

                        logger.info(f"Page {page} OK - Collected {len(listings)} listings (Total: {total_collected})")
                        time.sleep(0.5)  # Rate limiting

                    except Exception as e:
                        logger.error(f"Error collecting data: {e}")
                        break

        if not all_data:
            logger.warning("No data collected from Mexican market")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.json_normalize(all_data)
        logger.info(f"Raw data collected: {len(df)} records")

        # Select relevant columns (matching your existing script)
        relevant_cols = [
            "make", "model", "year", "price", "miles", "city", "state", "vin",
            "trim", "exterior_color", "condition", "engine", "transmission",
            "drive_train", "fuel_type", "body_type", "doors", "cylinders",
            "horsepower", "highway_mpg", "city_mpg"
        ]

        # Keep only columns that exist
        available_cols = [col for col in relevant_cols if col in df.columns]
        df = df[available_cols]

        # Remove duplicates
        if 'vin' in df.columns:
            df = df.drop_duplicates(subset="vin")
            logger.info(f"After removing duplicates: {len(df)} records")

        # Remove nulls in critical columns
        critical_cols = ["price", "miles", "year"]
        available_critical = [col for col in critical_cols if col in df.columns]
        if available_critical:
            df = df.dropna(subset=available_critical)
            logger.info(f"After removing nulls in critical columns: {len(df)} records")

        # Clean price data
        if 'price' in df.columns:
            def clean_price(price):
                try:
                    return int(price)
                except:
                    return None

            df["price"] = df["price"].apply(clean_price)
            df = df.dropna(subset=["price"])

        # Basic feature engineering
        current_year = 2024  # Using 2024 as current year
        if 'year' in df.columns:
            df["car_age"] = current_year - df["year"]

        if 'miles' in df.columns and 'car_age' in df.columns:
            df["miles_per_year"] = df["miles"] / df["car_age"].clip(lower=1)

        # Market features
        if 'model' in df.columns and 'price' in df.columns:
            df["avg_price_model"] = df.groupby("model")["price"].transform("mean")
            df["price_vs_market"] = df["price"] - df["avg_price_model"]

        # Final cleanup
        df = df.dropna()
        logger.info(f"Final clean dataset: {len(df)} records")

        return df

    def collect_mexico_market_scraping(self,
                                       cities: List[str] = None,
                                       year_ranges: List[str] = None,
                                       max_pages: int = 10) -> pd.DataFrame:
        """
        Collect car data from MercadoLibre Mexico using web scraping.
        Scrapes real Mexican car listings and returns with same format as API.

        Args:
            cities: List of Mexican cities (for reference, MercadoLibre uses pagination)
            year_ranges: List of year ranges (for reference)
            max_pages: Maximum pages to scrape

        Returns:
            DataFrame with MercadoLibre car market data
        """
        from bs4 import BeautifulSoup
        import requests

        logger.warning("Web scraping MercadoLibre Mexico - respecting website terms of service")

        BASE_URL = "https://autos.mercadolibre.com.mx/"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        all_cars = []
        total_collected = 0

        try:
            # Pagination using offset (MercadoLibre uses 48 results per page)
            for offset in range(0, max_pages * 48, 48):
                url = f"{BASE_URL}_Desde_{offset}"

                logger.info(f"Scraping MercadoLibre page offset: {offset}")

                try:
                    response = requests.get(url, headers=headers, timeout=10)

                    if response.status_code != 200:
                        logger.warning(f"HTTP {response.status_code} at offset {offset}")
                        break

                    soup = BeautifulSoup(response.text, "html.parser")

                    listings = soup.find_all("li", class_="ui-search-layout__item")

                    if not listings:
                        logger.info(f"No listings found at offset {offset}")
                        break

                    page_cars = []
                    for car_element in listings:
                        try:
                            # Extract title
                            title_elem = car_element.find("h2")
                            title = title_elem.text.strip() if title_elem else "Unknown"

                            # Extract price
                            price_elem = car_element.find("span", class_="andes-money-amount__fraction")
                            price = None
                            if price_elem:
                                try:
                                    price = int(price_elem.text.replace(",", "").replace(".", ""))
                                except:
                                    price = None

                            # Extract details (year, km)
                            details = car_element.find_all("li", class_="ui-search-card-attributes__attribute")

                            year = None
                            mileage = None

                            for detail in details:
                                detail_text = detail.text.strip().lower()

                                # Extract km (mileage)
                                if "km" in detail_text:
                                    try:
                                        km_str = detail_text.replace("km", "").replace(",", "").strip()
                                        mileage = int(float(km_str))
                                    except:
                                        pass

                                # Extract year (4-digit number)
                                parts = detail.text.strip().split()
                                for part in parts:
                                    if part.isdigit() and len(part) == 4:
                                        try:
                                            y = int(part)
                                            if 1990 <= y <= 2024:
                                                year = y
                                        except:
                                            pass

                            # Extract location
                            location_elem = car_element.find("span", class_="ui-search-item__group__element")
                            location = location_elem.text.strip() if location_elem else "Mexico"

                            # Parse title to extract make and model
                            make, model, trim = self._parse_car_title(title)

                            # Only include if we have essential data
                            if price and year and make:
                                car_data = {
                                    "title": title,
                                    "make": make,
                                    "model": model,
                                    "trim": trim,
                                    "year": year,
                                    "price": price,
                                    "miles": mileage if mileage else 0,
                                    "city": location.split(",")[0] if "," in location else location,
                                    "state": "MX",
                                    "vin": f"ML{offset}{len(page_cars)}",
                                    "exterior_color": "Unknown",
                                    "condition": "Good",
                                    "engine": "Unknown",
                                    "transmission": "Unknown",
                                    "drive_train": "Unknown",
                                    "fuel_type": "Unknown",
                                    "body_type": "Unknown",
                                    "doors": None,
                                    "cylinders": None,
                                    "horsepower": None,
                                    "highway_mpg": None,
                                    "city_mpg": None,
                                    "scraped_at": pd.Timestamp.now().isoformat(),
                                    "source": "mercadolibre"
                                }

                                page_cars.append(car_data)
                                total_collected += 1

                        except Exception as e:
                            logger.debug(f"Error parsing listing: {e}")
                            continue

                    logger.info(f"Offset {offset}: Collected {len(page_cars)} listings (Total: {total_collected})")
                    all_cars.extend(page_cars)

                    # Delay between requests
                    time.sleep(1)

                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error at offset {offset}: {e}")
                    break

        except Exception as e:
            logger.error(f"Error in scraping: {e}")

        if not all_cars:
            logger.warning("No data collected from MercadoLibre scraping")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_cars)
        logger.info(f"Raw scraped data: {len(df)} records from MercadoLibre")

        # Apply same cleaning as API method
        df = self._clean_scraped_data(df)

        return df

    def _parse_car_title(self, title: str) -> tuple:
        """
        Parse MercadoLibre car title to extract make, model, and trim.
        Typical format: "Make Model Trim Year Km"
        
        Args:
            title: Full car title from MercadoLibre
            
        Returns:
            Tuple of (make, model, trim)
        """
        common_makes = {
            'toyota': 'Toyota', 'honda': 'Honda', 'ford': 'Ford', 'chevrolet': 'Chevrolet',
            'bmw': 'BMW', 'mercedes': 'Mercedes-Benz', 'audi': 'Audi', 'nissan': 'Nissan',
            'hyundai': 'Hyundai', 'kia': 'Kia', 'volkswagen': 'Volkswagen', 'mazda': 'Mazda',
            'suzuki': 'Suzuki', 'mitsubishi': 'Mitsubishi', 'isuzu': 'Isuzu', 'ram': 'RAM',
            'jeep': 'Jeep', 'chrysler': 'Chrysler', 'dodge': 'Dodge', 'lexus': 'Lexus',
            'infiniti': 'Infiniti', 'acura': 'Acura', 'cadillac': 'Cadillac', 'lincoln': 'Lincoln'
        }

        title_lower = title.lower().strip()
        words = title_lower.split()

        make = "Unknown"
        model = "Unknown"
        trim = "Unknown"

        # Find make
        for i, word in enumerate(words):
            clean_word = word.replace(",", "").replace(".", "")
            if clean_word in common_makes:
                make = common_makes[clean_word]
                # Model is typically the next word
                if i + 1 < len(words):
                    model = words[i + 1].replace(",", "").strip()
                # Trim is typically the word after model (if not a year)
                if i + 2 < len(words):
                    potential_trim = words[i + 2].replace(",", "").strip()
                    if not potential_trim.isdigit():
                        trim = potential_trim
                break

        return make, model, trim

    def _clean_scraped_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same cleaning logic as the API method to scraped data.
        """
        # Select relevant columns (matching API method)
        relevant_cols = [
            "make", "model", "year", "price", "miles", "city", "state", "vin",
            "trim", "exterior_color", "condition", "engine", "transmission",
            "drive_train", "fuel_type", "body_type", "doors", "cylinders",
            "horsepower", "highway_mpg", "city_mpg"
        ]

        # Keep only columns that exist
        available_cols = [col for col in relevant_cols if col in df.columns]
        df = df[available_cols]

        # Remove duplicates
        if 'vin' in df.columns:
            df = df.drop_duplicates(subset="vin")
            logger.info(f"After removing duplicates: {len(df)} records")

        # Remove nulls in critical columns
        critical_cols = ["price", "miles", "year"]
        available_critical = [col for col in critical_cols if col in df.columns]
        if available_critical:
            df = df.dropna(subset=available_critical)
            logger.info(f"After removing nulls in critical columns: {len(df)} records")

        # Clean price data
        if 'price' in df.columns:
            def clean_price(price):
                try:
                    return int(price)
                except:
                    return None

            df["price"] = df["price"].apply(clean_price)
            df = df.dropna(subset=["price"])

        # Basic feature engineering (same as API method)
        current_year = 2024
        if 'year' in df.columns:
            df["car_age"] = current_year - df["year"]

        if 'miles' in df.columns and 'car_age' in df.columns:
            df["miles_per_year"] = df["miles"] / df["car_age"].clip(lower=1)

        # Market features
        if 'model' in df.columns and 'price' in df.columns:
            df["avg_price_model"] = df.groupby("model")["price"].transform("mean")
            df["price_vs_market"] = df["price"] - df["avg_price_model"]

        # Final cleanup
        df = df.dropna()
        logger.info(f"Final scraped dataset: {len(df)} records")

        return df

    def collect_mexico_market_combined(self,
                                      api_key: Optional[str] = None,
                                      cities: List[str] = None,
                                      year_ranges: List[str] = None,
                                      max_pages_api: int = 50,
                                      max_pages_scraping: int = 10) -> pd.DataFrame:
        """
        Collect data using both API and scraping methods, then combine results.

        Args:
            api_key: MarketCheck API key
            cities: Cities to collect from
            year_ranges: Year ranges to collect
            max_pages_api: Max pages for API collection
            max_pages_scraping: Max pages for scraping

        Returns:
            Combined DataFrame from both sources
        """
        logger.info("Starting combined data collection (API + Scraping)")

        # Collect from API
        api_data = pd.DataFrame()
        try:
            logger.info("Collecting data via MarketCheck API...")
            api_data = self.collect_mexico_market_data(
                api_key=api_key,
                cities=cities,
                year_ranges=year_ranges,
                max_pages=max_pages_api
            )
            if not api_data.empty:
                api_data['data_source'] = 'api'
                logger.info(f"API collection successful: {len(api_data)} records")
        except Exception as e:
            logger.error(f"API collection failed: {e}")

        # Collect from scraping
        scraping_data = pd.DataFrame()
        try:
            logger.info("Collecting data via web scraping...")
            scraping_data = self.collect_mexico_market_scraping(
                cities=cities,
                year_ranges=year_ranges,
                max_pages=max_pages_scraping
            )
            if not scraping_data.empty:
                scraping_data['data_source'] = 'scraping'
                logger.info(f"Scraping collection successful: {len(scraping_data)} records")
        except Exception as e:
            logger.error(f"Scraping collection failed: {e}")

        # Combine datasets
        combined_data = pd.DataFrame()

        if not api_data.empty and not scraping_data.empty:
            # Both sources have data
            combined_data = pd.concat([api_data, scraping_data], ignore_index=True)
            logger.info(f"Combined data from both sources: {len(combined_data)} records")
            logger.info(f"API: {len(api_data)}, Scraping: {len(scraping_data)}")

        elif not api_data.empty:
            # Only API data
            combined_data = api_data.copy()
            logger.info(f"Using only API data: {len(combined_data)} records")

        elif not scraping_data.empty:
            # Only scraping data
            combined_data = scraping_data.copy()
            logger.info(f"Using only scraping data: {len(combined_data)} records")

        else:
            logger.error("No data collected from either source")
            return pd.DataFrame()

        # Remove duplicates across sources (same VIN)
        if 'vin' in combined_data.columns:
            before_dedup = len(combined_data)
            combined_data = combined_data.drop_duplicates(subset='vin')
            after_dedup = len(combined_data)
            duplicates_removed = before_dedup - after_dedup
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate VINs across sources")

        logger.info(f"Final combined dataset: {len(combined_data)} records")
        return combined_data

    def collect_us_states_data(self,
                              states: Optional[List[str]] = None,
                              api_key: Optional[str] = None,
                              max_pages: int = 50) -> pd.DataFrame:
        """
        Collect car data from US states using MarketCheck API.

        Args:
            states: List of US state codes (e.g., ['CA', 'TX', 'NY']).
                    If None, collects from major states
            api_key: MarketCheck API key
            max_pages: Maximum pages to collect per state

        Returns:
            DataFrame with car listings from specified states
        """
        if api_key:
            self.api_key = api_key

        if not self.api_key:
            logger.warning("No API key provided. MarketCheck API requires authentication.")
            return pd.DataFrame()

        # Default to major US states if none specified
        if states is None:
            states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']

        logger.info(f"Collecting data from {len(states)} US states: {states}")

        all_data = []
        endpoint = f"{self.base_url}/search/car/active"

        for state in states:
            logger.info(f"Collecting data from state: {state}")
            state_data = []

            for page in range(max_pages):
                try:
                    params = {
                        'api_key': api_key or self.api_key,
                        'state': state,
                        'rows': 50,
                        'page': page
                    }

                    response = self.session.get(endpoint, params=params)
                    response.raise_for_status()

                    data = response.json()
                    listings = data.get('listings', [])

                    if not listings:
                        logger.info(f"No more listings for {state} at page {page + 1}")
                        break

                    # Extract relevant fields
                    for listing in listings:
                        processed_listing = {
                            'vin': listing.get('vin'),
                            'make': listing.get('make'),
                            'model': listing.get('model'),
                            'year': listing.get('year'),
                            'price': listing.get('price'),
                            'miles': listing.get('miles'),
                            'city': listing.get('heading', '').split(',')[0] if listing.get('heading') else '',
                            'state': state,
                            'condition': 'Good',  # Default
                            'exterior_color': listing.get('color', 'Unknown'),
                            'body_type': listing.get('body_type', 'Unknown'),
                            'transmission': listing.get('transmission', 'Unknown'),
                            'engine': listing.get('engine', 'Unknown'),
                            'drive_train': listing.get('drivetrain', 'Unknown'),
                            'fuel_type': listing.get('fuel_type', 'Unknown'),
                            'horsepower': None,
                            'city_mpg': None,
                            'highway_mpg': None,
                            'cylinders': None,
                            'doors': None,
                            'trim': listing.get('trim', ''),
                            'data_source': 'marketcheck_api'
                        }
                        state_data.append(processed_listing)

                    logger.info(f"State {state} - Page {page + 1}: {len(listings)} listings")

                    # Rate limiting
                    time.sleep(0.5)

                except requests.exceptions.RequestException as e:
                    logger.error(f"Error fetching data for {state} page {page + 1}: {e}")
                    break

            if state_data:
                all_data.extend(state_data)
                logger.info(f"Total from {state}: {len(state_data)} listings")

        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"Total collected from US states: {len(df)} records")
            return df
        else:
            logger.warning("No data collected from US states")
            return pd.DataFrame()

if __name__ == "__main__":
    # Initialize collector (replace with actual API key)
    collector = CarDataCollector(api_key="YOUR_API_KEY_HERE")

    # Define popular car makes to collect data for
    popular_makes = [
        "Toyota", "Honda", "Ford", "Chevrolet", "Nissan",
        "BMW", "Mercedes-Benz", "Audi", "Volkswagen", "Hyundai"
    ]

    # Collect data
    car_data = collector.collect_multiple_pages(
        makes=popular_makes,
        total_rows=100  # Small sample for testing
    )

    # Save data
    if not car_data.empty:
        collector.save_data(car_data, "car_listings_sample.csv")