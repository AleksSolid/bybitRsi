import os
import re
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict
from urllib.parse import urlencode, urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


@dataclass
class Listing:
    title: str
    price_rub: Optional[int]
    year: Optional[int]
    mileage_km: Optional[int]
    city: Optional[str]
    link: str
    image_url: Optional[str]
    seller_type: Optional[str]


class AutoRuScraper:
    BASE_URL = "https://auto.ru/"

    def __init__(
        self,
        delay_seconds: float = None,
        user_agent: Optional[str] = None,
        timeout_seconds: int = None,
        respect_robots: Optional[bool] = None,
    ) -> None:
        self.delay_seconds = (
            float(delay_seconds)
            if delay_seconds is not None
            else float(os.getenv("AUTO_RU_REQUEST_DELAY_SECONDS", "3.0"))
        )
        self.timeout_seconds = (
            int(timeout_seconds)
            if timeout_seconds is not None
            else int(os.getenv("AUTO_RU_TIMEOUT_SECONDS", "20"))
        )
        self.respect_robots = (
            bool(int(respect_robots))
            if isinstance(respect_robots, (int, float))
            else (os.getenv("AUTO_RU_RESPECT_ROBOTS", "1") not in {"0", "false", "False"})
        )

        default_ua = (
            os.getenv("AUTO_RU_USER_AGENT")
            or user_agent
            or (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": default_ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "ru,en;q=0.9",
                "Connection": "keep-alive",
            }
        )

        self._last_request_ts = 0.0
        self._robots = None  # type: Optional[RobotFileParser]
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _init_robots(self) -> None:
        if self._robots is not None or not self.respect_robots:
            return
        robots_url = urljoin(self.BASE_URL, "robots.txt")
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
            self._robots = parser
        except Exception as exc:
            # If robots cannot be read, log and proceed cautiously
            self.logger.warning("Failed to read robots.txt: %s", exc)
            self._robots = None

    def _is_allowed(self, target_url: str) -> bool:
        if not self.respect_robots:
            return True
        self._init_robots()
        if self._robots is None:
            # Be conservative if robots are unavailable
            return False
        return self._robots.can_fetch(self.session.headers.get("User-Agent", "*"), target_url)

    def _respect_rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_ts
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _get(self, url: str) -> requests.Response:
        if not self._is_allowed(url):
            raise PermissionError(f"robots.txt disallows fetching: {url}")
        self._respect_rate_limit()
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response

    @staticmethod
    def _safe_int(text: Optional[str]) -> Optional[int]:
        if not text:
            return None
        digits = re.sub(r"[^0-9]", "", text)
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None

    @staticmethod
    def _build_search_path(city: Optional[str], make: Optional[str], model: Optional[str]) -> str:
        # Examples:
        #   /moskva/cars/audi/a4/all/
        #   /moskva/cars/audi/all/
        #   /moskva/cars/all/
        parts = []
        if city:
            parts.append(city.strip("/"))
        parts.append("cars")
        if make:
            parts.append(make.strip("/"))
            if model:
                parts.append(model.strip("/"))
        parts.append("all")
        return "/".join(parts) + "/"

    def build_search_url(
        self,
        city: Optional[str] = None,
        make: Optional[str] = None,
        model: Optional[str] = None,
        price_from: Optional[int] = None,
        price_to: Optional[int] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        page: int = 1,
        additional_params: Optional[Dict[str, str]] = None,
    ) -> str:
        path = self._build_search_path(city, make, model)
        params: Dict[str, str] = {
            "output_type": "list",
            "page": str(page),
        }
        if price_from:
            params["price_from"] = str(price_from)
        if price_to:
            params["price_to"] = str(price_to)
        if year_from:
            params["year_from"] = str(year_from)
        if year_to:
            params["year_to"] = str(year_to)
        if additional_params:
            params.update({k: str(v) for k, v in additional_params.items()})

        query = urlencode(params)
        return urljoin(self.BASE_URL, f"{path}?{query}")

    def _parse_listing_card(self, node) -> Optional[Listing]:
        # Try multiple selector strategies to be resilient to markup changes
        # Link
        link_tag = (
            node.select_one("a.ListingItemTitle__link")
            or node.select_one("a.ListingItemTitle")
            or node.select_one("a.Link")
        )
        href = link_tag["href"].strip() if link_tag and link_tag.has_attr("href") else None
        if not href:
            return None

        # Title
        title = (
            (link_tag.get_text(strip=True) if link_tag else None)
            or (node.select_one("h3") or node.select_one("h2") or node.select_one("h1"))
        )
        if hasattr(title, "get_text"):
            title = title.get_text(strip=True)
        title = title or "Объявление"

        # Price
        price_tag = (
            node.select_one("div.ListingItemPrice__price")
            or node.select_one("span.ListingItemPrice__content")
            or node.find(text=re.compile(r"\d[\d\s\u00A0]*\s*₽"))
        )
        price_text = (
            price_tag.get_text(strip=True) if hasattr(price_tag, "get_text") else str(price_tag or "")
        )
        price_rub = self._safe_int(price_text)

        # Year (often encoded in the title)
        year = None
        m_year = re.search(r"(19|20)\d{2}", title)
        if m_year:
            year = int(m_year.group(0))

        # Mileage
        mileage_tag = (
            node.select_one("div.ListingItem__kmAge")
            or node.select_one("div.ListingItemTechSummaryDesktop__cell")
            or node.find(text=re.compile(r"\d[\d\s\u00A0]*\s*км"))
        )
        mileage_text = (
            mileage_tag.get_text(strip=True) if hasattr(mileage_tag, "get_text") else str(mileage_tag or "")
        )
        mileage_km = self._safe_int(mileage_text)

        # City
        city_tag = node.select_one("span.ListingItem__place") or node.find(
            "span", attrs={"data-mark": "DeliveryBadge__region"}
        )
        city = city_tag.get_text(strip=True) if city_tag else None

        # Image
        image_tag = (
            node.select_one("img.ListingItemThumb__img")
            or node.select_one("img")
        )
        image_url = image_tag["src"].strip() if image_tag and image_tag.has_attr("src") else None

        # Seller type
        seller_tag = node.find(text=re.compile("Частное лицо|Дилер", re.IGNORECASE))
        seller_type = None
        if seller_tag:
            text = seller_tag if isinstance(seller_tag, str) else seller_tag.get_text(strip=True)
            seller_type = "Дилер" if "дилер" in text.lower() else "Частное лицо"

        return Listing(
            title=title,
            price_rub=price_rub,
            year=year,
            mileage_km=mileage_km,
            city=city,
            link=href,
            image_url=image_url,
            seller_type=seller_type,
        )

    def parse_listings(self, html: str) -> List[Listing]:
        soup = BeautifulSoup(html, "lxml")
        candidates = []
        # Primary selectors
        candidates.extend(soup.select("div.ListingItem"))
        # Fallbacks
        candidates.extend(soup.select("article"))
        candidates.extend(soup.select("div[data-bem]"))

        results: List[Listing] = []
        seen_links = set()
        for node in candidates:
            listing = self._parse_listing_card(node)
            if listing and listing.link not in seen_links:
                results.append(listing)
                seen_links.add(listing.link)
        return results

    def search(
        self,
        city: Optional[str] = None,
        make: Optional[str] = None,
        model: Optional[str] = None,
        price_from: Optional[int] = None,
        price_to: Optional[int] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        pages: int = 1,
        additional_params: Optional[Dict[str, str]] = None,
    ) -> List[Listing]:
        all_results: List[Listing] = []
        for page in range(1, max(1, int(pages)) + 1):
            url = self.build_search_url(
                city=city,
                make=make,
                model=model,
                price_from=price_from,
                price_to=price_to,
                year_from=year_from,
                year_to=year_to,
                page=page,
                additional_params=additional_params,
            )
            self.logger.info("Fetching %s", url)
            try:
                response = self._get(url)
            except PermissionError as e:
                self.logger.error(str(e))
                break
            except requests.RequestException as e:
                self.logger.error("Request failed: %s", e)
                break

            listings = self.parse_listings(response.text)
            if not listings:
                # Stop early if a page yields no listings
                break
            all_results.extend(listings)
        return all_results