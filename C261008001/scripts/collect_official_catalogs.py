from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup


USER_AGENT = "AcademicResearchCatalogCollector/1.0"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6"}


@dataclass(frozen=True)
class CatalogSource:
    source_site: str
    brand: str
    category: str
    catalog_url: str
    product_path: str
    exclude_parts: tuple[str, ...] = ()
    max_products: int = 30


SOURCES = (
    CatalogSource(
        "logitech_official", "罗技", "mouse",
        "https://www.logitech.com/zh-cn/shop/c/gaming-mice", "/shop/p/", max_products=24,
    ),
    CatalogSource(
        "logitech_official", "罗技", "keyboard",
        "https://www.logitechg.com/zh-cn/shop/c/gaming-keyboards", "/shop/p/", max_products=22,
    ),
    CatalogSource(
        "razer_official", "雷蛇", "mouse",
        "https://cn.razerzone.com/pc/gaming-mice", "/gaming-mice/",
        ("/buy", "-line"), 18,
    ),
    CatalogSource(
        "razer_official", "雷蛇", "keyboard",
        "https://cn.razerzone.com/pc/gaming-keyboards-and-keypads", "/gaming-keyboards/",
        ("/buy", "-line"), 18,
    ),
    CatalogSource(
        "rog_official", "ROG", "mouse",
        "https://rog.asus.com.cn/mice-mouse-pads/mice-series/", "/mice-mouse-pads/mice/",
        ("/wtb/",), 20,
    ),
    CatalogSource(
        "rog_official", "ROG", "keyboard",
        "https://rog.asus.com.cn/keyboards/keyboards-series/", "/keyboards/keyboards/",
        ("/wtb/",), 24,
    ),
    CatalogSource(
        "dareu_official", "达尔优", "mouse",
        "https://cn.dareu.com/mouse/", "/products/mouse/", max_products=24,
    ),
    CatalogSource(
        "dareu_official", "达尔优", "keyboard",
        "https://cn.dareu.com/keyboard/", "/products/keyboard/", max_products=28,
    ),
    CatalogSource(
        "keychron_official", "Keychron", "keyboard",
        "https://www.keychron.com/collections/mechanical-keyboard-for-gaming", "/products/",
        ("/products/keycaps", "/products/switch"), 16,
    ),
)

VGN_CATALOG_URL = "https://www.vgnlab.com.cn/?tab=1"
VGN_STATIC_CATALOG_URL = "https://static.vgnclub.com/vgn/static/js/webIndex.js"


class Collector:
    def __init__(self, delay: float, timeout: float) -> None:
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.last_request_at: dict[str, float] = {}
        self.robots: dict[str, RobotFileParser] = {}

    def _robot_parser(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self.robots:
            parser = RobotFileParser(urljoin(origin, "/robots.txt"))
            try:
                parser.read()
            except OSError:
                parser = RobotFileParser()
                parser.parse(["User-agent: *", "Allow: /"])
            self.robots[origin] = parser
        return self.robots[origin]

    def fetch(self, url: str) -> requests.Response:
        if not self._robot_parser(url).can_fetch(USER_AGENT, url):
            raise PermissionError(f"robots.txt 不允许采集：{url}")
        host = urlparse(url).netloc
        elapsed = time.monotonic() - self.last_request_at.get(host, 0.0)
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        response = self.session.get(url, timeout=self.timeout)
        self.last_request_at[host] = time.monotonic()
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return response


def unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        normalized = value.split("#", 1)[0]
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def product_links(source: CatalogSource, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        url = urljoin(source.catalog_url, anchor["href"])
        parsed = urlparse(url)
        if parsed.netloc != urlparse(source.catalog_url).netloc:
            continue
        if source.product_path not in parsed.path:
            continue
        if any(part in parsed.path.lower() for part in source.exclude_parts):
            continue
        if parsed.path.rstrip("/") == urlparse(source.catalog_url).path.rstrip("/"):
            continue
        links.append(url)
    return unique(links)[: source.max_products]


def json_objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from json_objects(child)


def json_ld_products(soup: BeautifulSoup) -> list[dict[str, Any]]:
    products = []
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            value = json.loads(script.string or script.get_text())
        except (json.JSONDecodeError, TypeError):
            continue
        for item in json_objects(value):
            item_type = item.get("@type")
            types = item_type if isinstance(item_type, list) else [item_type]
            if "Product" in types:
                products.append(item)
    return products


def first_meta(soup: BeautifulSoup, *keys: str) -> str | None:
    for key in keys:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
    return None


def parse_offer(product: dict[str, Any], soup: BeautifulSoup) -> tuple[float | None, str | None]:
    offers = product.get("offers")
    if isinstance(offers, list):
        offers = next((item for item in offers if isinstance(item, dict)), None)
    if isinstance(offers, dict):
        price = offers.get("price") or offers.get("lowPrice")
        currency = offers.get("priceCurrency")
        try:
            if price is not None and float(str(price).replace(",", "")) > 0:
                return float(str(price).replace(",", "")), str(currency or "").upper() or None
        except ValueError:
            pass

    meta_price = first_meta(soup, "product:price:amount")
    meta_currency = first_meta(soup, "product:price:currency")
    if meta_price:
        try:
            return float(meta_price.replace(",", "")), (meta_currency or "").upper() or None
        except ValueError:
            pass

    text = soup.get_text(" ", strip=True)
    for pattern, currency in ((r"[￥¥]\s*([1-9]\d*(?:\.\d{1,2})?)", "CNY"),
                              (r"(?:US\s*)?\$\s*([1-9]\d*(?:\.\d{1,2})?)", "USD")):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)), currency
    return None, None


def normalize_connection(text: str) -> str:
    lowered = text.lower()
    has_24g = bool(re.search(r"2\.4\s*g|lightspeed|hyperspeed|speednova|无线", lowered))
    has_bt = "bluetooth" in lowered or "蓝牙" in text
    has_wired = any(value in lowered for value in ("wired", "有线", "usb-c", "type-c"))
    count = sum((has_24g, has_bt, has_wired))
    if count >= 2:
        return "multi_mode"
    if has_24g:
        return "wireless_2_4g"
    if has_bt:
        return "bluetooth"
    if has_wired:
        return "wired"
    return "unknown"


def maximum_number(text: str, pattern: str, multiplier: int = 1) -> int | None:
    values = []
    for value in re.findall(pattern, text, flags=re.IGNORECASE):
        try:
            values.append(int(float(value.replace(",", "")) * multiplier))
        except ValueError:
            continue
    return max(values) if values else None


def extract_specs(category: str, text: str) -> dict[str, Any]:
    weight_values = [float(value) for value in re.findall(r"(?<!\d)(\d{2,3}(?:\.\d+)?)\s*(?:g|克)\b", text, re.I)]
    weight = min((value for value in weight_values if 20 <= value <= 2000), default=None)

    hz_values = [int(value.replace(",", "")) for value in re.findall(r"([1-9]\d{2,4}(?:,\d{3})?)\s*Hz", text, re.I)]
    khz_values = [int(float(value) * 1000) for value in re.findall(r"([1-9](?:\.\d+)?)\s*kHz", text, re.I)]
    polling_rate = max(hz_values + khz_values, default=None)

    result: dict[str, Any] = {
        "connection_type": normalize_connection(text),
        "weight_g": weight,
        "polling_rate_hz": polling_rate,
        "rgb": "true" if re.search(r"\bRGB\b|灯效|Chroma|Aura", text, re.I) else "false",
        "dpi_max": None,
        "sensor_model": None,
        "button_count": None,
        "handedness": None,
        "switch_type": None,
        "switch_name": None,
        "layout_size": None,
        "key_count": None,
        "actuation_mm": None,
        "hot_swappable": None,
    }
    if category == "mouse":
        result["dpi_max"] = maximum_number(text, r"([1-9]\d{2,5}(?:,\d{3})?)\s*dpi")
        result["handedness"] = (
            "left" if "左手" in text else "symmetric" if "对称" in text else
            "right" if re.search(r"右手|人体工学", text) else None
        )
    else:
        layout = re.search(r"\b(60|65|68|75|80|87|96|98|100)%", text)
        if layout:
            result["layout_size"] = f"{layout.group(1)}%"
        elif re.search(r"\bTKL\b|无数字键", text, re.I):
            result["layout_size"] = "TKL"
        result["key_count"] = maximum_number(text, r"(?<!\d)(\d{2,3})\s*键")
        if re.search(r"磁轴|霍尔|magnetic|hall effect", text, re.I):
            result["switch_type"] = "magnetic"
        elif re.search(r"光轴|optical", text, re.I):
            result["switch_type"] = "optical"
        elif re.search(r"机械轴|mechanical", text, re.I):
            result["switch_type"] = "mechanical"
        elif re.search(r"薄膜|membrane", text, re.I):
            result["switch_type"] = "membrane"
        actuation = re.search(r"([01](?:\.\d+)?)\s*mm[^。；;]{0,30}(?:触发|actuation)", text, re.I)
        result["actuation_mm"] = float(actuation.group(1)) if actuation else None
        result["hot_swappable"] = "true" if re.search(r"热插拔|hot[- ]?swap", text, re.I) else "false"
    return result


def parse_product(source: CatalogSource, url: str, html: str, collected_at: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    products = json_ld_products(soup)
    product = products[0] if products else {}
    heading = soup.find("h1")
    name = str(product.get("name") or (heading.get_text(" ", strip=True) if heading else "") or
               first_meta(soup, "og:title") or urlparse(url).path.rstrip("/").split("/")[-1])
    description = str(product.get("description") or first_meta(soup, "og:description", "description") or "")
    image = product.get("image") or first_meta(soup, "og:image")
    if isinstance(image, list):
        image = image[0] if image else None
    price, currency = parse_offer(product, soup)
    text = f"{name}\n{description}\n{soup.get_text(' ', strip=True)}"
    specs = extract_specs(source.category, text)
    identifier = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    row = {
        "platform": source.source_site,
        "platform_product_id": slug or identifier,
        "category": source.category,
        "brand": source.brand,
        "model": name,
        "product_name": name,
        "shop_name": f"{source.brand}官网",
        "product_url": url,
        "image_url": image,
        "is_active": "true",
        "list_price": price,
        "sale_price": price,
        "coupon_price": None,
        "currency": currency,
        "promotion_text": None,
        "collected_at": collected_at,
        "source_site": source.source_site,
        "source_catalog_url": source.catalog_url,
        "source_text": description[:1000],
        **specs,
    }
    return row


def parse_vgn_catalog(script_text: str, collected_at: str) -> list[dict[str, Any]]:
    """Parse the public VGN product configuration embedded in webIndex.js."""
    rows: list[dict[str, Any]] = []
    category: str | None = None
    title: str | None = None
    subtitle = ""

    for raw_line in script_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        title_match = re.match(r"title\s*:\s*['\"]([^'\"]+)['\"]", line)
        if title_match:
            value = title_match.group(1).strip()
            if value == "键盘":
                category, title, subtitle = "keyboard", None, ""
            elif value == "鼠标":
                category, title, subtitle = "mouse", None, ""
            elif value in {"耳机", "配件"}:
                category, title, subtitle = None, None, ""
            elif category:
                title, subtitle = value, ""
            continue

        subtitle_match = re.match(r"subTitle\s*:\s*['\"]([^'\"]*)['\"]", line)
        if subtitle_match and category and title:
            subtitle = subtitle_match.group(1).strip()
            continue

        url_match = re.match(r"url\s*:\s*['\"](/web/sku-[^'\"]+)['\"]", line)
        if not (url_match and category and title):
            continue

        path = url_match.group(1)
        product_url = urljoin("https://www.vgnlab.com.cn", path)
        product_id = path.rstrip("/").split("-")[-1]
        text = f"{title}\n{subtitle}"
        row = {
            "platform": "vgn_official",
            "platform_product_id": product_id,
            "category": category,
            "brand": "VGN",
            "model": title,
            "product_name": title,
            "shop_name": "VGN 官网",
            "product_url": product_url,
            "image_url": None,
            "is_active": "true",
            "list_price": None,
            "sale_price": None,
            "coupon_price": None,
            "currency": None,
            "promotion_text": None,
            "collected_at": collected_at,
            "source_site": "vgn_official",
            "source_catalog_url": VGN_CATALOG_URL,
            "source_text": subtitle,
            **extract_specs(category, text),
        }
        rows.append(row)
        title, subtitle = None, ""

    return rows


def collect(
    output_path: Path,
    log_path: Path,
    delay: float,
    timeout: float,
    existing_path: Path | None = None,
    only_vgn: bool = False,
) -> None:
    collector = Collector(delay=delay, timeout=timeout)
    collected_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    source_counts: dict[str, int] = {}

    if existing_path:
        with existing_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))

    for source in (() if only_vgn else SOURCES):
        key = f"{source.source_site}:{source.category}"
        try:
            catalog = collector.fetch(source.catalog_url)
            links = product_links(source, catalog.text)
        except (requests.RequestException, PermissionError) as exc:
            errors.append(f"{key} 目录失败：{exc}")
            continue

        count = 0
        for url in links:
            try:
                response = collector.fetch(url)
                rows.append(parse_product(source, url, response.text, collected_at))
                count += 1
            except (requests.RequestException, PermissionError, ValueError) as exc:
                errors.append(f"{url}：{exc}")
        source_counts[key] = count
        print(f"{key}：发现 {len(links)}，成功 {count}")

    try:
        vgn_response = collector.fetch(VGN_STATIC_CATALOG_URL)
        vgn_rows = parse_vgn_catalog(vgn_response.text, collected_at)
        rows.extend(vgn_rows)
        for category in ("mouse", "keyboard"):
            source_counts[f"vgn_official:{category}"] = sum(
                row["category"] == category for row in vgn_rows
            )
        print(f"vgn_official：成功 {len(vgn_rows)}")
    except (requests.RequestException, PermissionError, ValueError) as exc:
        errors.append(f"vgn_official 目录失败：{exc}")

    deduplicated = {row["product_url"]: row for row in rows}
    rows = sorted(deduplicated.values(), key=lambda row: (row["brand"], row["category"], row["product_name"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    log = {
        "collected_at": collected_at,
        "output": str(output_path),
        "record_count": len(rows),
        "source_counts": source_counts,
        "price_count": sum(
            str(row.get("sale_price") or "").strip().lower() not in {"", "nan", "none"}
            for row in rows
        ),
        "errors": errors,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="低频采集电竞键鼠品牌官网公开目录")
    parser.add_argument("--output", type=Path, default=Path("data/raw/official_catalog.csv"))
    parser.add_argument("--log", type=Path, default=Path("logs/official_catalog_collection.json"))
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--existing", type=Path, help="先加载已有 CSV，再追加新来源")
    parser.add_argument("--only-vgn", action="store_true", help="仅采集 VGN 公开目录")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    collect(
        arguments.output,
        arguments.log,
        max(arguments.delay, 0.5),
        arguments.timeout,
        arguments.existing,
        arguments.only_vgn,
    )
