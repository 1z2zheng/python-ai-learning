#!/usr/bin/env python3
# coding: utf-8
"""Fetch historical prices for product URLs listed in an Excel file."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_URL = "https://apapia.manmanbuy.com/ChromeWidgetServices/WidgetServices.ashx"
TREND_PATTERN = re.compile(
    r"Date\.UTC\((\d{4}),(\d{1,2}),(\d{1,2})\),(-?\d+(?:\.\d+)?)"
)
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
OFFICIAL_WEB_SCRIPT = Path(__file__).with_name("official_web_query_v2.mjs")


def encode_product_url(value: str) -> str:
    """Apply the double-encoding expected by the legacy widget endpoint."""
    replacements = {
        "/": "%252F",
        "?": "%253F",
        "=": "%253D",
        ":": "%253A",
        "&": "%26",
    }
    return "".join(replacements.get(char, char) for char in value)


def safe_filename(value: str, limit: int = 100) -> str:
    value = INVALID_FILENAME_CHARS.sub("_", value).strip(" .")
    return value[:limit].rstrip(" .") or "product"


def parse_lower_date(value: Any) -> str:
    text = str(value).strip()
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    match = re.search(r"\d+", text)
    if not match:
        return ""
    timestamp = int(match.group(0))
    if timestamp > 10_000_000_000:
        timestamp //= 1000
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).date().isoformat()


def parse_history(raw_history: str) -> pd.DataFrame:
    rows = []
    for year, month, day, price in TREND_PATTERN.findall(raw_history or ""):
        year_value = int(year)
        month_value = int(month)
        normalized_year = year_value + month_value // 12
        normalized_month = month_value % 12 + 1
        normalized_date = dt.date(normalized_year, normalized_month, 1) + dt.timedelta(
            days=int(day) - 1
        )
        rows.append({"date_日期": normalized_date, "price_价格": float(price)})
    if not rows:
        raise ValueError("接口响应中没有可解析的历史价格数据")
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["date_日期"], keep="last")
        .sort_values("date_日期")
        .reset_index(drop=True)
    )


def parse_official_history(raw_history: str) -> pd.DataFrame:
    """Parse the datePrice array returned by the supported Manmanbuy web page."""
    try:
        points = json.loads(f"[{raw_history}]")
    except json.JSONDecodeError as exc:
        raise ValueError("慢慢买官方网页响应中的 datePrice 无法解析") from exc
    rows = []
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            continue
        timestamp_ms, price = point[0], point[1]
        try:
            date_value = dt.datetime.fromtimestamp(
                float(timestamp_ms) / 1000, tz=dt.timezone.utc
            ).date()
            promotion_text = str(point[2]).strip() if len(point) > 2 and point[2] else ""
            rows.append({
                "date_日期": date_value,
                "price_价格": float(price),
                "promotion_text": promotion_text,
            })
        except (TypeError, ValueError, OSError):
            continue
    if not rows:
        raise ValueError("慢慢买官方网页响应中没有可解析的历史价格数据")
    return (
        pd.DataFrame(rows)
        .sort_values(["date_日期", "price_价格"], ascending=[True, True])
        .drop_duplicates(subset=["date_日期"], keep="first")
        .sort_values("date_日期")
        .reset_index(drop=True)
    )


def fetch_product_official_web(
    product_url: str,
    timeout: float = 75.0,
    cdp_url: str = "http://127.0.0.1:9222",
    node_executable: str | None = None,
) -> dict[str, Any]:
    """Query the official Manmanbuy page through a user-authenticated Edge session."""
    node = node_executable or os.environ.get("MANMANBUY_NODE") or shutil.which("node")
    if not node:
        bundled = Path(
            r"C:\Users\29148\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
        )
        node = str(bundled) if bundled.exists() else None
    if not node:
        raise RuntimeError("未找到 Node.js；请安装 Node.js 或设置 MANMANBUY_NODE")
    if not OFFICIAL_WEB_SCRIPT.exists():
        raise RuntimeError(f"缺少官方网页查询脚本：{OFFICIAL_WEB_SCRIPT}")

    completed = subprocess.run(
        [node, str(OFFICIAL_WEB_SCRIPT), product_url, cdp_url],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = (completed.stderr or completed.stdout).strip()[-500:]
        raise RuntimeError(f"官方网页查询器未返回有效 JSON：{detail}") from exc
    status = result.get("status")
    if status == "login_required":
        raise RuntimeError("慢慢买登录已失效，请在专用 Edge 窗口中重新登录")
    if status != "ok" or not isinstance(result.get("data"), dict):
        raise RuntimeError(f"慢慢买官方网页查询失败：{status or 'unknown'}")
    return result["data"]


def fetch_product(product_url: str, timeout: float = 20.0) -> dict[str, Any]:
    if not re.match(r"^https?://", product_url, flags=re.IGNORECASE):
        raise ValueError(f"不是有效的 HTTP(S) 商品链接：{product_url}")

    payload = (
        "c_devid=2C5039AF-99D0-4800-BC36-DEB3654D202C&username=&qs=true&"
        "c_engver=1.5.0&c_devtoken=&c_devmodel=iPhone%20SE&c_contype=wifi&"
        f"t={int(time.time() * 1000)}&c_win=w_320_h_568&p_url={encode_product_url(product_url)}&"
        "c_ostype=ios&jsoncallback=%3F&c_ctrl=w_search_trend0_f_content&"
        "methodName=getBiJiaInfo_wxsmall&c_devtype=phone&jgzspic=no&"
        "c_operator=%E4%B8%AD%E5%9B%BD%E7%A7%BB%E5%8A%A8&c_appver=3.6.0&"
        "bj=false&c_dp=2&c_osver=10.3.3"
    )
    headers = {
        "Host": "apapia.manmanbuy.com",
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        "Accept-Encoding": "gzip",
        "Connection": "close",
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_3 like Mac OS X) "
            "AppleWebKit/603.3.8 (KHTML, like Gecko) Mobile/14G60 mmbWebBrowse"
        ),
    }

    response = requests.get(API_URL, headers=headers, data=payload, timeout=timeout)
    response.raise_for_status()
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        preview = re.sub(r"\s+", " ", response.text[:160]).strip()
        content_type = response.headers.get("Content-Type", "unknown")
        raise ValueError(
            f"接口未返回有效 JSON（HTTP {response.status_code}，"
            f"Content-Type={content_type}，响应摘要={preview!r}）"
        ) from exc
    if body.get("ok") != 1 or not isinstance(body.get("single"), dict):
        raise ValueError(f"接口返回失败：{body.get('msg') or '未知原因'}")
    return body["single"]


def fetch_and_save(
    product_url: str,
    output_dir: Path,
    timeout: float = 75.0,
    provider: str = "official-web",
    cdp_url: str = "http://127.0.0.1:9222",
    node_executable: str | None = None,
) -> Path:
    if provider == "official-web":
        product = fetch_product_official_web(
            product_url, timeout=timeout, cdp_url=cdp_url, node_executable=node_executable
        )
        title = str(product.get("spName") or "product")
        history = parse_official_history(str(product.get("datePrice") or ""))
        item_url = product.get("spUrl") or product_url
    else:
        product = fetch_product(product_url, timeout=timeout)
        title = str(product.get("title") or "product")
        history = parse_history(str(product.get("jiagequshi") or ""))
        item_url = product.get("url") or product_url
    sku_match = re.search(r"\d+", product_url)
    sku = sku_match.group(0) if sku_match else "unknown"

    history["sku_id"] = sku
    history["title_名称"] = title
    history["lowerPrice_最低价格"] = product.get("lowerPrice")
    history["lowerDate_最低价格日期"] = parse_lower_date(product.get("lowerDate"))
    history["itemurl_商品链接"] = item_url

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sku}_{safe_filename(title)}.csv"
    history.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def read_urls(input_path: Path) -> list[str]:
    frame = pd.read_excel(input_path)
    if "web" not in frame.columns:
        raise ValueError(f"{input_path} 缺少名为 web 的列")
    urls = [str(value).strip() for value in frame["web"].dropna()]
    if not urls:
        raise ValueError(f"{input_path} 的 web 列没有商品链接")
    return urls


def run(
    input_path: Path,
    output_dir: Path,
    retries: int,
    timeout: float,
    delay: float,
    provider: str,
    cdp_url: str,
    node_executable: str | None,
) -> int:
    urls = read_urls(input_path)
    failures = 0
    print(f"读取到 {len(urls)} 个商品链接")

    for index, product_url in enumerate(urls, start=1):
        for attempt in range(1, retries + 1):
            try:
                output_path = fetch_and_save(
                    product_url,
                    output_dir,
                    timeout=timeout,
                    provider=provider,
                    cdp_url=cdp_url,
                    node_executable=node_executable,
                )
                print(f"[{index}/{len(urls)}] 成功：{output_path}")
                break
            except (requests.RequestException, ValueError, KeyError) as exc:
                if attempt >= retries:
                    failures += 1
                    print(f"[{index}/{len(urls)}] 失败：{product_url}；{exc}")
                else:
                    wait_seconds = delay * attempt
                    print(
                        f"[{index}/{len(urls)}] 第 {attempt} 次失败：{exc}；"
                        f"{wait_seconds:.1f} 秒后重试"
                    )
                    time.sleep(wait_seconds)
        if index < len(urls) and delay > 0:
            time.sleep(delay)

    print(f"完成：成功 {len(urls) - failures}，失败 {failures}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抓取商品历史价格并保存为 CSV")
    parser.add_argument("--input", type=Path, default=Path("webnow.xls"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--provider", choices=["official-web", "legacy"], default="official-web",
        help="official-web 使用已登录 Edge；legacy 使用已停维的旧接口",
    )
    parser.add_argument("--cdp-url", default="http://127.0.0.1:9222")
    parser.add_argument("--node", dest="node_executable")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.retries < 1:
        raise SystemExit("--retries 必须大于等于 1")
    return run(
        args.input,
        args.output_dir,
        args.retries,
        args.timeout,
        args.delay,
        args.provider,
        args.cdp_url,
        args.node_executable,
    )


if __name__ == "__main__":
    raise SystemExit(main())
