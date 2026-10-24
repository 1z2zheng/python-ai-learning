from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "seed" / "demo_80_products.csv"
TEMPLATE = ROOT / "data" / "seed" / "products_template.csv"

BRANDS = ["罗技", "雷蛇", "ROG", "VGN", "达尔优"]
DATES = ["2026-09-08T10:00:00+08:00", "2026-09-22T10:00:00+08:00", "2026-10-08T10:00:00+08:00"]


def mouse_row(brand: str, brand_index: int, variant: int, date_index: int) -> dict[str, object]:
    product_number = brand_index * 8 + variant + 1
    base_price = [399, 459, 599, 249, 199][brand_index] + variant * 45
    factor = [1.10, 1.05, 1.00][date_index]
    sale_price = round(base_price * factor, 2)
    coupon_price = round(sale_price - 30, 2) if date_index == 2 and variant % 2 == 0 else ""
    return {
        "platform": "manual",
        "platform_product_id": f"DEMO-MOUSE-{product_number:03d}",
        "category": "mouse",
        "brand": brand,
        "model": f"DEMO-M{product_number:03d}",
        "product_name": f"{brand} 演示鼠标 M{product_number:03d}",
        "shop_name": "功能测试数据 非真实市场",
        "product_url": "",
        "image_url": "",
        "is_active": "true",
        "connection_type": ["wired", "wireless_2_4g", "multi_mode"][variant % 3],
        "weight_g": 52 + (variant * 4 + brand_index) % 34,
        "polling_rate_hz": [1000, 2000, 4000, 8000][variant % 4],
        "rgb": "true" if variant % 3 == 0 else "false",
        "warranty_months": [12, 18, 24][brand_index % 3],
        "dpi_max": [16000, 20000, 26000, 32000, 35000][(variant + brand_index) % 5],
        "sensor_model": f"演示传感器 S{(variant % 4) + 1}",
        "button_count": 5 + variant % 4,
        "handedness": "symmetric" if variant % 2 == 0 else "right",
        "switch_type": "",
        "switch_name": "",
        "layout_size": "",
        "key_count": "",
        "actuation_mm": "",
        "hot_swappable": "",
        "list_price": round(base_price * 1.18, 2),
        "sale_price": sale_price,
        "coupon_price": coupon_price,
        "currency": "CNY",
        "promotion_text": "演示优惠" if coupon_price != "" else "",
        "collected_at": DATES[date_index],
    }


def keyboard_row(brand: str, brand_index: int, variant: int, date_index: int) -> dict[str, object]:
    product_number = brand_index * 8 + variant + 1
    base_price = [499, 599, 899, 299, 259][brand_index] + variant * 40
    factor = [1.12, 1.06, 1.00][date_index]
    sale_price = round(base_price * factor, 2)
    coupon_price = round(sale_price - 40, 2) if date_index == 2 and variant % 2 == 1 else ""
    switch_type = ["mechanical", "magnetic", "optical"][variant % 3]
    actuation = {"mechanical": 1.8, "magnetic": 0.1, "optical": 0.8}[switch_type]
    return {
        "platform": "manual",
        "platform_product_id": f"DEMO-KEY-{product_number:03d}",
        "category": "keyboard",
        "brand": brand,
        "model": f"DEMO-K{product_number:03d}",
        "product_name": f"{brand} 演示键盘 K{product_number:03d}",
        "shop_name": "功能测试数据 非真实市场",
        "product_url": "",
        "image_url": "",
        "is_active": "true",
        "connection_type": ["wired", "multi_mode", "wireless_2_4g"][variant % 3],
        "weight_g": 620 + variant * 55 + brand_index * 10,
        "polling_rate_hz": [1000, 2000, 4000, 8000][variant % 4],
        "rgb": "true" if variant % 4 != 0 else "false",
        "warranty_months": [12, 18, 24][brand_index % 3],
        "dpi_max": "",
        "sensor_model": "",
        "button_count": "",
        "handedness": "",
        "switch_type": switch_type,
        "switch_name": f"演示轴体 {(variant % 5) + 1}",
        "layout_size": ["65%", "75%", "98%", "100%"][variant % 4],
        "key_count": [68, 81, 98, 104][variant % 4],
        "actuation_mm": actuation,
        "hot_swappable": "false" if switch_type == "magnetic" else "true",
        "list_price": round(base_price * 1.20, 2),
        "sale_price": sale_price,
        "coupon_price": coupon_price,
        "currency": "CNY",
        "promotion_text": "演示优惠" if coupon_price != "" else "",
        "collected_at": DATES[date_index],
    }


def main() -> None:
    with TEMPLATE.open("r", encoding="utf-8-sig", newline="") as handle:
        fieldnames = next(csv.reader(handle))
    rows = []
    for brand_index, brand in enumerate(BRANDS):
        for variant in range(8):
            for date_index in range(len(DATES)):
                rows.append(mouse_row(brand, brand_index, variant, date_index))
                rows.append(keyboard_row(brand, brand_index, variant, date_index))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已生成 {len(rows)} 条价格记录，覆盖 80 款演示商品: {OUTPUT}")


if __name__ == "__main__":
    main()

