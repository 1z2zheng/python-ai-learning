from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


DIMENSION_KEYWORDS = {
    "手感体验": ("手感", "触感", "按键", "微动", "轴体", "回弹", "键程", "清脆", "段落感", "顺滑"),
    "响应性能": ("响应", "延迟", "回报率", "dpi", "传感器", "灵敏", "精准", "跟手", "丢帧", "断连", "连接", "无线", "蓝牙"),
    "舒适度": ("舒服", "舒适", "重量", "轻量", "轻巧", "轻便", "偏轻", "偏重", "累手", "握持", "趴握", "抓握", "人体工学", "大小", "手掌"),
    "耐用性": ("耐用", "做工", "质量", "松动", "异响", "双击", "涂层", "续航", "充电", "故障", "坏", "维修", "售后", "划痕", "分模线"),
    "外观设计": ("外观", "颜值", "灯效", "灯光", "rgb", "颜色", "设计", "好看", "炫酷"),
    "性价比": ("价格", "售价", "价位", "太贵", "小贵", "偏贵", "便宜", "划算", "性价比", "物超所值", "降价", "价保", "优惠", "不值"),
}

POSITIVE_TERMS = (
    "非常好", "很好", "好用", "满意", "不错", "舒服", "舒适", "稳定", "灵敏",
    "精准", "跟手", "顺滑", "清脆", "扎实", "耐用", "漂亮", "好看", "炫酷",
    "划算", "物超所值", "性价比高", "续航给力", "无延迟", "不丢帧", "无异响",
    "没问题", "没有问题", "推荐", "喜欢", "友好", "快", "轻巧", "轻便",
)

NEGATIVE_TERMS = (
    "不好", "不舒服", "不稳定", "不灵敏", "不精准", "不跟手", "不耐用", "不值",
    "失望", "卡顿", "断连", "延迟", "丢帧", "异响", "松动", "双击", "累手",
    "太重", "偏重", "太轻", "偏轻", "太贵", "售价偏高", "价格高", "小贵",
    "故障", "坏了", "维修", "划痕", "破损", "缺点", "黏手", "打滑", "偏滑",
    "恶心", "坑", "无法", "问题", "短板", "不行", "担心", "怀疑",
)

PLACEHOLDER_PATTERNS = (
    "此用户未及时填写评价内容",
    "系统默认好评",
    "默认好评",
    "评价方未及时做出评价",
)

SERVICE_TERMS = ("物流", "快递", "发货", "客服", "送货", "包装", "店铺", "商家", "京东")
PRODUCT_TERMS = tuple(dict.fromkeys(term for terms in DIMENSION_KEYWORDS.values() for term in terms))


def normalize_review_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def review_scope(platform_product_id: object, source_product_id: object) -> str:
    target = str(platform_product_id or "").strip()
    source = str(source_product_id or "").strip()
    if not source or source.lower() in {"nan", "none", "<na>"}:
        return "历史样本未记录来源SKU"
    if source == target:
        return "目标SKU"
    return "其他SKU待核验"


def extract_dimensions(content: str) -> list[str]:
    lowered = normalize_review_text(content).lower()
    return [
        dimension
        for dimension, keywords in DIMENSION_KEYWORDS.items()
        if any(keyword.lower() in lowered for keyword in keywords)
    ]


def quality_reasons(content: str, duplicate: bool = False) -> list[str]:
    text = normalize_review_text(content)
    reasons: list[str] = []
    if not text:
        reasons.append("空文本")
    if any(pattern in text for pattern in PLACEHOLDER_PATTERNS):
        reasons.append("平台占位评价")
    if len(re.sub(r"\W", "", text)) < 4:
        reasons.append("文本过短")
    if duplicate:
        reasons.append("重复文本")
    has_service = any(term in text for term in SERVICE_TERMS)
    has_product = any(term.lower() in text.lower() for term in PRODUCT_TERMS)
    if has_service and not has_product:
        reasons.append("仅含物流客服信息")
    if not extract_dimensions(text):
        reasons.append("未识别产品属性")
    return list(dict.fromkeys(reasons))


def split_clauses(content: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。！？!?；;，,\n]", content) if part.strip()]


def _term_count(text: str, terms: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(term.lower()) for term in terms)


def classify_dimension_sentiment(content: str, dimension: str, rating: float | None) -> tuple[str, int]:
    keywords = DIMENSION_KEYWORDS[dimension]
    clauses = [
        clause for clause in split_clauses(normalize_review_text(content))
        if any(keyword.lower() in clause.lower() for keyword in keywords)
    ]
    evidence = " ".join(clauses) if clauses else normalize_review_text(content)
    return classify_text_sentiment(evidence)


def classify_text_sentiment(content: str) -> tuple[str, int]:
    """Non-overlapping longest matches with local negation; never infer from stars."""
    terms = {term: 1 for term in POSITIVE_TERMS}
    terms.update({term: -1 for term in NEGATIVE_TERMS})
    terms.update({"不推荐": -1, "响应慢": -1, "无延迟": 1,
                  "没有延迟": 1, "没有异响": 1, "不累手": 1,
                  "不双击": 1, "没有问题": 1, "没问题": 1})
    pattern = re.compile("|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)))
    signs = []
    for clause in split_clauses(content):
        for match in pattern.finditer(clause):
            term = match.group()
            sign = terms[term]
            prefix = clause[max(0, match.start() - 5):match.start()]
            suffix = clause[match.end():match.end() + 6]
            if re.search(r"(?:没有|毫无|并不|不|没|无|未)(?:很|太|明显|任何)?$", prefix):
                sign *= -1
            if sign < 0 and re.match(r"(?:几乎)?(?:无感|没有|不明显|很低|极低)", suffix):
                sign = 1
            signs.append(sign)
    positive, negative = 1 in signs, -1 in signs
    if positive and negative:
        return "混合", 0
    if positive:
        return "正向", 1
    if negative:
        return "负向", -1
    return "中性", 0
