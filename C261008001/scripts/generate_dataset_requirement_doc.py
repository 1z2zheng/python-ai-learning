from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(r"D:\C261008001")
OUTPUT = ROOT / "output" / "电竞外设竞品分析项目数据集采集与交付规范.docx"

NAVY = "1F4E78"
PALE_BLUE = "EAF2F8"
PALE_GRAY = "F6F7F8"
BORDER = "D9D9D9"
BLACK = RGBColor(0, 0, 0)
GRAY = RGBColor(89, 89, 89)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=110, bottom=100, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "6")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), BORDER)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_repeat_paragraph_font(paragraph, size=9, color=BLACK, bold=False):
    for run in paragraph.runs:
        run.font.name = "微软雅黑"
        run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold


def remove_paragraph_borders(paragraph_or_style):
    if hasattr(paragraph_or_style, "_p"):
        p_pr = paragraph_or_style._p.get_or_add_pPr()
    else:
        p_pr = paragraph_or_style.element.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is not None:
        p_pr.remove(borders)


def add_table(doc, headers, rows, widths=None, alignments=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    set_repeat_table_header(header)
    for idx, text in enumerate(headers):
        cell = header.cells[idx]
        cell.text = text
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_shading(cell, NAVY)
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        set_repeat_paragraph_font(p, size=9, color=RGBColor(255, 255, 255), bold=True)
    for ridx, row_data in enumerate(rows):
        row = table.add_row()
        for idx, value in enumerate(row_data):
            cell = row.cells[idx]
            cell.text = str(value)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            if ridx % 2 == 1:
                set_cell_shading(cell, PALE_BLUE)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            if alignments and idx < len(alignments):
                p.alignment = alignments[idx]
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            set_repeat_paragraph_font(p, size=8.7)
    if widths:
        for row in table.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Cm(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_bullet(doc, text, level=0):
    style = "List Bullet" if level == 0 else "List Bullet 2"
    p = doc.add_paragraph(style=style)
    p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.25
    return p


def add_number(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.25
    return p


def add_field(paragraph, instruction):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.line_spacing = 1.38
    normal.paragraph_format.space_after = Pt(6)

    title = styles["Title"]
    title.font.name = "微软雅黑"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    title.font.size = Pt(22)
    title.font.bold = True
    title.font.color.rgb = BLACK
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(12)
    remove_paragraph_borders(title)

    for style_name, size, before, after in (
        ("Heading 1", 15, 16, 8),
        ("Heading 2", 12, 12, 6),
        ("Heading 3", 10.5, 9, 4),
    ):
        style = styles[style_name]
        style.font.name = "微软雅黑"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for name in ("List Bullet", "List Bullet 2", "List Number"):
        style = styles[name]
        style.font.name = "宋体"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        style.font.size = Pt(10.5)


def build_document():
    doc = Document()
    configure_styles(doc)
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.35)
    section.right_margin = Cm(2.35)

    title = doc.add_paragraph(style="Title")
    title.add_run("电竞外设竞品分析项目数据集采集与交付规范")
    remove_paragraph_borders(title)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("适用于基于 Streamlit 的竞品定价与偏好可视化系统")
    run.font.name = "微软雅黑"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
    run.font.size = Pt(12)
    run.font.color.rgb = GRAY
    subtitle.paragraph_format.space_after = Pt(14)

    intro = doc.add_paragraph()
    intro.add_run("使用说明  ").bold = True
    intro.add_run(
        "本规范用于指导京东、天猫及其他合规来源的电竞键盘和电竞鼠标数据采集、整理与交付。"
        "研究数据集应以可追溯的真实商品记录为主体，至少覆盖 80 个唯一商品、5 个品牌和 2 个平台。"
        "每条价格记录必须带采集时间；用于价格趋势分析时，同一商品应保留多期快照。"
    )

    doc.add_heading("一 数据集目标和边界", level=1)
    p = doc.add_paragraph()
    p.add_run("研究对象。 ").bold = True
    p.add_run("电竞鼠标与电竞键盘，包括面向游戏场景销售的有线、无线和多模产品。普通办公外设、二手商品、配件、键帽、鼠标垫和维修服务不纳入核心样本。")
    p = doc.add_paragraph()
    p.add_run("分析用途。 ").bold = True
    p.add_run("数据将支持市场概览、用户偏好、价格分布、促销强度、竞品对比和价格趋势等功能。商品身份、规格和价格快照必须能够通过稳定主键关联。")
    p = doc.add_paragraph()
    p.add_run("真实性要求。 ").bold = True
    p.add_run("论文和系统验收使用的数据必须来自合规的官方接口、商家后台导出、经授权的联盟接口、许可明确的开放数据集，或有记录的人工采集。演示数据必须单独标记，不得作为真实研究样本。")

    doc.add_heading("二 样本规模和覆盖要求", level=1)
    rows = [
        ("唯一商品数", "至少 80", "100 至 150", "按平台商品 ID 去重；跨平台同型号可分别保留"),
        ("品类", "鼠标不少于 30，键盘不少于 30", "两类数量尽量接近", "剩余样本可按研究重点分配"),
        ("品牌", "至少 5 个", "8 至 12 个", "同时覆盖头部品牌和中腰部品牌"),
        ("平台", "至少 2 个", "京东与天猫为主，可增加其他平台", "每个平台应形成可比较的有效样本"),
        ("价格快照", "每个商品至少 3 期", "每周 1 次，连续 8 至 12 周", "采集时间尽量固定，保留历史记录"),
        ("有效字段率", "核心字段不低于 95%", "规格字段不低于 85%", "缺失值使用空值，不使用虚构内容补齐"),
    ]
    add_table(
        doc,
        ["指标", "最低要求", "建议目标", "说明"],
        rows,
        widths=[3.0, 3.5, 4.1, 5.5],
        alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
    )

    doc.add_heading("三 推荐的数据表结构", level=1)
    p = doc.add_paragraph("建议采用五张逻辑表。若数据量较小，也可以先使用一张宽表导入，再由系统拆分入库。")
    rows = [
        ("products", "商品主表", "一行一个平台商品", "platform 与 platform_product_id"),
        ("product_specs", "规格表", "一行一个商品的结构化规格", "product_key"),
        ("price_snapshots", "价格快照表", "一行表示某商品在某时点的价格", "product_key 与 collected_at"),
        ("market_metrics", "销量或热度表", "一行表示某时点的指标", "product_key 与 observed_at"),
        ("reviews_summary", "评价汇总表", "一行表示某时点的评价统计", "product_key 与 observed_at"),
        ("data_provenance", "数据来源表", "记录文件或接口的来源和授权信息", "source_batch_id"),
    ]
    add_table(doc, ["表名", "用途", "记录粒度", "建议主键"], rows, widths=[3.3, 4.0, 5.2, 3.6])

    doc.add_heading("四 商品主表必需字段", level=1)
    rows = [
        ("platform", "文本", "是", "jd、tmall、taobao、suning 等统一代码", "jd"),
        ("platform_product_id", "文本", "是", "平台商品或 SKU 的稳定标识", "100012345678"),
        ("category", "文本", "是", "mouse 或 keyboard", "mouse"),
        ("brand", "文本", "是", "标准化品牌名，不混用中英文别名", "Logitech"),
        ("product_name", "文本", "是", "页面展示的完整商品名称", "罗技 G502 X 游戏鼠标"),
        ("model", "文本", "建议", "厂商型号；用于跨平台匹配", "G502 X"),
        ("shop_name", "文本", "建议", "店铺名称", "品牌官方旗舰店"),
        ("product_url", "文本", "建议", "可复核的商品页面地址", "https://..."),
        ("image_url", "文本", "可选", "商品主图地址，仅用于展示", "https://...jpg"),
        ("is_active", "布尔", "建议", "采集时是否在售", "true"),
        ("source_batch_id", "文本", "是", "关联来源和采集批次", "JD_20261009_01"),
    ]
    add_table(
        doc,
        ["字段", "类型", "级别", "定义和规则", "示例"],
        rows,
        widths=[3.1, 1.5, 1.6, 6.1, 3.7],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
    )

    doc.add_heading("五 价格和促销快照字段", level=1)
    p = doc.add_paragraph()
    p.add_run("关键规则。 ").bold = True
    p.add_run("价格表必须保留历史快照，不能用新价格覆盖旧价格。sale_price 是当时页面可直接购买的价格；coupon_price 仅在优惠券条件明确且可计算时填写。")
    rows = [
        ("product_key", "文本", "是", "关联商品主表的内部键"),
        ("list_price", "数值", "建议", "划线价或标价；无可靠数值时留空"),
        ("sale_price", "数值", "是", "采集时页面显示的实际销售价，单位为元"),
        ("coupon_price", "数值", "可选", "满足明确条件后的到手价"),
        ("currency", "文本", "是", "人民币统一填写 CNY"),
        ("promotion_text", "文本", "建议", "满减、直降、赠品、会员价等原始说明"),
        ("stock_status", "文本", "建议", "in_stock、out_of_stock 或 unknown"),
        ("collected_at", "日期时间", "是", "北京时间，推荐 YYYY-MM-DD HH:MM:SS"),
    ]
    add_table(doc, ["字段", "类型", "级别", "定义和规则"], rows, widths=[3.5, 2.0, 2.0, 8.5])

    doc.add_heading("六 电竞鼠标规格字段", level=1)
    rows = [
        ("connection_type", "有线、2.4G、蓝牙、多模", "统一使用受控枚举"),
        ("weight_g", "产品重量，单位克", "仅保存数值；不含线材时应在备注说明"),
        ("polling_rate_hz", "回报率，单位 Hz", "例如 1000、4000、8000"),
        ("dpi_max", "最大 DPI", "保存厂商公开的最大值"),
        ("sensor_model", "传感器型号", "无法核实则留空"),
        ("button_count", "可编程或物理按键数", "统一采用项目定义的统计口径"),
        ("handedness", "右手、左手、对称", "使用 right、left、ambidextrous"),
        ("rgb", "是否支持灯光", "true、false 或 null"),
        ("battery_hours", "标称续航小时数", "有线产品留空"),
        ("warranty_months", "保修期月数", "按官方或店铺明确说明填写"),
    ]
    add_table(doc, ["字段", "含义", "清洗规则"], rows, widths=[4.0, 5.2, 6.8])

    doc.add_heading("七 电竞键盘规格字段", level=1)
    rows = [
        ("connection_type", "有线、2.4G、蓝牙、多模", "统一使用受控枚举"),
        ("weight_g", "产品重量，单位克", "仅保存数值"),
        ("polling_rate_hz", "回报率，单位 Hz", "无法核实则留空"),
        ("switch_type", "机械、光轴、磁轴、薄膜等", "使用统一类别"),
        ("switch_name", "具体轴体名称", "保留厂商原名"),
        ("layout_size", "60、65、75、80、96、100 等", "统一为整数百分比或项目枚举"),
        ("key_count", "按键数量", "仅保存整数"),
        ("actuation_mm", "触发行程，单位毫米", "磁轴可记录可调范围说明"),
        ("hot_swappable", "是否支持热插拔", "true、false 或 null"),
        ("rgb", "是否支持灯光", "true、false 或 null"),
        ("warranty_months", "保修期月数", "按官方或店铺明确说明填写"),
    ]
    add_table(doc, ["字段", "含义", "清洗规则"], rows, widths=[4.0, 5.2, 6.8])

    doc.add_heading("八 销量热度和评价数据", level=1)
    p = doc.add_paragraph("这两类数据不是价格分析的最低必需项，但能明显增强偏好分析和竞品排序。平台页面上的“已售”“评价数”“热度”等指标口径可能不同，因此必须记录指标类型和观察时间。")
    rows = [
        ("metric_name", "指标名称", "sales_count、review_count、favorite_count、rank 等"),
        ("metric_value", "指标值", "保存数值，不把“10万+”直接当作精确值"),
        ("metric_type", "可信度类型", "actual、estimated 或 proxy"),
        ("metric_period", "统计周期", "daily、monthly、cumulative 或 unknown"),
        ("observed_at", "观察时间", "北京时间日期时间"),
        ("rating_score", "评分", "记录原始分制，并可换算为 5 分制"),
        ("positive_rate", "好评率", "使用 0 至 1 的小数"),
        ("review_count", "评价数量", "无法确认精确数量时同时保存原始文本"),
        ("keyword", "评价关键词", "只保存脱敏后的汇总关键词"),
        ("keyword_count", "关键词频次", "不得包含用户昵称、账号或联系方式"),
    ]
    add_table(doc, ["字段", "含义", "记录要求"], rows, widths=[4.0, 4.3, 7.7])

    doc.add_heading("九 数据来源和合规记录", level=1)
    p = doc.add_paragraph()
    p.add_run("允许的获取方式。 ").bold = True
    p.add_run("官方开放平台接口、商家后台导出、经授权的联盟接口、许可明确的开放数据集、品牌官网公开规格，以及记录了采集日期和页面地址的人工整理。")
    p = doc.add_paragraph()
    p.add_run("禁止的获取方式。 ").bold = True
    p.add_run("绕过登录、验证码、访问限制或平台技术措施的批量抓取；来源不明的数据包；包含个人信息的评价明细；把推算值写成实际销量；把演示数据标为真实数据。")
    rows = [
        ("source_batch_id", "批次唯一编号", "JD_20261009_01"),
        ("source_platform", "来源平台或网站", "京东"),
        ("acquisition_method", "API、后台导出、开放数据、人工记录", "官方 API"),
        ("source_location", "接口名称、菜单路径或公开页面", "商品查询接口"),
        ("authorization_type", "授权或许可依据", "应用授权"),
        ("exported_at", "导出或采集时间", "2026-10-09 10:00:00"),
        ("collector", "采集人或责任人", "项目成员姓名"),
        ("license_or_limits", "使用范围和限制", "仅用于毕业设计研究"),
        ("file_sha256", "原始文件哈希值", "用于证明文件未被替换"),
        ("notes", "口径、缺失或异常说明", "销量字段为累计评价数代理"),
    ]
    add_table(doc, ["字段", "说明", "示例"], rows, widths=[4.2, 7.2, 4.6])

    doc.add_heading("十 文件组织和命名", level=1)
    p = doc.add_paragraph("原始文件保持不变，放入项目的 data/raw 目录。清洗后的文件放入 data/processed，导入数据库的结果由系统管理。建议目录如下。")
    for item in (
        "data/raw/jd/products_20261009.xlsx",
        "data/raw/tmall/products_20261009.xlsx",
        "data/raw/provenance_20261009.xlsx",
        "data/processed/products_clean.csv",
        "data/processed/product_specs_clean.csv",
        "data/processed/price_snapshots_clean.csv",
    ):
        add_bullet(doc, item)
    p = doc.add_paragraph()
    p.add_run("编码和格式。 ").bold = True
    p.add_run("CSV 使用 UTF-8 编码；日期时间采用北京时间；金额保留两位小数；布尔值使用 true、false 或空值；缺失值留空，不使用 0、未知、无或横线混代。")

    doc.add_heading("十一 数据清洗规则", level=1)
    for text in (
        "以 platform 和 platform_product_id 识别平台内唯一商品；同一页面的不同规格若价格和参数差异明显，应按 SKU 拆分。",
        "品牌名称建立标准表，例如“罗技”“Logitech”“LOGITECH”统一到一个标准品牌名，同时保留原始文本供复核。",
        "价格字段只保留数字和小数点，所有人民币价格统一为元；满减后的价格只有在条件明确时才写入 coupon_price。",
        "重量统一为克，回报率统一为 Hz，触发行程统一为毫米，保修期统一为月。",
        "重复快照按商品键和 collected_at 检查；若同一时间存在多条记录，保留来源更完整的一条并记录处理原因。",
        "异常价格需回查原页面或原始文件，不得仅凭经验修改。无法核实的记录应标记为待复核或排除。",
        "原始数据、清洗脚本和清洗后数据分别保存，确保任何分析结果都能追溯到来源记录。",
    ):
        add_number(doc, text)

    doc.add_heading("十二 交付验收标准", level=1)
    rows = [
        ("□", "样本数量", "唯一真实商品不少于 80 个"),
        ("□", "品类覆盖", "鼠标和键盘各不少于 30 个"),
        ("□", "品牌覆盖", "至少 5 个品牌，品牌字段已标准化"),
        ("□", "平台覆盖", "至少 2 个合规来源平台"),
        ("□", "价格历史", "同一商品至少 3 期快照，推荐 8 至 12 期"),
        ("□", "核心完整性", "平台、商品 ID、品类、品牌、名称、售价和时间完整率不低于 95%"),
        ("□", "规格可用性", "主要规格字段能够支持偏好分组和竞品对比"),
        ("□", "来源可追溯", "每个数据文件或批次均有 provenance 记录"),
        ("□", "隐私合规", "不包含买家昵称、账号、地址、电话或其他个人信息"),
        ("□", "演示隔离", "模拟数据与真实数据使用明确标记并可独立排除"),
        ("□", "格式通过", "CSV 或 XLSX 能正常读取，字段名与类型符合本规范"),
        ("□", "抽样复核", "随机抽取至少 10 个商品，价格和规格与来源记录一致"),
    ]
    add_table(
        doc,
        ["检查", "项目", "通过条件"],
        rows,
        widths=[1.4, 3.7, 10.9],
        alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
    )

    doc.add_heading("附录 最小可交付字段清单", level=1)
    p = doc.add_paragraph("如果当前只能获得一张表，至少应包含以下字段。该结构可直接映射到现有项目的数据导入模板。")
    code = (
        "platform, platform_product_id, category, brand, product_name, model, shop_name, "
        "product_url, list_price, sale_price, coupon_price, promotion_text, collected_at, "
        "connection_type, weight_g, polling_rate_hz, dpi_max, sensor_model, button_count, "
        "switch_type, switch_name, layout_size, key_count, actuation_mm, hot_swappable, rgb, "
        "warranty_months, source_batch_id"
    )
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.7)
    p.paragraph_format.right_indent = Cm(0.7)
    run = p.add_run(code)
    run.font.name = "Consolas"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
    run.font.size = Pt(9)

    footer = section.footer
    footer_p = footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer_p.add_run("电竞外设竞品分析项目数据集规范  |  第 ")
    footer_run.font.name = "宋体"
    footer_run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    footer_run.font.size = Pt(9)
    footer_run.font.color.rgb = GRAY
    add_field(footer_p, "PAGE")
    tail = footer_p.add_run(" 页")
    tail.font.name = "宋体"
    tail._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    tail.font.size = Pt(9)
    tail.font.color.rgb = GRAY

    doc.core_properties.title = "电竞外设竞品分析项目数据集采集与交付规范"
    doc.core_properties.subject = "电竞键盘和电竞鼠标真实结构化数据的采集字段与验收标准"
    doc.core_properties.author = ""
    doc.core_properties.keywords = "电竞外设, 数据集, 京东, 天猫, 价格分析, Streamlit"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_document()
