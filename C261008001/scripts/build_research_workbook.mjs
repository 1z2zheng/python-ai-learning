import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "file:///C:/Users/29148/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const inputPath = process.argv[2] ?? "data/raw/official_catalog_20261009.csv";
const outputPath = process.argv[3] ?? "output/电竞键鼠真实研究数据_20261009.xlsx";
const previewDir = process.argv[4] ?? "output/previews_20261009";

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ""; }
    else if (ch === '\n') { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  return rows;
}

function columnName(index) {
  let value = index + 1, result = "";
  while (value) { value -= 1; result = String.fromCharCode(65 + (value % 26)) + result; value = Math.floor(value / 26); }
  return result;
}

const raw = (await fs.readFile(inputPath, "utf8")).replace(/^\uFEFF/, "");
const parsed = parseCsv(raw);
const headers = parsed[0];
const numberFields = new Set(["list_price", "sale_price", "coupon_price", "weight_g", "polling_rate_hz", "dpi_max", "button_count", "key_count", "actuation_mm"]);
const records = parsed.slice(1).filter((r) => r.some((v) => v !== "")).map((row) =>
  headers.map((header, i) => {
    const value = row[i] ?? "";
    if (numberFields.has(header) && value !== "") return Number(value);
    return value;
  })
);
const idx = Object.fromEntries(headers.map((h, i) => [h, i]));
const priced = records.filter((r) => typeof r[idx.sale_price] === "number" && Number.isFinite(r[idx.sale_price]));
const brands = [...new Set(records.map((r) => r[idx.brand]))].sort((a, b) => a.localeCompare(b, "zh-CN"));
const sources = [...new Set(records.map((r) => r[idx.source_site]))].sort();
const lastCol = columnName(headers.length - 1);
const lastRow = records.length + 1;

const wb = Workbook.create();
const summary = wb.worksheets.add("数据概览");
const catalog = wb.worksheets.add("商品目录");
const pricedSheet = wb.worksheets.add("已标价样本");
const notes = wb.worksheets.add("数据说明");
const font = "Arial";
const navy = "#17365D", blue = "#D9EAF7", pale = "#F3F6F9", amber = "#FFF2CC";

summary.showGridLines = false;
summary.getRange("A2:H2").merge();
summary.getRange("A2").values = [["电竞鼠标、键盘真实研究数据概览"]];
summary.getRange("A2:H2").format = { font: { name: font, size: 16, bold: true, color: navy } };
summary.getRange("A3:H3").format.borders = { bottom: { style: "thin", color: "#9EADBA" } };
summary.getRange("A4:B9").values = [["指标", "结果"], ["真实商品记录", null], ["电竞鼠标", null], ["电竞键盘", null], ["品牌数", null], ["明确售价记录", null]];
summary.getRange("B5:B9").formulas = [[`=COUNTA('商品目录'!$A$2:$A$${lastRow})`], [`=COUNTIF('商品目录'!$C$2:$C$${lastRow},"mouse")`], [`=COUNTIF('商品目录'!$C$2:$C$${lastRow},"keyboard")`], [`=COUNTA(A12:A${11 + brands.length})`], [`=COUNT('商品目录'!$L$2:$L$${lastRow})`]];
summary.getRange("A4:B4").format = { fill: navy, font: { name: font, bold: true, color: "#FFFFFF" } };
summary.getRange("A5:B9").format.font = { name: font, size: 11 };
summary.getRange("A11:B11").values = [["品牌", "样本数"]];
summary.getRange(`A12:A${11 + brands.length}`).values = brands.map((v) => [v]);
summary.getRange(`B12:B${11 + brands.length}`).formulas = brands.map((_, i) => [`=COUNTIF('商品目录'!$D$2:$D$${lastRow},A${12 + i})`]);
summary.getRange("D11:E11").values = [["品类", "样本数"]];
summary.getRange("D12:E13").values = [["电竞鼠标", null], ["电竞键盘", null]];
summary.getRange("E12:E13").formulas = [["=B6"], ["=B7"]];
summary.getRange("G4:H7").values = [["研究提示", "说明"], ["价格覆盖", null], ["适用范围", "品牌官网公开目录、型号与部分规格的截面研究"], ["限制", "未明确展示售价的商品保持为空，不推断、不填 0"]];
summary.getRange("H5").formulas = [[`=TEXT(B9/B5,"0.0%")&"（"&B9&"/"&B5&"）"`]];
for (const range of ["A11:B11", "D11:E11", "G4:H4"]) summary.getRange(range).format = { fill: navy, font: { name: font, bold: true, color: "#FFFFFF" } };
summary.getRange("G5:H7").format = { fill: amber, font: { name: font, size: 10 }, wrapText: true };
summary.getRange("A4:B9").format.borders = { preset: "inside", style: "thin", color: "#D9E1E8" };
summary.getRange("A11:B17").format.borders = { preset: "inside", style: "thin", color: "#D9E1E8" };
summary.getRange("D11:E13").format.borders = { preset: "inside", style: "thin", color: "#D9E1E8" };
summary.getRange("A:H").format.font = { name: font };
summary.getRange("A:A").format.columnWidth = 22; summary.getRange("B:B").format.columnWidth = 14;
summary.getRange("D:D").format.columnWidth = 18; summary.getRange("E:E").format.columnWidth = 14;
summary.getRange("G:G").format.columnWidth = 16; summary.getRange("H:H").format.columnWidth = 48;

function populateDataSheet(sheet, data, tableName) {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${lastCol}${data.length + 1}`).values = [headers, ...data];
  sheet.getRange(`A1:${lastCol}1`).format = { fill: navy, font: { name: font, bold: true, color: "#FFFFFF" }, wrapText: true, horizontalAlignment: "center", verticalAlignment: "center" };
  sheet.getRange(`A2:${lastCol}${data.length + 1}`).format = { font: { name: font, size: 9 }, verticalAlignment: "top" };
  sheet.freezePanes.freezeRows(1); sheet.freezePanes.freezeColumns(4);
  const table = sheet.tables.add(`A1:${lastCol}${data.length + 1}`, true, tableName);
  table.style = "TableStyleMedium2";
  for (let c = 0; c < headers.length; c += 1) sheet.getRange(`${columnName(c)}:${columnName(c)}`).format.columnWidth = 14;
  for (const h of ["model", "product_name", "shop_name"]) sheet.getRange(`${columnName(idx[h])}:${columnName(idx[h])}`).format.columnWidth = 24;
  for (const h of ["product_url", "image_url", "source_catalog_url"]) sheet.getRange(`${columnName(idx[h])}:${columnName(idx[h])}`).format.columnWidth = 34;
  sheet.getRange(`${columnName(idx.source_text)}:${columnName(idx.source_text)}`).format.columnWidth = 48;
  sheet.getRange(`${columnName(idx.source_text)}2:${columnName(idx.source_text)}${data.length + 1}`).format.wrapText = true;
  for (const h of ["list_price", "sale_price", "coupon_price"]) sheet.getRange(`${columnName(idx[h])}2:${columnName(idx[h])}${data.length + 1}`).format.numberFormat = "0.00";
  sheet.getRange(`${columnName(idx.collected_at)}2:${columnName(idx.collected_at)}${data.length + 1}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
}

populateDataSheet(catalog, records, "OfficialCatalogTable");
populateDataSheet(pricedSheet, priced, "PricedSamplesTable");

notes.showGridLines = false;
notes.getRange("A2:F2").merge(); notes.getRange("A2").values = [["数据口径、来源与复现说明"]];
notes.getRange("A2:F2").format = { font: { name: font, size: 16, bold: true, color: navy } };
notes.getRange("A4:B10").values = [
  ["项目", "说明"],
  ["采集日期", "2026-10-09（Asia/Shanghai）"],
  ["数据性质", "品牌官网公开目录的真实截面数据；每行代表一个公开商品或系列页面"],
  ["合规方式", "仅请求 robots.txt 允许的公开页面，低频访问；不登录、不使用账号 Cookie、不绕过验证码或风控"],
  ["价格规则", "仅保留页面明确展示且可解析的售价；未展示价格一律为空"],
  ["用途", "适合品牌、品类、连接方式、规格覆盖与商品目录研究"],
  ["限制", "价格覆盖不足以直接完成跨平台价格趋势研究；还需取得京东/天猫合规授权数据及多时点快照"],
];
notes.getRange("A12:C12").values = [["来源代码", "品牌", "公开目录/配置来源"]];
const sourceRows = sources.map((source) => {
  const row = records.find((r) => r[idx.source_site] === source);
  return [source, row[idx.brand], row[idx.source_catalog_url]];
});
notes.getRange(`A13:C${12 + sourceRows.length}`).values = sourceRows;
for (const range of ["A4:B4", "A12:C12"]) notes.getRange(range).format = { fill: navy, font: { name: font, bold: true, color: "#FFFFFF" } };
notes.getRange("A5:B10").format = { font: { name: font, size: 10 }, wrapText: true, verticalAlignment: "top" };
notes.getRange(`A13:C${12 + sourceRows.length}`).format = { font: { name: font, size: 10 }, verticalAlignment: "top" };
notes.getRange("A:A").format.columnWidth = 22; notes.getRange("B:B").format.columnWidth = 64; notes.getRange("C:C").format.columnWidth = 72;
notes.freezePanes.freezeRows(4);

wb.recalculate();
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["数据概览", "商品目录", "已标价样本", "数据说明"]) {
  const preview = await wb.render({ sheetName, autoCrop: "all", scale: sheetName === "数据概览" ? 1.2 : 0.65, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const check = await wb.inspect({ kind: "formula", sheetId: "数据概览", range: "A1:H20", maxChars: 5000 });
console.log(check.ndjson ?? check);
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, records: records.length, priced: priced.length, brands: brands.length, previews: previewDir }));
