import crypto from "node:crypto";
import fs from "node:fs/promises";

let chromium;
try {
  ({ chromium } = await import("playwright"));
} catch {
  ({ chromium } = await import(
    "file:///C:/Users/29148/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs"
  ));
}

const targetsPath = process.argv[2] ?? "data/manmanbuy/esports_targets.csv";
const outputPath = process.argv[3] ?? "data/raw/jd_research_20261009.json";
const cdpUrl = process.argv[4] ?? "http://127.0.0.1:9222";

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"';
        index += 1;
      } else quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(value);
      value = "";
    } else value += char;
  }
  values.push(value);
  return values;
}

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean);
  const headers = parseCsvLine(lines.shift());
  return lines.map((line) => Object.fromEntries(headers.map((key, index) => [key, parseCsvLine(line)[index] ?? ""])));
}

function redact(text) {
  return String(text ?? "")
    .replace(/[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}/g, "[邮箱已脱敏]")
    .replace(/(?<!\d)1[3-9]\d{9}(?!\d)/g, "[手机号已脱敏]")
    .replace(/(?:微信|vx|V信)[:：\s]*[A-Za-z][-_A-Za-z0-9]{5,19}/gi, "[联系方式已脱敏]")
    .trim();
}

function hashReviewId(value) {
  return crypto.createHash("sha256").update(`jd:${value}`).digest("hex").slice(0, 24);
}

const targetLimit = Number.parseInt(process.env.JD_TARGET_LIMIT ?? "0", 10);
const productDelayMs = Math.max(5000, Number.parseInt(process.env.JD_PRODUCT_DELAY_MS ?? "7000", 10));
const startIndex = Math.max(0, Number.parseInt(process.env.JD_TARGET_START_INDEX ?? "0", 10));
const enabledTargets = parseCsv(await fs.readFile(targetsPath, "utf8"))
  .filter((row) => row.platform === "jd" && /^(true|1|yes|是)$/i.test(row.enabled));
const targets = enabledTargets.slice(startIndex, targetLimit > 0 ? startIndex + targetLimit : undefined);
const browser = await chromium.connectOverCDP(cdpUrl);
const context = browser.contexts()[0];
const page = await context.newPage();
page.setDefaultTimeout(8000);
const products = [];
const errors = [];

for (let index = 0; index < targets.length; index += 1) {
  const target = targets[index];
  let wareData = null;
  let commentData = null;
  const responseHandler = async (response) => {
    const url = response.url();
    if (!url.includes("api.m.jd.com")) return;
    try {
      if (url.includes("functionId=pc_detailpage_wareBusiness")) wareData = await response.json();
      if (url.includes("functionId=getLegoWareDetailComment")) commentData = await response.json();
    } catch {}
  };
  page.on("response", responseHandler);
  try {
    await page.goto(target.product_url, { waitUntil: "domcontentloaded", timeout: 45000 });
    const deadline = Date.now() + 18000;
    while ((!wareData || !commentData) && Date.now() < deadline) await page.waitForTimeout(250);
    if (!wareData) throw new Error("商品详情接口未返回");
    const comments = (commentData?.commentInfoList ?? [])
      .map((item) => ({
        source_review_id: hashReviewId(item.commentId ?? `${target.sku}-${item.commentDate}-${item.commentData}`),
        source_product_id: item.referenceId ? String(item.referenceId) : null,
        rating: Number(item.commentScore) || null,
        reviewed_at: item.commentDate || null,
        content: redact(item.commentData),
        variant: (item.wareAttribute ?? []).map((entry) => entry?.text ?? entry?.value ?? entry).filter(Boolean),
      }))
      .filter((item) => item.content);
    products.push({
      sku: target.sku,
      category: target.category,
      brand: target.brand,
      target_name: target.product_name,
      product_url: target.product_url,
      collected_at: new Date().toISOString(),
      title: wareData?.skuHeadVO?.skuTitle ?? target.product_name,
      title_from_source: Boolean(wareData?.skuHeadVO?.skuTitle),
      attributes: wareData?.productAttributeVO?.attributes ?? [],
      review_summary: commentData ? {
        all_count: commentData.allCnt ?? null,
        all_count_text: commentData.allCntStr ?? null,
        good_count: commentData.goodCnt ?? null,
        normal_count: commentData.normalCnt ?? null,
        bad_count: commentData.badCnt ?? null,
        good_rate_text: commentData.goodRate ?? null,
        tags: (commentData.tagStatisticsinfoList ?? []).map(({ name, count, countText }) => ({ name, count, countText })),
      } : null,
      comments,
    });
    process.stderr.write(`[${index + 1}/${targets.length}] ${target.sku}: specs=${wareData?.productAttributeVO?.attributes?.length ?? 0}, reviews=${comments.length}\n`);
  } catch (error) {
    const errorText = String(error?.message ?? error);
    errors.push({ sku: target.sku, error: errorText });
    process.stderr.write(`[${index + 1}/${targets.length}] ${target.sku}: failed: ${errorText}\n`);
  } finally {
    page.off("response", responseHandler);
  }
  if (index < targets.length - 1) await page.waitForTimeout(productDelayMs);
}

const result = {
  source: "jd_authenticated_product_page",
  collected_at: new Date().toISOString(),
  target_count: targets.length,
  success_count: products.length,
  error_count: errors.length,
  privacy: "User nickname, user ID, avatar and region are excluded; review IDs are hashed.",
  products,
  errors,
};
await fs.mkdir(outputPath.replace(/[\\/][^\\/]+$/, ""), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify(result, null, 2), "utf8");
process.stdout.write(JSON.stringify({
  output: outputPath,
  target_count: targets.length,
  success_count: products.length,
  error_count: errors.length,
  review_count: products.reduce((sum, item) => sum + item.comments.length, 0),
}));
await page.close().catch(() => {});
process.exit(errors.length ? 1 : 0);
