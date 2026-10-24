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
const outputPath = process.argv[3] ?? "data/raw/jd_reviews_stratified.json";
const cdpUrl = process.argv[4] ?? "http://127.0.0.1:9222";
const startIndex = Math.max(0, Number.parseInt(process.env.JD_REVIEW_START_INDEX ?? "0", 10));
const targetLimit = Math.max(0, Number.parseInt(process.env.JD_REVIEW_TARGET_LIMIT ?? "0", 10));
const requestDelayMs = Math.max(1000, Number.parseInt(process.env.JD_REVIEW_REQUEST_DELAY_MS ?? "4000", 10));
const productDelayMs = Math.max(1000, Number.parseInt(process.env.JD_REVIEW_PRODUCT_DELAY_MS ?? "5000", 10));
const scoreGroups = [
  { score: 1, label: "negative" },
  { score: 2, label: "neutral" },
  { score: 3, label: "positive" },
];

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
  return lines.map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(headers.map((key, index) => [key, values[index] ?? ""]));
  });
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

const enabledTargets = parseCsv(await fs.readFile(targetsPath, "utf8"))
  .filter((row) => row.platform === "jd" && /^(true|1|yes|是)$/i.test(row.enabled));
const targets = enabledTargets.slice(startIndex, targetLimit > 0 ? startIndex + targetLimit : undefined);
const browser = await chromium.connectOverCDP(cdpUrl);
const context = browser.contexts()[0];
const page = await context.newPage();
page.setDefaultTimeout(12000);
const products = [];
const errors = [];
let consecutiveFullFailures = 0;
let rateLimited = false;

async function saveResult() {
  const result = {
    source: "jd_public_review_page_stratified",
    collected_at: new Date().toISOString(),
    target_count: targets.length,
    start_index: startIndex,
    success_count: products.filter((item) => item.comments.length > 0).length,
    error_count: errors.length,
    privacy: "Nickname, user ID, avatar, device information, region, images and replies are excluded; review IDs are hashed.",
    sampling_note: "One public page per score group (negative, neutral, positive), up to 10 reviews per group and product.",
    products,
    errors,
  };
  await fs.mkdir(outputPath.replace(/[\\/][^\\/]+$/, ""), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(result, null, 2), "utf8");
  return result;
}

for (let index = 0; index < targets.length; index += 1) {
  const target = targets[index];
  const comments = [];
  const groupErrors = [];
  for (const group of scoreGroups) {
    const params = new URLSearchParams({
      productId: target.sku,
      score: String(group.score),
      sortType: "5",
      page: "0",
      pageSize: "10",
      isShadowSku: "0",
      fold: "1",
    });
    const url = `https://club.jd.com/comment/productPageComments.action?${params}`;
    try {
      const response = await page.goto(url, { waitUntil: "domcontentloaded", timeout: 45000 });
      if (!response?.ok()) throw new Error(`HTTP ${response?.status() ?? "unknown"}`);
      const body = await response.text();
      if (body.includes('系统繁忙')) throw new Error('rate_limited:系统繁忙');
      const payload = JSON.parse(body);
      for (const item of payload.comments ?? []) {
        const content = redact(item.content);
        if (!content) continue;
        comments.push({
          source_review_id: hashReviewId(item.id ?? item.guid ?? `${target.sku}-${item.creationTime}-${content}`),
          source_product_id: String(item.referenceId ?? item.productId ?? target.sku),
          sample_group: group.label,
          rating: Number(item.score) || group.score,
          reviewed_at: item.creationTime || null,
          content,
          variant: [item.productColor, item.productSize, item.referenceName].filter(Boolean),
        });
      }
    } catch (error) {
      groupErrors.push({ score: group.score, error: String(error?.message ?? error) });
      if (/rate_limited|HTTP (403|429)|验证码/.test(String(error?.message ?? error))) {
        rateLimited = true;
        break;
      }
    }
    await page.waitForTimeout(requestDelayMs);
  }
  const uniqueComments = comments.filter(
    (item, position, items) => items.findIndex((candidate) => candidate.source_review_id === item.source_review_id) === position,
  );
  products.push({
    sku: target.sku,
    category: target.category,
    brand: target.brand,
    target_name: target.product_name,
    product_url: target.product_url,
    collected_at: new Date().toISOString(),
    attributes: [],
    review_summary: null,
    review_sampling: {
      method: "public_review_page_stratified_by_score",
      score_groups: scoreGroups,
      page_size_per_group: 10,
      errors: groupErrors,
    },
    comments: uniqueComments,
  });
  if (groupErrors.length) errors.push({ sku: target.sku, errors: groupErrors });
  consecutiveFullFailures = groupErrors.length === scoreGroups.length
    ? consecutiveFullFailures + 1
    : 0;
  process.stderr.write(
    `[${index + 1}/${targets.length}] ${target.sku}: reviews=${uniqueComments.length}, group_errors=${groupErrors.length}`
      + `${groupErrors[0] ? `, first_error=${groupErrors[0].error}` : ""}\n`,
  );
  await saveResult();
  if (rateLimited) {
    process.stderr.write('平台返回频控信号，已保存当前结果并停止请求。\n');
    break;
  }
  if (consecutiveFullFailures >= 2) {
    process.stderr.write("连续两个商品的全部评分组失败，已停止以避免触发更严格的频控。\n");
    break;
  }
  if (index < targets.length - 1) await page.waitForTimeout(productDelayMs);
}

const result = await saveResult();
process.stdout.write(JSON.stringify({
  output: outputPath,
  target_count: targets.length,
  success_count: result.success_count,
  error_count: errors.length,
  review_count: products.reduce((sum, item) => sum + item.comments.length, 0),
}));
await page.close().catch(() => {});
process.exit(errors.length ? 1 : 0);
