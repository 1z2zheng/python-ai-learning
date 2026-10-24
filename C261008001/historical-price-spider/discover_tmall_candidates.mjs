import fs from "node:fs/promises";

let chromium;
try {
  ({ chromium } = await import("playwright"));
} catch {
  ({ chromium } = await import(
    "file:///C:/Users/29148/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs"
  ));
}

const outputPath = process.argv[2] ?? "data/raw/tmall_candidates.json";
const cdpUrl = process.argv[3] ?? "http://127.0.0.1:9222";
const searches = [
  ["100041149481", "mouse", "罗技", "G502 X有线", "罗技 G502 X 有线"],
  ["100105751409", "mouse", "雷蛇", "炼狱蝰蛇V3极速版", "雷蛇 炼狱蝰蛇 V3 极速版"],
  ["100102201664", "mouse", "ROG", "月刃2 ACE", "ROG 月刃2 ACE"],
  ["100058570772", "mouse", "ROG", "龙鳞ACE", "ROG 龙鳞 ACE"],
  ["100010255665", "mouse", "罗技", "GPW二代", "罗技 GPW 二代"],
  ["10092558210376", "keyboard", "罗技", "PRO X TKL", "罗技 PRO X TKL 键盘"],
  ["100068436132", "keyboard", "雷蛇", "猎魂光蛛V3专业版", "雷蛇 猎魂光蛛 V3 专业版"],
  ["100068436130", "keyboard", "雷蛇", "猎魂光蛛V3专业迷你版", "雷蛇 猎魂光蛛 V3 专业迷你版"],
  ["100070965890", "keyboard", "达尔优", "EK75", "达尔优 EK75"],
  ["100200260117", "keyboard", "达尔优", "COOL68", "达尔优 COOL68"],
  ["100197744867", "keyboard", "VGN", "V98Pro V3", "VGN V98Pro V3"],
];

function canonicalCandidate(item) {
  const url = new URL(item.href);
  const itemId = url.searchParams.get("id");
  if (!itemId || url.hostname !== "detail.tmall.com") return null;
  const text = item.text.replace(/\s+/g, " ").trim();
  return {
    item_id: itemId,
    product_url: `https://detail.tmall.com/item.htm?id=${itemId}`,
    listing_text: text,
    official_signal: /官方旗舰店|品牌旗舰|天猫自营|官方店/.test(text),
  };
}

const browser = await chromium.connectOverCDP(cdpUrl);
const context = browser.contexts()[0];
const page = await context.newPage();
page.setDefaultTimeout(15000);
const results = [];

for (let index = 0; index < searches.length; index += 1) {
  const [jdSku, category, brand, model, query] = searches[index];
  const searchUrl = `https://s.taobao.com/search?q=${encodeURIComponent(query)}&tab=tmall`;
  try {
    await page.goto(searchUrl, { waitUntil: "domcontentloaded", timeout: 45000 });
    await page.waitForTimeout(6500);
    const bodyText = await page.locator("body").innerText();
    if (/验证码|滑块验证|请登录/.test(bodyText.slice(0, 1500))) {
      throw new Error("淘宝搜索要求登录或验证")
    }
    const links = await page.locator("a").evaluateAll((anchors) => anchors.map((anchor) => ({
      text: (anchor.innerText || "").trim().slice(0, 600),
      href: anchor.href,
    })).filter((item) => item.href.includes("item.htm")));
    const candidates = links
      .map(canonicalCandidate)
      .filter(Boolean)
      .filter((item, position, items) => items.findIndex((candidate) => candidate.item_id === item.item_id) === position)
      .slice(0, 12);
    results.push({ jd_sku: jdSku, category, brand, model, query, candidates, error: null });
    process.stderr.write(`[${index + 1}/${searches.length}] ${model}: ${candidates.length} 个天猫候选\n`);
  } catch (error) {
    results.push({ jd_sku: jdSku, category, brand, model, query, candidates: [], error: String(error?.message ?? error) });
    process.stderr.write(`[${index + 1}/${searches.length}] ${model}: failed\n`);
  }
  if (index < searches.length - 1) await page.waitForTimeout(2500);
}

const payload = {
  source: "authenticated_taobao_search",
  collected_at: new Date().toISOString(),
  query_count: searches.length,
  success_count: results.filter((item) => !item.error).length,
  note: "Candidates require exact-model validation before collection or import.",
  results,
};
await fs.mkdir(outputPath.replace(/[\\/][^\\/]+$/, ""), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify(payload, null, 2), "utf8");
console.log(JSON.stringify({
  output: outputPath,
  query_count: searches.length,
  success_count: payload.success_count,
  candidate_count: results.reduce((sum, item) => sum + item.candidates.length, 0),
}));
await page.close().catch(() => {});
process.exit(payload.success_count === searches.length ? 0 : 1);
