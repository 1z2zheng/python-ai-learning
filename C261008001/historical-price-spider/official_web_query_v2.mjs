let chromium;
try {
  ({ chromium } = await import("playwright"));
} catch {
  ({ chromium } = await import(
    "file:///C:/Users/29148/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs"
  ));
}

const productUrl = process.argv[2];
const cdpUrl = process.argv[3] ?? "http://127.0.0.1:9222";
if (!productUrl) throw new Error("缺少商品链接");

const finish = (payload, exitCode = 0) => {
  process.stdout.write(JSON.stringify(payload));
  process.exit(exitCode);
};

const browser = await chromium.connectOverCDP(cdpUrl);
const pages = browser.contexts().flatMap((context) => context.pages());
let page = pages.find((item) => item.url().includes("tool.manmanbuy.com/HistoryLowest.aspx"));
if (!page) page = await browser.contexts()[0].newPage();

const responses = [];
page.on("response", async (response) => {
  if (!response.url().includes("/api.ashx")) return;
  try {
    const body = await response.json();
    if (body?.code === 0 && body?.data && !Array.isArray(body.data)) {
      responses.push(body);
    }
  } catch {
    // Ignore auxiliary API calls and non-JSON responses.
  }
});

await page.bringToFront();
await page.goto(
  `https://tool.manmanbuy.com/HistoryLowest.aspx?url=${encodeURIComponent(productUrl)}`,
  { waitUntil: "domcontentloaded", timeout: 45000 },
);
await page.locator("#historykey").fill(productUrl);

const loginRequired = await page.locator(".PClogin").evaluateAll((items) =>
  items.some((item) => getComputedStyle(item).display !== "none")
);
if (loginRequired) finish({ status: "login_required" }, 2);

const waitForResponse = async (timeoutMs) => {
  const deadline = Date.now() + timeoutMs;
  while (!responses.length && Date.now() < deadline) await page.waitForTimeout(250);
};

await waitForResponse(30000);
if (!responses.length) {
  await page.locator("#searchHistory").click();
  await waitForResponse(30000);
}

if (responses.length) finish({ status: "ok", data: responses.at(-1).data });

const state = await page.evaluate(() => ({
  text: (document.body?.innerText ?? "").slice(0, 2000),
  trendVisible: (() => {
    const element = document.querySelector(".trend-box");
    return element ? getComputedStyle(element).display !== "none" : false;
  })(),
}));
finish({ status: "timeout", ...state }, 3);
