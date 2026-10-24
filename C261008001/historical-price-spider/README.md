# 历史价格与商品页面采集适配器

本目录只保留项目正在使用的两个采集入口：

- `main.py`：通过慢慢买官方查询页面取得京东、淘宝或天猫历史价格。
- `collect_jd_research_data.mjs`：从已登录京东页面取得公开商品属性、评价汇总和脱敏评论样本。
- `collect_jd_stratified_reviews.mjs`：按好评、中评、差评抽取公开评论，排除昵称、头像、地区、设备信息、图片和商家回复。
- `discover_tmall_candidates.mjs`：在已登录淘宝搜索页为未配对的京东型号发现天猫候选链接，候选仍需经过型号与慢慢买标题二次核验。

先运行 `start_manmanbuy_edge.ps1`，在专用 Edge 窗口中手动登录需要的平台，并保持窗口开启。脚本连接 `http://127.0.0.1:9222`，不读取或输出账号密码和 Cookie。

从项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\import_manmanbuy_history.py --provider official-web

node historical-price-spider\collect_jd_research_data.mjs `
  data\manmanbuy\esports_targets.csv `
  data\raw\jd_research_20261009.json

node historical-price-spider\collect_jd_stratified_reviews.mjs `
  data\manmanbuy\esports_targets.csv `
  data\raw\jd_reviews_stratified.json

node historical-price-spider\discover_tmall_candidates.mjs `
  data\raw\tmall_candidates.json
```

分层评论脚本默认每次请求至少间隔4秒，并在连续两个商品的全部评分组失败后停止。可通过
`JD_REVIEW_START_INDEX` 和 `JD_REVIEW_TARGET_LIMIT` 分批执行。只对项目目标清单进行低频采集；
遇到验证码、登录失效、系统繁忙、403 或其他频控时停止，不绕过平台限制。
