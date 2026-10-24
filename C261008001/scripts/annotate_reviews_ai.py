"""Explicit per-record AI readings, separate from independent human labels."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "review_analysis"
DIMENSIONS = {"F": "手感体验", "R": "响应性能", "C": "舒适度", "D": "耐用性", "A": "外观设计", "V": "性价比"}
POLARITY = {"+": "正向", "-": "负向", "=": "中性", "*": "混合"}
OVERALL = {"P": "正向", "N": "负向", "O": "中性", "M": "混合"}

# Labels derive from reading the full supplied texts; no rating fallback or
# project classifier is called. These remain AI labels pending human review.
READINGS = """
1 P F+ C* |反馈清脆、操作方便；声音大导致办公适用性受限；未评价响应性能
2 P F+ R+ C+ A+ |按键顺滑、不累手、低延迟和灯效均获肯定；包装不作为耐用证据
3 P F+ R+ C+ D+ A+ V+ |手感与跟手体验获肯定，清理便利；值得入手属于价值肯定
4 P A+ D= C+ R+ |耐用尚待检验，不将品牌信任当作已验证耐久
5 P A+ F+ R+ C* |稳定且握持舒适；重量较重但手感更稳，舒适维度保留混合
6 P F+ C+ R+ D+ V+ |手感、定位、线材和价格价值明确正向
7 P F+ C+ |轻巧、按键清脆；手型适配建议仅针对特定握法
8 P F+ C+ |更轻、总体体验满意；模具变小未明确抱怨
9 P F+ C+ R+ D+ |长时不累、防滑、响应与续航均获肯定
10 P A+ F+ R+ C+ D+ |旧GPW左键故障不归于目标ROG，品控肯定但未游戏实测
11 P F+ C+ D+ A= |涂层和做工获肯定；仅说明灯效可调，外观情感中性
12 P D+ R+ A+ C+ |质保后耐久属于预期肯定；三模便携和握持获肯定
13 P A+ R+ C+ D+ |原文肯定响应、握持及品牌保障，耐用属于主观预期
14 P A+ F+ C+ R+ |外形、轻量与响应明确肯定
15 P F+ C+ R+ A+ D+ |手感、轻便、灵敏、贴纸和做工获肯定
16 P F+ R+ C+ A+ V+ |键鼠套装综合评价，需核验具体型号归属
17 P F+ C+ R+ A+ D+ V+ |文本明确提及GPW3，应复核与目标型号是否一致
18 P A+ C+ F+ R+ |GPW3版本描述，归属待核验
19 P F+ R+ D+ |安静舒适、切换稳定，产品质感正向；包装单独排除
20 P F+ C+ A+ |紧凑布局和媒体键便利，外观手感双正向
21 P F+ V+ D+ C+ |手感值价、续航长、携带方便
22 P F+ R+ C+ D+ A+ V+ |六类属性均有明确肯定
23 P F+ C+ D+ V+ |键位舒适且质量肯定；价格贵但认为值得
24 M F+ A+ V- |线性轴与白色可读性肯定，价格小贵；包装投诉属服务
25 P F+ C+ R+ |按压、声音、跟手肯定；最初普通观感未明确否定外观
26 M F= A* R+ |颜色可选性遗憾、灯光满意；茶青之间为触感描述
27 M F+ D* |当前键帽质感好；担心未来打油，旧款蜕皮不归当前款
28 P F+ C+ R+ |光轴与按钮便利，防误触功能获肯定
29 P A+ C+ D+ F+ R+ |腕托、材质、手感和触发设置获肯定
30 P F+ R+ D+ A+ |手感、响应、用料和颜值明确正向
31 P F= R= A= C= |大量规格与整体推荐，缺少逐维体验，维度情感保留中性
32 P F+ C+ |小巧及腕托舒适，包装完好不推断耐久
33 P D+ V+ |模板化泛化表达，仍有明确质量与价值评价
34 P F+ R+ D+ A+ |光轴游戏体验、键帽质感和白色外观获肯定
35 P F+ R+ A+ V+ |触发与调节便利、颜值、价值明确正向
36 P A+ R+ F+ C= D= |灯光不刺眼、反馈好；重量与材质描述不外推耐久
37 P D+ F+ A+ R+ C+ V+ |做工、回弹、灯效、响应及布局和价值肯定
38 P F+ R+ C+ A+ D+ |手感声音适用，汗手键帽获肯定
39 P F+ R+ A+ V+ C+ |手感、声音、灯光、价值及腕托获肯定
40 P A+ D+ R+ |不油、快捷操作和联动获肯定
41 P F+ R+ A+ D+ |回弹、动感听感、灯效和用料正向
42 P A+ R+ D= |不刺眼与多设备便利；充电款为客观属性
43 M F+ C* A+ R+ |整体满足需求但侧键不适应，保留舒适适用性混合
44 P F+ C+ R+ D= |旧键盘八年不坏不当作当前A98寿命证据
45 P D+ C+ A+ R+ |续航、模具、外观和低延迟均正向
46 P F+ A+ D+ V+ |按键反馈、颜色、做工和价值正向
47 P A+ C+ D= R+ |耐用明确未知；响应够用与布局获肯定
48 P F+ C+ R+ A+ D+ V+ |多属性明确肯定；隐喻归手感，声音适用归舒适
52 M F+ C+ A* R* |外观肯定但担心积灰；驱动及唤醒配对延迟投诉
53 P F+ R+ C+ D+ |定位、反馈、握持和线材明确正向
56 P F+ C+ R+ D+ |舒适、响应、续航与做工肯定
60 P F+ C+ R+ A+ D+ |重量握法、传感器、RGB及可维护性肯定
63 P A+ F+ C+ R+ D+ |定位、握法、涂层和续航获肯定
64 P F+ C+ R+ D+ A= |品牌泛化键鼠评价，缺少具体SKU；灯效只说可设置
65 P C+ F+ R+ |先前担心延迟被实际体验否定，不标负向
66 P F+ C+ R+ A+ D+ |GPW3及RAPID组合评价，具体型号待核验
70 M V* F- D+ |有优惠但价格高，缺音量旋钮不便，做工好
72 P D+ A+ F+ C+ |质量、颜值和安静手感肯定
78 P F+ V+ |光轴体验不错且实惠；旧设备寿命不归当前款
81 P R+ A+ C+ |游戏响应与RT/ST、白色和宿舍安静性获肯定
88 P F+ R+ C+ D+ A+ |轴体、连接、长时舒适及键帽外观正向
89 P F+ R+ C+ D+ A+ |回弹、无线延迟、布局、维护与透光性肯定
91 P F+ C+ R+ A+ D+ V+ |六维均有明确体验肯定
93 P F+ D+ C+ |静音适合家有小孩；额外A950不归当前键盘
96 M A+ C+ F* D+ R+ |整体正向但组合功能键不好记，交互手感维度混合
97 P F+ R+ D+ V+ C= |质量和价值肯定；分量足未明确舒适好坏
98 P A+ F+ R+ |外观灯效、轴体和打字流畅获肯定
99 P D+ F+ R+ A+ V+ C+ |做工、低杂音、响应、灯效和价格正向
100 P A+ F+ R+ |外观、大键无杂音和模式切换获肯定
101 P A+ D+ F+ R+ |配色、键帽、回弹和响应正向
102 P F+ C+ A+ D+ |轻触发、静音、灯光、续航防尘肯定
103 P A+ F+ R+ |与100文本重复，仍独立保留原始记录供复核
104 P D+ C+ A+ R+ V+ F+ |耐用、配列、响应与价值肯定；含泛化模板表达
105 P F+ C+ A+ R+ |紧凑与疲劳体验肯定；60%描述与目标98配列需核验
106 P C+ F+ R= D= A+ |握法和外观肯定；DPI可调、充电功能属客观描述
107 P R+ A+ V+ D= C+ |模式便利、外形和价格肯定；朋友键盘寿命不归该鼠标
108 P F+ C+ R+ D+ A+ V+ |六维均获明确肯定
229 N D- R- F- |旋钮故障、按键失灵及软件依赖影响使用
230 N D- F- C- |短期按键故障，侧键影响盲打
231 N D- R- F- |按键失灵连点及疑似二手；真实性属于用户怀疑，未核实
232 N D- F- V- |轴体不可自行更换与维修成本投诉；价格数字矛盾待复核
233 M F+ D* R- |按键质感好但材质遗憾；USB识别与宏依赖投诉
234 N F- R- C- D- V- |键位需大力，换货后仍有问题，高价失望
235 N F- R- D- C- |多次失灵、连点和强触发，旧V3优点不归当前款
236 N R- D- V- |两月断触，品控与价位不匹配
237 N F- R- C- D- |两个设备同样按键触发困难
238 M D* |质量尚可但售后差，耐用及维护维度混合
239 N F- R- D- |部分按键失灵，国补为交易属性
240 M A+ C+ R- F* |漂亮安静，但反应慢；黄轴手感与反馈疑虑
241 M F* C- R- |多余功能键硌手且依赖驱动，其余尚可
242 N A- V- |设计否定与同价位替代比较；包装规则投诉另记
243 N F- R- D- |个别按键无响应，正在换货
244 N F- R- D- |两次换新仍失灵
245 N F- R- D- |无响应及重复字符，属于故障负向
246 M A+ C- F- |外观好，但键位拥挤误触
247 N F- C- |侧边按键操作别扭
248 N F- C- |宏键布局不习惯且易误按
249 M R* F+ A+ C- V- D= |配置与灯效肯定但有线软件依赖、体积和价格抱怨
250 M D= |包装压痕投诉；明确产品未压坏，不推断产品故障
251 O |无产品信息，歌词与表情文本；排除研究分析
252 P D+ V+ |泛化模板重复正评，质量和价格明确正向
253 P A+ D+ F+ V= |灯效、做工和手感好；价格不高不低为中性
254 P D+ A+ R+ |泛化模板整体正评，未提供具体体验细节
255 P A+ C+ F+ |灯效、键距、绿轴反馈及舒适性正向
256 P D+ A+ R+ |与254文本重复，保留原始来源待复核
257 M F* R* A+ |灯效和静音肯定，黄轴黏滞且回弹慢；旧鼠标故障不归目标
258 P F+ C+ D+ A+ R+ |绿轴、腕托、做工、灯效和响应明确正向
259 N D- |盖子扣不紧与客服无解释，维护质量投诉
260 N F- D- C- |滚轮声音异常，舒适及做工负向
261 N D- F- V- |两个月滚轮坏，高价失望
262 N D- F- R- |不足一月按键滚轮失灵，售后投诉
263 N F- R- D- |新鼠标追踪漂移和手感差；旧设备优点不归目标
264 N R- D- V- |断线、故障无解决方案与退换障碍造成损失
265 N V- |升级价格、缺配件和品牌服务投诉，无具体响应体验
266 N F- C- R- |触感不适、误触DPI导致乱跳
267 M F+ C* R= A= D- |手感进步，取消配重、滚轮偏移，包装服务投诉
268 N F- D- |滚轮疑似损坏，属于用户推测需核实
269 N C- F- R- V- |轻量与模具不适配，配重缺失，性能及价值否定
270 N D- |脚贴划痕与疑似非新品，未独立验证真实性
271 N D- |多次换货按键划痕，做工负向
272 M D- V- F* C+ R- |价格、材质、驱动和声音负向；键位及重量正向
273 M C- F- R+ D+ |防滑手感差；无线升级和续航正向
274 M F* R* |新滚轮过于灵敏不习惯，其余一致；旧侧键坏不归新款
275 M F* C+ |反馈强烈需要适应，侧键选项与轻量肯定
276 P V+ R+ C= |无线性价比、反馈及设计任务体验肯定；非游戏用途
277 P C+ R+ |握感、轻便和灵敏肯定
278 P F+ C+ R+ |当前有线手感与延迟肯定；旧无线麻烦不归目标
279 M C+ F* V* D= |轻量脚垫好、价格勉强、线短；未来开裂是担忧
280 P F+ C+ R+ D+ V+ |与6文本重复，保留待复核
281 P F+ R+ A+ D+ |定位、响应及磨砂外壳质感获肯定
282 P A+ D= C+ R+ |与4重复；耐用仍待检验，不归正向寿命
283 P R+ C+ A+ |响应、便携、外观和舒服；undefined为采集噪声
284 M C* R+ A= D* |初始重后适应，功能正常但细节有改进空间
285 N D- |短期需技术处理但故障细节缺失，需重点复核
286 N D- F- V- |18天滚轮异响，维修限制与价格服务不匹配
287 N R- V- |Mac兼容性失败且高价比较负向，不能推广所有设备
288 N R- D- |接收器两月失灵
289 N D- F- R- |一个月滚轮问题及售后解释不足
290 N R- D- |无线间歇失灵、有线替代，无法正常使用
291 N V- |价保资格和降价投诉；欺诈属于用户指控未核验
292 N V- |短期降价引发价格价值不满
293 N F= V- |手感一般与高价、价保和服务投诉
294 N F- R- D- |按键不灵，明显使用故障
295 N D- R- V- |充电反复断连与维修时间成本
296 M R+ F* C* D* |跟手按键肯定，但汗手涂层黏手；版本为V4待核验
297 N |仅包装、送礼和疑似真假投诉，不推断产品耐久
298 M D= |包装划痕疑虑但鼠标本身没问题，产品质量中性
299 M F+ R+ V- |产品手感响应肯定，整体价保体验负向
300 P A+ V+ D+ |外观、性价比和质量明确肯定，服务另记
301 M D- |能用但分模线明显，做工负向
302 N |仅包装与真假怀疑，不归产品本体质量
303 N D- |附赠防滑贴状态与补发投诉，属于附件质量需单独复核
304 N C- F- V- |偏轻、偏滑及不适配配重，高价负向
305 P F+ C+ D+ A+ R+ V+ |手感、适配、做工和续航好，降价入手正向
306 P C+ F+ |人体工学与触感明确舒服
307 O |平台占位评价，无真实体验内容
308 P D+ |泛化质量满意，服务部分不归产品
309 P F+ C+ |与7文本重复，手掌适配需条件限定
310 P D+ |键鼠质感肯定，具体目标型号不明确
311 P |整体不错，未指向具体属性
312 P |整体升级满意，无具体属性评价
313 P |主要评价物流客服，无可核验产品属性
314 P F+ C+ D+ R+ A+ V+ |六维肯定；不用担心双击为正向预期，不当实际寿命验证
"""


def main():
    source = OUT / "review_manual_annotation_template.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    labels = {}
    for line in READINGS.strip().splitlines():
        head, note = line.split("|", 1)
        fields = head.split()
        row_id, sentiment = int(fields[0]), OVERALL[fields[1]]
        aspects = {DIMENSIONS[code[0]]: POLARITY[code[1]] for code in fields[2:]}
        if row_id in labels:
            raise ValueError(f"重复标注编号 {row_id}")
        labels[row_id] = (sentiment, aspects, note.strip())
    ids = {int(row["review_row_id"]) for row in raw}
    if ids != set(labels):
        raise ValueError(f"缺少{ids-set(labels)}；额外{set(labels)-ids}")
    output_rows = []
    for row in raw:
        sentiment, aspects, note = labels[int(row["review_row_id"])]
        output_rows.append({
            "复核状态": "待人工复核", "评论编号": int(row["review_row_id"]),
            "评论正文": row["content_clean"], "AI整体情感": sentiment,
            "AI涉及维度": "；".join(aspects),
            **{"AI"+dimension: aspects.get(dimension, "未涉及") for dimension in DIMENSIONS.values()},
            "AI判断依据及疑点": note,
            "人工复核整体情感": "", "人工复核维度及情感": "",
            "复核人": "", "复核日期": "", "人工复核备注": "",
            "入口SKU": row["platform_product_id"], "SKU归属": row["sku_scope"],
            "标签来源": "AI逐条辅助标注，非独立人工标签",
            "原评论哈希ID": row["source_review_id"],
        })
    csv_path = OUT / "评论AI辅助标注待复核.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (OUT / "review_ai_annotations.json").write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows":len(output_rows), "output":str(csv_path), "original_human_template_modified":False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
