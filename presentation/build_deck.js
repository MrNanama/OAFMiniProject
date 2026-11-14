// Builds presentation/OAF_Rebalancing_Flows.pptx from presentation/deck_data.json (all numbers come from the
// frozen backtests). Run: NODE_PATH=<dir with pptxgenjs, react, react-dom, react-icons, sharp> node build_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const SKILL = process.env.PPTX_SKILL_DIR;
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));
const D = require("./deck_data.json");
const OUT = path.join(__dirname, "OAF_Rebalancing_Flows.pptx");

const THEME = {
  name: "OAF Rebalancing Flows",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2233", lt1: "FFFFFF", dk2: "14213D", lt2: "EEF2F7",
    accent1: "1F6FB2",   // OAF blue
    accent2: "E3872D",   // amber  = stocks (SPY)
    accent3: "138A8A",   // teal   = bonds (TLT)
    accent4: "8C97A8",   // slate  = context / other strategies
    accent5: "C2413A",   // red    = risk
    accent6: "2E9A5C",   // green  = positive
    hlink: "1F6FB2", folHlink: "5B4B9E",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Rebalancing Flows - Oxford Alpha Fund";
pres.subject = "Systematic strategy pitch: month-end stock-bond rebalancing flows";
pres.author = "Oxford Alpha Fund quantitative research";
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.8, y: 1.25, w: 7.6, h: 0.45, fontSize: 14, bold: true, color: C.background2, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.8, w: 7.6, h: 1.5, fontSize: 54, bold: true, color: C.background1, margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.8, y: 3.35, w: 7.6, h: 0.9, fontSize: 24, color: C.background2, margin: 0, valign: "top" }, text: "" } },
    { placeholder: { options: { name: "meta", type: "body", x: 0.8, y: 6.2, w: 11.5, h: 0.5, fontSize: 14, color: C.accent4, margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 1.0, fontSize: 34, bold: true, color: C.text1, valign: "middle", margin: 0, align: "left" }, text: "" } },
    { text: { text: "Oxford Alpha Fund  |  Rebalancing Flows  |  November 2026", options: { x: 0.6, y: 7.0, w: 8.0, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 12.1, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent4, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 1.0, fontSize: 36, bold: true, color: C.background1, valign: "middle", margin: 0, align: "left" }, text: "" } },
  ],
});

// ---------------------------------------------------------------- helpers
async function icon(Icon, color = "#FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const num = (v, d = 2) => (v < 0 ? "\u2212" : "") + Math.abs(v).toFixed(d);
const pct = (v, d = 1) => num(v * 100, d) + "%";
const AX = {   // identical axis/label styling on every chart
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  titleFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, dataLabelFontSize: 11,
  catAxisLabelColor: "5A6478", valAxisLabelColor: "5A6478", dataLabelColor: "1B2233",
  valGridLine: { color: "E3E8EF", size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: true, catAxisLineColor: "C9D1DC", valAxisLineShow: false, showLegend: false,
  titleFontSize: 13, titleColor: "1B2233", catAxisMajorTickMark: "none", catAxisMinorTickMark: "none",
  valAxisMajorTickMark: "none",
};
function statCard(slide, x, y, w, h, value, label, color, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name + " card" });
  slide.addText(value, { x: x + 0.25, y: y + 0.25, w: w - 0.5, h: 0.85, fontSize: 40, bold: true, color, fontFace: THEME.headFontFace, margin: 0, valign: "top", isTextBox: true, objectName: name + " value" });
  slide.addText(label, { x: x + 0.25, y: y + 1.15, w: w - 0.5, h: h - 1.3, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: name + " label" });
}
function numberCircle(slide, x, y, d, n, color, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color }, objectName: name });
  slide.addText(String(n), { x, y, w: d, h: d, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name + " number" });
}
const cumDev = () => {
  const i = D.cum.dates.indexOf("2020-12");
  return { dates: D.cum.dates.slice(0, i + 1), values: D.cum.values.slice(0, i + 1) };
};
const yearLabels = (dates) => dates.map((d) => (d.endsWith("-01") && Number(d.slice(0, 4)) % 2 === 0 ? d.slice(0, 4) : ""));

async function build() {
  const IC = {
    data: await icon(fa.FaDatabase), exec: await icon(fa.FaExchangeAlt), look: await icon(fa.FaSearch),
    lock: await icon(fa.FaLock), count: await icon(fa.FaListOl), review: await icon(fa.FaUserShield),
  };
  const sd = D.stats_dev, sh = D.stats_holdout;

  // ============================================================ 1. title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("OXFORD ALPHA FUND  ·  QUANTITATIVE STRATEGY PITCH", { placeholder: "kicker" });
  s.addText("Rebalancing Flows", { placeholder: "title" });
  s.addText("Trading the month-end stock–bond squeeze", { placeholder: "subtitle" });
  s.addText("November 2026  ·  research frozen 14 Nov 2026  ·  one-shot out-of-sample test 2021–2026", { placeholder: "meta" });
  // motif: stocks (amber) and bonds (teal) trading places
  s.addShape(pres.shapes.OVAL, { x: 8.9, y: 1.5, w: 2.6, h: 2.6, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: "stocks circle" });
  s.addText([{ text: "Stocks", options: { bold: true, fontSize: 22, breakLine: true } }, { text: "SPY", options: { fontSize: 14 } }], { x: 8.9, y: 1.5, w: 2.6, h: 2.6, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "stocks label" });
  s.addShape(pres.shapes.OVAL, { x: 10.2, y: 3.4, w: 2.6, h: 2.6, fill: { color: C.accent3 }, line: { color: C.accent3 }, objectName: "bonds circle" });
  s.addText([{ text: "Bonds", options: { bold: true, fontSize: 22, breakLine: true } }, { text: "TLT", options: { fontSize: 14 } }], { x: 10.2, y: 3.4, w: 2.6, h: 2.6, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "bonds label" });
  s.addNotes("Rebalancing Flows: a systematic strategy that trades the predictable price pressure created when balanced funds and pensions rebalance between stocks and bonds at month-end. It is one of ten strategies from our multi-strategy research programme. Research was frozen on 14 Nov 2026 before a single out-of-sample test on 2021-2026 data.");

  // ============================================================ 2. executive summary
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("A small month-end edge that held up out of sample", { placeholder: "title" });
  statCard(s, 0.6, 1.6, 2.9, 2.2, sh.sharpe.toFixed(2), "Sharpe ratio out of sample (2021–26, never seen in research)", C.accent2, "holdout sharpe");
  statCard(s, 3.75, 1.6, 2.9, 2.2, sd.sharpe.toFixed(2), "Sharpe ratio in development (2004–20, net of costs)", C.accent1, "dev sharpe");
  statCard(s, 0.6, 4.1, 2.9, 2.2, pct(sh.max_dd), "Largest drawdown out of sample", C.accent3, "holdout dd");
  statCard(s, 3.75, 4.1, 2.9, 2.2, num(D.corr_holdout.SPY), "Correlation with the S&P 500 out of sample", C.text2, "spy corr");
  s.addText([
    { text: "What", options: { bold: true, color: C.accent1, breakLine: true } },
    { text: "Funds with fixed stock/bond weights must sell the month's winner near month-end. That pushes the stock–bond spread against month-to-date performance, and it rebounds once the flows are done.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 14, breakLine: true } },
    { text: "How", options: { bold: true, color: C.accent1, breakLine: true } },
    { text: "SPY vs TLT. Signal at day −4; positions on days −2/−1 (against the month) and +1/+2 (with it). Invested about four days a month.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 14, breakLine: true } },
    { text: "Why believe it", options: { bold: true, color: C.accent1, breakLine: true } },
    { text: "Best out-of-sample result of the 10 frozen strategies, and the underlying price pattern was stronger after 2020 than before.", options: {} },
  ], { x: 7.1, y: 1.6, w: 5.6, h: 4.7, fontSize: 16, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "summary text" });
  s.addNotes(`Four numbers. Out of sample, on 2021-2026 data that was locked away during research, the strategy earned a Sharpe ratio of ${sh.sharpe.toFixed(2)}, with a worst drawdown of ${pct(sh.max_dd)} and essentially zero correlation with the S&P 500. In development it earned ${sd.sharpe.toFixed(2)} after costs. The idea is a flow effect: rebalancers must trade against the month's winner, which creates temporary price pressure we can provide liquidity to.`);

  // ============================================================ 3. where it came from
  pres.addSection({ title: "The idea" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "The idea" });
  s.addText("Best out-of-sample result in our programme", { placeholder: "title" });
  const NAMES = { s01_gp_trend: "Gaussian-process trend", s02_factor_momentum: "Factor momentum", s05_principal_portfolios: "Principal portfolios",
    s08_ml_stock_ranker: "ML stock ranker", s09_vrp_timing: "Volatility risk premium", s10_turn_of_month: "Month-turn flows",
    s11_cross_asset_carry: "Commodity carry", s12_crypto_trend: "Crypto trend", s15_rebalancing_flows: "Rebalancing Flows",
    s16_treasury_auctions: "Treasury auction cycle" };
  const book = [...D.book].sort((a, b) => a.sr_holdout - b.sr_holdout);
  const avgDev = D.book.reduce((t, b) => t + b.sr_dev, 0) / D.book.length;
  const avgHo = D.book.reduce((t, b) => t + b.sr_holdout, 0) / D.book.length;
  s.addChart(pres.charts.BAR, [{ name: "Holdout Sharpe", labels: book.map((b) => NAMES[b.strategy]), values: book.map((b) => b.sr_holdout) }], {
    x: 0.6, y: 1.5, w: 7.4, h: 5.2, barDir: "bar", ...AX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.00;"\u2212"0.00',
    chartColors: book.map((b) => (b.strategy === "s15_rebalancing_flows" ? HEX.accent2 : HEX.accent4)),
    valAxisMinVal: -0.4, valAxisMaxVal: 1.2, valAxisMajorUnit: 0.4, valAxisLabelFormatCode: '0.0;"\u2212"0.0',
    catAxisLabelPos: "low", showTitle: true, title: "Sharpe ratio, out of sample 2021–2026 (net of costs)", objectName: "holdout sharpe chart",
  });
  s.addText([
    { text: "16", options: { fontSize: 40, bold: true, color: C.accent1, fontFace: THEME.headFontFace, breakLine: true } },
    { text: "strategy ideas tested, inspired by earlier OAF decks (Bayesian quadrature, macro momentum, RL pairs, volume filters)", options: { fontSize: 14, breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: "10", options: { fontSize: 40, bold: true, color: C.accent1, fontFace: THEME.headFontFace, breakLine: true } },
    { text: "kept after six rounds of improvement on 2004–2020 data, then frozen and tested once on 2021–2026", options: { fontSize: 14, breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: `${num(avgDev)} → ${num(avgHo)}`, options: { fontSize: 32, bold: true, color: C.accent4, fontFace: THEME.headFontFace, breakLine: true } },
    { text: `average Sharpe of the ten, development (2004–20) → holdout. Rebalancing Flows went the other way: ${num(sd.sharpe)} → ${num(sh.sharpe)}`, options: { fontSize: 14 } },
  ], { x: 8.5, y: 1.5, w: 4.2, h: 5.2, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "programme facts" });
  s.addNotes("This strategy came out of a research programme in which we built a realistic backtester and tested sixteen new ideas inspired by earlier OAF decks. Six were retired on development data, ten were frozen and tested once on 2021-2026. Most strategies decayed out of sample, which is normal (average Sharpe roughly halved). Rebalancing Flows was the exception: its Sharpe ratio doubled.");

  // ============================================================ 4. mechanism
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "The idea" });
  s.addText("Why it works: rebalancers trade against winners", { placeholder: "title" });
  const steps = [
    ["Fixed weights", "Pensions, target-date and balanced funds hold fixed stock/bond mixes (e.g. 60/40) and rebalance on a calendar."],
    ["Month-to-date drift", "When stocks beat bonds during the month, these portfolios end up overweight equities."],
    ["Month-end selling", "To get back to target they sell stocks and buy bonds in the last days of the month, pushing the spread down."],
    ["Rebound", "Once the flow is done, the pressure fades and the stock–bond spread recovers in the first days of the next month."],
  ];
  const cw = 2.75, cg = 0.37;
  steps.forEach(([h, t], i) => {
    const x = 0.6 + i * (cw + cg);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: cw, h: 3.6, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `step ${i + 1} card` });
    numberCircle(s, x + 0.25, 1.95, 0.6, i + 1, i === 2 ? C.accent3 : i === 3 ? C.accent2 : C.accent1, `step ${i + 1} badge`);
    s.addText(h, { x: x + 0.25, y: 2.7, w: cw - 0.5, h: 0.5, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: `step ${i + 1} heading` });
    s.addText(t, { x: x + 0.25, y: 3.2, w: cw - 0.5, h: 2.0, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `step ${i + 1} text` });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + cw + 0.04, y: 3.35, w: 0.29, h: 0.4, fill: { color: C.accent4 }, line: { color: C.accent4 }, objectName: `arrow ${i + 1}` });
  });
  s.addText("Mechanism documented by Harvey, Mazzoleni & Melone (2025), “The Unintended Consequences of Rebalancing”. We are paid for absorbing a predictable, non-informational flow.", { x: 0.6, y: 5.75, w: 12.1, h: 0.8, fontSize: 14, italic: true, color: C.accent4, margin: 0, isTextBox: true, objectName: "source note" });
  s.addNotes("The economics. Trillions sit in funds with fixed stock/bond targets. If equities outperform during the month, these funds become overweight equities and must sell stocks and buy bonds near month-end, regardless of price. That temporary, non-informational pressure moves the stock-bond spread against the month's winner, and it reverses once the flow is absorbed. We are effectively being paid to provide liquidity to rebalancers.");

  // ============================================================ 5. dev evidence
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "The idea" });
  s.addText("Evidence: pressure into month-end, then rebound", { placeholder: "title" });
  const qLab = ["Q1 stocks lagged", "Q2", "Q3", "Q4", "Q5 stocks led"];
  const signCol = (vals) => vals.map((v) => (v < 0 ? HEX.accent3 : HEX.accent2));
  const ed = D.event_dev;
  s.addChart(pres.charts.BAR, [{ name: "Days -2/-1", labels: qLab, values: ed.late_bp }], {
    x: 0.6, y: 1.5, w: 6.0, h: 4.4, barDir: "col", ...AX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0;"\u2212"0',
    chartColors: signCol(ed.late_bp), valAxisLabelFormatCode: '0;"\u2212"0', showTitle: true,
    title: "Stocks minus bonds on the last two days of the month (bp)", objectName: "dev late chart",
  });
  s.addChart(pres.charts.BAR, [{ name: "Days +1/+2", labels: qLab, values: ed.early_bp }], {
    x: 6.75, y: 1.5, w: 6.0, h: 4.4, barDir: "col", ...AX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0;"\u2212"0',
    chartColors: signCol(ed.early_bp), valAxisLabelFormatCode: '0;"\u2212"0', showTitle: true,
    title: "Stocks minus bonds on the first two days of the next month (bp)", objectName: "dev early chart",
  });
  s.addText(`2004–2020, ${ed.n} months, grouped by month-to-date stocks-minus-bonds return at day −4 (Q1 average ${num(ed.mtd_bp[0] / 100, 1)}%, Q5 +${num(ed.mtd_bp[4] / 100, 1)}%). Correlation ${num(ed.corr_late)} into month-end, +${num(ed.corr_early)} after it. Amber: stocks beat bonds; teal: bonds beat stocks. Bonds tend to win the last days of every month (index extension); the trade is the difference across quintiles.`, { x: 0.6, y: 6.0, w: 12.1, h: 0.85, fontSize: 12, color: C.accent4, margin: 0, valign: "top", isTextBox: true, objectName: "dev evidence caption" });
  s.addNotes(`Development data, ${ed.n} months. We sort months by how far stocks were ahead of bonds four days before month-end. Left: when stocks had led the most (Q5), stocks lagged bonds by about ${Math.abs(ed.late_bp[4]).toFixed(0)} basis points over the last two days; when they had lagged (Q1) the gap is only ${Math.abs(ed.late_bp[0]).toFixed(0)}. Right: the pressure then reverses - Q5 months see stocks rebound ${ed.early_bp[4].toFixed(0)} bp over the first two days of the next month, Q1 months ${ed.early_bp[0].toFixed(0)}. The rebound rises steadily from Q1 to Q5, and the month-end pressure is concentrated in the months where stocks led most (Q4-Q5) - the pattern a flow story predicts.`);

  // ============================================================ 6. rule
  pres.addSection({ title: "Strategy" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Strategy" });
  s.addText("The rule: four days a month, two ETFs", { placeholder: "title" });
  const days = ["−4", "−3", "−2", "−1", "+1", "+2", "+3"];
  const tx0 = 0.9, tstep = 1.55, ty = 2.55;
  s.addShape(pres.shapes.LINE, { x: tx0, y: ty, w: tstep * 6, h: 0, line: { color: C.accent4, width: 2 }, objectName: "timeline" });
  s.addShape(pres.shapes.LINE, { x: tx0 + 3.5 * tstep, y: 1.9, w: 0, h: 1.55, line: { color: C.text1, width: 1.5, dashType: "dash" }, objectName: "month end marker" });
  s.addText("month end", { x: tx0 + 3.5 * tstep - 0.8, y: 1.5, w: 1.6, h: 0.3, fontSize: 12, italic: true, color: C.text1, align: "center", margin: 0, isTextBox: true, objectName: "month end label" });
  days.forEach((d, i) => {
    const cx = tx0 + i * tstep;
    const col = i === 0 ? C.text2 : (i === 2 || i === 3) ? C.accent3 : (i === 4 || i === 5) ? C.accent2 : C.accent4;
    s.addShape(pres.shapes.OVAL, { x: cx - 0.28, y: ty - 0.28, w: 0.56, h: 0.56, fill: { color: col }, line: { color: col }, objectName: `day ${d} dot` });
    s.addText(d, { x: cx - 0.28, y: ty - 0.28, w: 0.56, h: 0.56, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `day ${d} label` });
  });
  s.addText("Signal", { x: tx0 - 0.75, y: 2.95, w: 1.5, h: 0.35, fontSize: 14, bold: true, color: C.text2, align: "center", margin: 0, isTextBox: true, objectName: "signal label" });
  s.addText("Against the month", { x: tx0 + 2 * tstep - 0.4, y: 2.95, w: tstep + 0.8, h: 0.35, fontSize: 14, bold: true, color: C.accent3, align: "center", margin: 0, isTextBox: true, objectName: "contrarian label" });
  s.addText("With the rebound", { x: tx0 + 4 * tstep - 0.4, y: 2.95, w: tstep + 0.8, h: 0.35, fontSize: 14, bold: true, color: C.accent2, align: "center", margin: 0, isTextBox: true, objectName: "rebound label" });
  const rules = [
    ["1  Measure", "At the close of day −4: month-to-date log return of SPY minus TLT, divided by its 36-month standard deviation → z (capped at ±2)."],
    ["2  Filter", "Skip the month if |z| < 0.5: a weak imbalance means little rebalancing pressure but full trading costs."],
    ["3  Trade", "Days −2/−1: hold −z × (long SPY, short TLT). Days +1/+2: hold +z. Flat otherwise. Legs inverse-volatility weighted."],
    ["4  Execute", "Market-on-close orders on the exchange's published calendar; book scaled to 10% annualised volatility while invested."],
  ];
  rules.forEach(([h, t], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 3.7 + Math.floor(i / 2) * 1.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.95, h: 1.3, rectRadius: 0.06, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `rule ${i + 1} card` });
    s.addText([{ text: h, options: { bold: true, fontSize: 15, color: C.accent1, breakLine: true } }, { text: t, options: { fontSize: 13, color: C.text1 } }],
      { x: x + 0.2, y: y + 0.1, w: 5.55, h: 1.1, valign: "top", margin: 0, isTextBox: true, objectName: `rule ${i + 1} text` });
  });
  s.addNotes("The rule is deliberately simple. Four days before month-end we measure how far stocks are ahead of bonds this month, scaled by how big that gap usually is. If the imbalance is meaningful, we bet against it over the last two days of the month and with the rebound over the first two days of the next. Everything trades at the close on days we know in advance, using two of the most liquid ETFs in the world.");

  // ============================================================ 7. testing discipline
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Strategy" });
  s.addText("Built to avoid backtest traps", { placeholder: "title" });
  const checks = [
    [IC.data, "Real, audited data", "Yahoo total-return prices for SPY and TLT, a data audit, and conservative cleaning of bad prints."],
    [IC.exec, "Realistic execution", "Decide at one close, trade at the next. 2 bp per side, short-borrow and leverage financing charged."],
    [IC.look, "No look-ahead", "Truncation test at random dates and at the strategy's own decision dates; scheduled exchange calendar."],
    [IC.lock, "Locked holdout", "2021–26 data were unreadable in code until the strategy was frozen (git-recorded), then tested once."],
    [IC.count, "Counted trials", `Only ${D.robust.trials} variants of this strategy were ever run, all logged. Deflated Sharpe ${D.robust.dsr.toFixed(2)}.`],
    [IC.review, "Independent reviews", "Three adversarial code reviews; every confirmed issue was fixed before the freeze."],
  ];
  checks.forEach(([img, h, t], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.6 + Math.floor(i / 2) * 1.6;
    s.addShape(pres.shapes.OVAL, { x, y, w: 0.85, h: 0.85, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `check ${i + 1} circle` });
    s.addImage({ data: img, x: x + 0.22, y: y + 0.22, w: 0.41, h: 0.41, altText: h, objectName: `check ${i + 1} icon` });
    s.addText([{ text: h, options: { bold: true, fontSize: 16, color: C.text1, breakLine: true } }, { text: t, options: { fontSize: 13, color: C.text1 } }],
      { x: x + 1.05, y: y - 0.05, w: 4.85, h: 1.35, valign: "top", margin: 0, isTextBox: true, objectName: `check ${i + 1} text` });
  });
  s.addText("Lessons built in from earlier OAF projects: include trading costs (Volume Filters), use point-in-time data rather than revised series (Macro Momentum), and test on long, broad samples (RL Pairs).", { x: 0.6, y: 6.35, w: 12.1, h: 0.55, fontSize: 12, italic: true, color: C.accent4, margin: 0, isTextBox: true, objectName: "lessons note" });
  s.addNotes("Why trust these numbers. Real data, realistic execution and costs, an automated look-ahead test, a holdout period that the code physically could not read until we froze the strategy, a full log of every variant we tried, and three independent adversarial reviews. These address the issues we saw in earlier projects: missing costs, revised macro data, and very short test windows.");

  // ============================================================ 8. dev performance
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Development: Sharpe 0.52 after costs", { placeholder: "title" });
  const cd = cumDev();
  s.addChart(pres.charts.LINE, [{ name: "Cumulative return", labels: yearLabels(cd.dates), values: cd.values }], {
    x: 0.6, y: 1.5, w: 7.9, h: 5.2, ...AX, chartColors: [HEX.accent1], lineSize: 2.5, lineDataSymbol: "none",
    valAxisLabelFormatCode: '0"%";"\u2212"0"%"', catAxisLabelFrequency: 1, catAxisOrientation: "minMax",
    showTitle: true, title: "Cumulative net return, Nov 2004 – Dec 2020 (%)", objectName: "dev cumulative chart",
  });
  const devStats = [
    [sd.sharpe.toFixed(2), "Sharpe ratio (t-stat " + sd.t_hac.toFixed(1) + ")"],
    [pct(sd.ann_ret), "Annual return at " + pct(sd.ann_vol) + " volatility"],
    [pct(sd.max_dd), "Maximum drawdown"],
    [sd.pos_years, "Positive calendar years"],
  ];
  devStats.forEach(([v, l], i) => {
    s.addText([{ text: v, options: { fontSize: 30, bold: true, color: C.accent1, fontFace: THEME.headFontFace, breakLine: true } }, { text: l, options: { fontSize: 13, color: C.text1 } }],
      { x: 8.9, y: 1.55 + i * 1.18, w: 3.8, h: 1.05, valign: "top", margin: 0, isTextBox: true, objectName: `dev stat ${i + 1}` });
  });
  s.addText(`Invested about four days a month, so volatility is low; costs ${sd.cost_yr.toFixed(1)}% a year.`, { x: 8.9, y: 6.25, w: 3.8, h: 0.6, fontSize: 12, italic: true, color: C.accent4, margin: 0, isTextBox: true, objectName: "dev note" });
  s.addNotes(`In development the strategy earned ${pct(sd.ann_ret)} a year at only ${pct(sd.ann_vol)} volatility: a Sharpe ratio of ${sd.sharpe.toFixed(2)} after costs, with a t-statistic of ${sd.t_hac.toFixed(1)}. Volatility is low because the book is only invested around four days a month; the Sharpe ratio is what matters, and the position can be scaled. Note the weaker stretch in 2019-2020.`);

  // ============================================================ 9. holdout performance
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Out of sample: Sharpe 1.03, frozen code", { placeholder: "title" });
  const iDev = D.cum.dates.indexOf("2020-12");
  const devSer = D.cum.values.map((v, i) => (i <= iDev ? v : null));
  const hoSer = D.cum.values.map((v, i) => (i >= iDev ? v : null));
  s.addChart(pres.charts.LINE, [
    { name: "Development 2004–2020", labels: yearLabels(D.cum.dates), values: devSer },
    { name: "Holdout 2021–2026", labels: yearLabels(D.cum.dates), values: hoSer },
  ], {
    x: 0.6, y: 1.5, w: 7.9, h: 5.2, ...AX, chartColors: [HEX.accent4, HEX.accent2], lineSize: 2.5, lineDataSymbol: "none",
    valAxisLabelFormatCode: '0"%";"\u2212"0"%"', showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: "5A6478",
    showTitle: true, title: "Cumulative net return, Nov 2004 – Nov 2026 (%)", objectName: "holdout cumulative chart",
  });
  const hoStats = [
    [sh.sharpe.toFixed(2), "Sharpe ratio, 2021–2026 (t-stat " + sh.t_hac.toFixed(1) + ")"],
    [sh.pos_years, "Positive years (2026 to 13 Nov)"],
    [pct(sh.max_dd), "Maximum drawdown"],
    [pct(sh.ann_ret), "Annual return at " + pct(sh.ann_vol) + " volatility"],
  ];
  hoStats.forEach(([v, l], i) => {
    s.addText([{ text: v, options: { fontSize: 30, bold: true, color: C.accent2, fontFace: THEME.headFontFace, breakLine: true } }, { text: l, options: { fontSize: 13, color: C.text1 } }],
      { x: 8.9, y: 1.55 + i * 1.18, w: 3.8, h: 1.05, valign: "top", margin: 0, isTextBox: true, objectName: `holdout stat ${i + 1}` });
  });
  s.addText("Same code, same parameters: nothing was changed after seeing these years.", { x: 8.9, y: 6.25, w: 3.8, h: 0.6, fontSize: 12, italic: true, color: C.accent4, margin: 0, isTextBox: true, objectName: "holdout note" });
  s.addNotes(`The real test. The code was frozen and only then run once on 2021 to November 2026. The Sharpe ratio was ${sh.sharpe.toFixed(2)}, double the development figure, with ${sh.pos_years} positive years and a maximum drawdown of ${pct(sh.max_dd)}. Nothing was tuned after seeing these numbers.`);

  // ============================================================ 10. holdout mechanism
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("The mechanism got stronger after 2020", { placeholder: "title" });
  const eh = D.event_holdout;
  s.addChart(pres.charts.BAR, [{ name: "Days -2/-1", labels: qLab, values: eh.late_bp }], {
    x: 0.6, y: 1.5, w: 6.0, h: 4.4, barDir: "col", ...AX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0;"\u2212"0',
    chartColors: signCol(eh.late_bp), valAxisLabelFormatCode: '0;"\u2212"0', showTitle: true,
    title: "Stocks minus bonds on the last two days of the month (bp)", objectName: "holdout late chart",
  });
  s.addChart(pres.charts.BAR, [{ name: "Days +1/+2", labels: qLab, values: eh.early_bp }], {
    x: 6.75, y: 1.5, w: 6.0, h: 4.4, barDir: "col", ...AX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0;"\u2212"0',
    chartColors: signCol(eh.early_bp), valAxisLabelFormatCode: '0;"\u2212"0', showTitle: true,
    title: "Stocks minus bonds on the first two days of the next month (bp)", objectName: "holdout early chart",
  });
  s.addText(`2021–2026 holdout, ${eh.n} months, same grouping as the development evidence. Q5 minus Q1 into month-end: ${num(eh.late_bp[4] - eh.late_bp[0], 0)} bp (development ${num(ed.late_bp[4] - ed.late_bp[0], 0)} bp). Correlations ${num(eh.corr_late)} and +${num(eh.corr_early)} (development ${num(ed.corr_late)}, +${num(ed.corr_early)}). Consistent with more assets in target-date and balanced funds.`, { x: 0.6, y: 6.0, w: 12.1, h: 0.85, fontSize: 12, color: C.accent4, margin: 0, valign: "top", isTextBox: true, objectName: "holdout evidence caption" });
  s.addNotes("This is the most convincing slide. It is not just the P&L that held up; the price pattern itself repeated in data we never looked at, and more strongly. Months where stocks had led saw stocks lag sharply into month-end and rebound afterwards. A plausible reason is the continued growth of target-date and balanced funds.");

  // ============================================================ 11. year by year
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  const nPos = D.annual.ret.filter((v) => v > 0).length;
  s.addText(`Positive in ${nPos} of ${D.annual.ret.length} calendar years`, { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "Annual return", labels: D.annual.years.map(String), values: D.annual.ret }], {
    x: 0.6, y: 1.5, w: 12.1, h: 4.6, barDir: "col", ...AX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0;"\u2212"0.0',
    chartColors: D.annual.years.map((y) => (y >= 2021 ? HEX.accent2 : HEX.accent1)), valAxisLabelFormatCode: '0"%";"\u2212"0"%"',
    showTitle: true, title: "Net return by calendar year (%); blue = development, amber = holdout", objectName: "annual returns chart",
  });
  s.addText("2004 starts in November and 2026 ends on 13 November. The weak years 2019–2020 fall in development, and the strategy was kept anyway. 2026 to date: −4.0%.", { x: 0.6, y: 6.25, w: 12.1, h: 0.6, fontSize: 12, color: C.accent4, margin: 0, isTextBox: true, objectName: "annual note" });
  s.addNotes("Year by year the strategy is positive most of the time. The losing years cluster in 2005-06 and 2019-20; we saw 2019-20 during development and did not tune them away. 2026 is negative so far, a reminder that this is a modest edge, not a free lunch.");

  // ============================================================ 12. robustness
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Robust to parameters, not to late trades", { placeholder: "title" });
  const sensLab = { "(chosen)": "Chosen: skip |z| < 0.5", '{"z_min": 0.0}': "No filter", '{"z_min": 1.0}': "Stricter: skip |z| < 1",
    '{"late": [-3, -2, -1]}': "Month-end window −3…−1", '{"early": [1, 2, 3]}': "Rebound window +1…+3" };
  const order = ['{"early": [1, 2, 3]}', '{"late": [-3, -2, -1]}', '{"z_min": 1.0}', "(chosen)", '{"z_min": 0.0}'];   // drawn bottom-to-top
  const sens = order.map((o) => D.sens.find((r) => r.override === o));
  s.addChart(pres.charts.BAR, [{ name: "Dev Sharpe", labels: sens.map((r) => sensLab[r.override]), values: sens.map((r) => r.SR_04_20) }], {
    x: 0.6, y: 1.5, w: 7.2, h: 5.0, barDir: "bar", ...AX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.00;"\u2212"0.00',
    chartColors: sens.map((r) => (r.override === "(chosen)" ? HEX.accent2 : HEX.accent4)), valAxisMinVal: 0, valAxisMaxVal: 0.8,
    valAxisMajorUnit: 0.2, valAxisLabelFormatCode: '0.0;"\u2212"0.0', showTitle: true, title: "Development Sharpe ratio under alternative settings", objectName: "sensitivity chart",
  });
  const rb = D.robust;
  const robustCards = [
    [rb.sr_cost2x.toFixed(2), "Sharpe with all trading costs doubled", C.accent1],
    [`${rb.ci[0].toFixed(2)}–${rb.ci[1].toFixed(2)}`, "Bootstrap 95% interval for the development Sharpe", C.accent1],
    [num(rb.sr_lag1), "Sharpe if every trade is one day late: the edge lives on specific days, so orders must be pre-scheduled", C.accent5],
  ];
  robustCards.forEach(([v, l, col], i) => {
    s.addText([{ text: v, options: { fontSize: 30, bold: true, color: col, fontFace: THEME.headFontFace, breakLine: true } }, { text: l, options: { fontSize: 13, color: C.text1 } }],
      { x: 8.2, y: 1.55 + i * 1.6, w: 4.5, h: 1.45, valign: "top", margin: 0, isTextBox: true, objectName: `robust stat ${i + 1}` });
  });
  s.addNotes("Robustness. Nearby settings give similar results, and the setting we chose is not the best one on the grid, so we did not cherry-pick. Doubling costs still leaves a positive Sharpe. The one thing that kills it is trading a day late, because the edge is tied to specific days. In practice that means market-on-close orders on a calendar computed in advance.");

  // ============================================================ 13. portfolio fit
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("A true diversifier inside the book", { placeholder: "title" });
  const corrKeys = Object.keys(D.corr_holdout);
  const corrLab = corrKeys.map((k) => (k === "SPY" ? "S&P 500 (SPY)" : NAMES[k]));
  s.addChart(pres.charts.BAR, [{ name: "Correlation", labels: corrLab, values: corrKeys.map((k) => D.corr_holdout[k]) }], {
    x: 0.6, y: 1.5, w: 7.4, h: 5.2, barDir: "bar", ...AX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.00;"\u2212"0.00',
    chartColors: corrKeys.map((k) => (k === "SPY" ? HEX.dk2 : HEX.accent4)), valAxisMinVal: -0.5, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.25,
    valAxisLabelFormatCode: '0.00;"\u2212"0.00', catAxisLabelPos: "low", showTitle: true, title: "Weekly correlation of Rebalancing Flows, holdout 2021–2026", objectName: "correlation chart",
  });
  s.addText([
    { text: "0.00", options: { fontSize: 30, bold: true, color: C.accent1, fontFace: THEME.headFontFace, breakLine: true } },
    { text: "beta to the S&P 500 in development", options: { fontSize: 13, breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: `${D.combo_dev.sharpe.toFixed(2)} → ${D.combo_holdout.sharpe.toFixed(2)}`, options: { fontSize: 30, bold: true, color: C.accent1, fontFace: THEME.headFontFace, breakLine: true } },
    { text: `Sharpe of the equal-risk 10-strategy book, development → holdout (holdout max drawdown ${pct(D.combo_holdout.max_dd)}, positive every year)`, options: { fontSize: 13, breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: "\u00b10.10", options: { fontSize: 30, bold: true, color: C.accent2, fontFace: THEME.headFontFace, breakLine: true } },
    { text: "all holdout correlations with the S&P 500 and the other nine strategies lie within this band", options: { fontSize: 13 } },
  ], { x: 8.4, y: 1.5, w: 4.3, h: 5.2, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "portfolio facts" });
  s.addNotes("Portfolio fit. The strategy has no market beta and its correlations with the S&P 500 and with our other nine strategies are all within plus or minus 0.1. That is why it adds so much to a diversified book. The equal-risk combination of all ten strategies earned a Sharpe of 0.89 out of sample with a drawdown of about 8 percent.");

  // ============================================================ 14. risks
  pres.addSection({ title: "Risks and next steps" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Risks and next steps" });
  s.addText("Risks and how we would manage them", { placeholder: "title" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 14, valign: "middle" } });
  const cell = (t, b = false) => ({ text: t, options: { bold: b, color: C.text1, fontSize: 13, valign: "middle" } });
  const rows = [
    [hdr("Risk"), hdr("Why it matters"), hdr("How we would manage it")],
    [cell("Crowding", true), cell("Published in 2025; more capital may compete the edge away."), cell("Track a rolling 2-year Sharpe; cut size if it falls below zero.")],
    [cell("Execution timing", true), cell("One day late turns the Sharpe negative (−0.54)."), cell("Market-on-close orders from a calendar computed weeks ahead.")],
    [cell("Low volatility", true), cell("About 3–4% a year as backtested; returns are small unless scaled."), cell("Run inside the multi-strategy book; equity and bond futures for cheap leverage.")],
    [cell("Losing stretches", true), cell("2019–20 lost about 10%; 2026 is −4% to date."), cell("Small risk budget; diversification across the 10 strategies.")],
    [cell("Model risk", true), cell("Backtest assumes ETF costs and borrow at stated rates."), cell("Paper-trade first and compare fills with backtested prices.")],
  ];
  s.addTable(rows, { x: 0.6, y: 1.55, w: 12.1, colW: [2.2, 4.9, 5.0], rowH: [0.5, 0.8, 0.8, 0.8, 0.8, 0.8], border: { type: "solid", pt: 0.75, color: "D5DCE6" }, fill: { color: C.background1 }, margin: 0.08, objectName: "risk table" });
  s.addNotes("The main risks: crowding, execution timing, the small size of the returns as backtested, losing stretches, and model risk. Each has a concrete mitigation, most importantly scheduled market-on-close execution and running the strategy inside the diversified book.");

  // ============================================================ 15. next steps (dark)
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Risks and next steps" });
  s.addText("Next steps", { placeholder: "title" });
  const nxt = [
    ["Paper-trade", "Three months of market-on-close orders on the published calendar; compare fills with the backtest."],
    ["Widen the universe", "Same rule on other balanced pairs: international equities vs bonds, futures in place of ETFs."],
    ["Quarter-ends", "Rebalancing is largest at quarter-ends; test a heavier weight there (pre-registered, then out of sample)."],
    ["Combine", "Run alongside the other nine strategies at equal risk, with a drawdown-based size limit."],
  ];
  nxt.forEach(([h, t], i) => {
    const y = 1.75 + i * 1.15;
    numberCircle(s, 0.6, y, 0.65, i + 1, i % 2 === 0 ? C.accent2 : C.accent3, `next ${i + 1} badge`);
    s.addText([{ text: h, options: { bold: true, fontSize: 18, color: C.background1, breakLine: true } }, { text: t, options: { fontSize: 15, color: C.background2 } }],
      { x: 1.5, y: y - 0.05, w: 7.9, h: 1.05, valign: "top", margin: 0, isTextBox: true, objectName: `next ${i + 1} text` });
  });
  s.addText("Questions?", { x: 9.6, y: 3.0, w: 3.1, h: 1.0, fontSize: 36, bold: true, color: C.background1, fontFace: THEME.headFontFace, align: "right", margin: 0, isTextBox: true, objectName: "questions" });
  s.addNotes("Proposed next steps: paper-trade for three months, extend the rule to other stock/bond pairs and to futures, test a pre-registered quarter-end weighting, and run it inside the diversified book. Happy to take questions.");

  // ============================================================ 16. disclaimer
  pres.addSection({ title: "Disclaimer" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Disclaimer" });
  s.addText("Disclaimer", { placeholder: "title" });
  s.addText([
    { text: "Analyst disclaimer", options: { bold: true, fontSize: 16, breakLine: true } },
    { text: "This material was prepared by students for the educational purposes of the Oxford Alpha Fund. The Oxford Alpha Fund is not authorised or regulated by any financial authority to provide financial services or advice, and does not hold positions based on this research.", options: { fontSize: 14, breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "General disclaimer", options: { bold: true, fontSize: 16, breakLine: true } },
    { text: "Nothing here is an investment recommendation or an offer to buy or sell any security. All results are hypothetical backtests on historical data, after estimated trading costs; they do not reflect actual trading, and past performance is not a guide to future performance. Data come from public sources (Yahoo Finance, Kenneth French data library, federalreserve.gov, TreasuryDirect) and may contain errors. Opinions are those of the authors at the date of preparation and may change without notice.", options: { fontSize: 14 } },
  ], { x: 0.6, y: 1.6, w: 12.1, h: 4.6, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "disclaimer text" });
  s.addNotes("Standard disclaimer: educational material, hypothetical backtests, not investment advice.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
