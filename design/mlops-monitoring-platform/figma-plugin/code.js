const C = {
  bg: "#EEF3F9",
  surface: "#FFFFFF",
  soft: "#F7F9FC",
  line: "#E1E8F2",
  text: "#102033",
  muted: "#66758A",
  blue: "#2563EB",
  blueSoft: "#E8F0FF",
  green: "#16A34A",
  greenSoft: "#E7F7ED",
  yellow: "#D97706",
  yellowSoft: "#FFF4DF",
  red: "#DC2626",
  redSoft: "#FDE8E8",
  dark: "#0F1D2E"
};

const screens = [
  "Dashboard",
  "Model Detail",
  "Data Drift Monitoring",
  "Model Performance",
  "Alerts and Incidents",
  "Model Registry"
];

function rgb(hex) {
  const value = hex.replace("#", "");
  return {
    r: parseInt(value.slice(0, 2), 16) / 255,
    g: parseInt(value.slice(2, 4), 16) / 255,
    b: parseInt(value.slice(4, 6), 16) / 255
  };
}

function paint(hex) {
  return { type: "SOLID", color: rgb(hex) };
}

function rect(parent, x, y, w, h, fill, radius = 12, stroke = null) {
  const node = figma.createRectangle();
  parent.appendChild(node);
  node.x = x;
  node.y = y;
  node.resize(w, h);
  node.cornerRadius = radius;
  node.fills = [paint(fill)];
  if (stroke) {
    node.strokes = [paint(stroke)];
    node.strokeWeight = 1;
  }
  return node;
}

function text(parent, value, x, y, w, options = {}) {
  const node = figma.createText();
  parent.appendChild(node);
  node.x = x;
  node.y = y;
  node.resize(w, options.h || 24);
  node.fontName = { family: "Inter", style: options.style || "Regular" };
  node.characters = value;
  node.fontSize = options.size || 14;
  node.fills = [paint(options.color || C.text)];
  if (options.align) node.textAlignHorizontal = options.align;
  if (options.lineHeight) node.lineHeight = { unit: "PIXELS", value: options.lineHeight };
  return node;
}

function makeFrame(name, x) {
  const frame = figma.createFrame();
  frame.name = name;
  frame.x = x;
  frame.y = 0;
  frame.resize(1440, 1024);
  frame.fills = [paint(C.bg)];
  frame.clipsContent = true;
  figma.currentPage.appendChild(frame);
  return frame;
}

function badge(parent, value, x, y, kind = "info", w = 92) {
  const map = {
    healthy: [C.greenSoft, C.green],
    warning: [C.yellowSoft, C.yellow],
    critical: [C.redSoft, C.red],
    info: [C.blueSoft, C.blue],
    neutral: ["#EEF2F7", "#53657C"]
  };
  const colors = map[kind] || map.info;
  rect(parent, x, y, w, 28, colors[0], 14);
  text(parent, value, x, y + 6, w, { size: 11, style: "Bold", color: colors[1], align: "CENTER" });
}

function button(parent, value, x, y, w, primary = false) {
  rect(parent, x, y, w, 34, primary ? C.blue : C.surface, 10, primary ? C.blue : C.line);
  text(parent, value, x, y + 8, w, { size: 11, style: "Bold", color: primary ? "#FFFFFF" : C.text, align: "CENTER" });
}

function card(parent, x, y, w, h, title = null) {
  rect(parent, x, y, w, h, C.surface, 20, C.line);
  if (title) text(parent, title, x + 20, y + 18, w - 40, { size: 17, style: "Bold" });
}

function metric(parent, x, y, w, label, value, note, kind = "info") {
  card(parent, x, y, w, 132);
  text(parent, label, x + 18, y + 18, w - 120, { size: 12, style: "Bold", color: C.muted });
  badge(parent, kind === "info" ? "Live" : kind[0].toUpperCase() + kind.slice(1), x + w - 104, y + 14, kind, 84);
  text(parent, value, x + 18, y + 52, w - 36, { size: 30, style: "Bold" });
  text(parent, note, x + 18, y + 94, w - 36, { size: 12, color: C.muted });
}

function sidebar(parent, active) {
  rect(parent, 0, 0, 264, 1024, C.dark, 0);
  rect(parent, 28, 28, 40, 40, C.blue, 12);
  text(parent, "ML", 28, 38, 40, { size: 13, style: "Bold", color: "#FFFFFF", align: "CENTER" });
  text(parent, "ModelPulse", 80, 29, 150, { size: 18, style: "Bold", color: "#FFFFFF" });
  text(parent, "Production ML Control", 80, 54, 160, { size: 11, color: "#8EA2BB" });
  text(parent, "MONITOR", 32, 112, 160, { size: 10, style: "Bold", color: "#7890AD" });
  const items = ["Dashboard", "Models", "Data Drift", "Performance", "Alerts", "Registry", "Settings"];
  items.forEach((item, index) => {
    const y = 138 + index * 48 + (index === 6 ? 28 : 0);
    const isActive = item === active;
    if (isActive) rect(parent, 20, y, 224, 42, C.blue, 12);
    rect(parent, 34, y + 9, 24, 24, isActive ? "#3B82F6" : "#22334A", 8);
    text(parent, item.slice(0, 2).toUpperCase(), 34, y + 15, 24, { size: 8, style: "Bold", color: "#FFFFFF", align: "CENTER" });
    text(parent, item, 70, y + 12, 150, { size: 13, style: "Bold", color: isActive ? "#FFFFFF" : "#C4D2E5" });
  });
  rect(parent, 20, 850, 224, 118, "#18293F", 18, "#294059");
  text(parent, "Production health", 38, 870, 180, { size: 13, style: "Bold", color: "#FFFFFF" });
  text(parent, "Critical ML issues are surfaced with model, drift, and action context.", 38, 895, 174, {
    size: 11,
    color: "#9FB1CA",
    h: 48,
    lineHeight: 16
  });
}

function topbar(parent, placeholder, right = "Last 7 days") {
  rect(parent, 264, 0, 1176, 80, "#FFFFFF", 0, C.line);
  rect(parent, 296, 18, 420, 44, C.soft, 14, C.line);
  text(parent, placeholder, 314, 31, 360, { size: 13, color: C.muted });
  rect(parent, 1110, 20, 110, 40, C.surface, 12, C.line);
  text(parent, right, 1124, 32, 82, { size: 12, style: "Bold" });
  rect(parent, 1232, 20, 104, 40, C.surface, 12, C.line);
  text(parent, "Production", 1248, 32, 72, { size: 12, style: "Bold" });
  rect(parent, 1348, 20, 40, 40, C.blueSoft, 20);
  text(parent, "AK", 1348, 31, 40, { size: 12, style: "Bold", color: C.blue, align: "CENTER" });
}

function header(parent, eyebrow, title, subtitle, chips = []) {
  text(parent, eyebrow, 296, 110, 400, { size: 11, style: "Bold", color: C.blue });
  text(parent, title, 296, 132, 720, { size: 29, style: "Bold", h: 38 });
  text(parent, subtitle, 296, 176, 760, { size: 14, color: C.muted, h: 22 });
  chips.forEach((chip, i) => {
    const w = Math.max(96, chip.length * 7 + 26);
    const x = 1104 + i * 110;
    rect(parent, x, 122, w, 36, i === 0 ? C.blueSoft : C.surface, 11, i === 0 ? "#BFD2FF" : C.line);
    text(parent, chip, x + 12, 133, w - 24, { size: 11, style: "Bold", color: i === 0 ? C.blue : C.muted });
  });
}

function chart(parent, x, y, w, h, kind = "line") {
  card(parent, x, y, w, h);
  text(parent, kind === "bar" ? "Feature distributions" : "Quality trend", x + 20, y + 18, w - 40, { size: 17, style: "Bold" });
  const gx = x + 34;
  const gy = y + 66;
  const gw = w - 68;
  const gh = h - 96;
  rect(parent, gx, gy, gw, gh, C.soft, 16);
  for (let i = 1; i < 4; i++) rect(parent, gx + 18, gy + (gh / 4) * i, gw - 36, 1, C.line, 0);
  if (kind === "bar") {
    const values = [54, 86, 42, 122, 74, 104, 36, 90];
    values.forEach((value, i) => {
      const color = i === 3 || i === 5 ? C.red : i === 4 ? C.yellow : i % 2 ? C.blue : "#B9C8DD";
      rect(parent, gx + 36 + i * 58, gy + gh - value - 20, 28, value, color, 6);
    });
    return;
  }
  const svg = `<svg width="${gw}" height="${gh}" viewBox="0 0 ${gw} ${gh}" xmlns="http://www.w3.org/2000/svg"><path d="M24 ${gh - 76} C120 ${gh - 116} 190 ${gh - 88} 270 ${gh - 120} C370 ${gh - 150} 470 ${gh - 70} ${gw - 28} ${gh - 96}" fill="none" stroke="${C.blue}" stroke-width="4" stroke-linecap="round"/><path d="M24 ${gh - 106} C130 ${gh - 130} 230 ${gh - 126} 334 ${gh - 138} C432 ${gh - 150} 512 ${gh - 118} ${gw - 28} ${gh - 132}" fill="none" stroke="${C.green}" stroke-width="4" stroke-linecap="round"/></svg>`;
  const node = figma.createNodeFromSvg(svg);
  parent.appendChild(node);
  node.x = gx;
  node.y = gy;
}

function table(parent, x, y, w, columns, rows) {
  const rowH = 42;
  const h = 54 + rows.length * rowH;
  card(parent, x, y, w, h);
  const colW = w / columns.length;
  rect(parent, x, y, w, 44, C.soft, 20, C.line);
  columns.forEach((column, i) => text(parent, column.toUpperCase(), x + i * colW + 14, y + 16, colW - 20, { size: 9, style: "Bold", color: C.muted }));
  rows.forEach((row, r) => {
    const yy = y + 48 + r * rowH;
    rect(parent, x + 12, yy + rowH - 1, w - 24, 1, C.line, 0);
    row.forEach((cell, i) => {
      const value = Array.isArray(cell) ? cell[0] : cell;
      const kind = Array.isArray(cell) ? cell[1] : null;
      if (kind) badge(parent, value, x + i * colW + 12, yy + 8, kind, 92);
      else text(parent, value, x + i * colW + 14, yy + 11, colW - 20, { size: 11, style: i === 0 ? "Bold" : "Regular", color: i === 0 ? C.text : "#213349" });
    });
  });
}

function alertCard(parent, x, y, w, title, meta, severity) {
  card(parent, x, y, w, 86);
  text(parent, title, x + 18, y + 16, w - 320, { size: 13, style: "Bold" });
  text(parent, meta, x + 18, y + 42, w - 360, { size: 11, color: C.muted, h: 28, lineHeight: 15 });
  badge(parent, severity, x + w - 300, y + 18, severity.toLowerCase(), 84);
  button(parent, "Assign", x + w - 204, y + 26, 58);
  button(parent, "Investigate", x + w - 138, y + 26, 90, true);
  button(parent, "Resolve", x + w - 40 - 72, y + 26, 72);
}

function dashboard(x) {
  const f = makeFrame("01 - Main MLOps Dashboard", x);
  sidebar(f, "Dashboard");
  topbar(f, "Search models, features, alerts...");
  header(f, "MLOPS DASHBOARD", "Production model health overview", "Track quality, drift, alerts, incidents, and production readiness.", ["Growth ML", "All models", "7D"]);
  metric(f, 296, 232, 258, "System status", "92.4%", "SLO compliance, -2.1% week over week", "warning");
  metric(f, 572, 232, 258, "Healthy models", "8", "Stable quality and drift levels", "healthy");
  metric(f, 848, 232, 258, "Warning models", "3", "Review degradation signals", "warning");
  metric(f, 1124, 232, 258, "Critical models", "1", "Immediate response required", "critical");
  chart(f, 296, 388, 724, 286, "line");
  card(f, 1040, 388, 342, 286, "Drift and model alerts");
  alertCard(f, 1062, 438, 298, "High data drift", "fraud_detection_v2 - drift score 0.81", "Critical");
  alertCard(f, 1062, 536, 298, "F1 below baseline", "churn_prediction_v3 - current 0.872", "Warning");
  table(f, 296, 696, 1086, ["Model", "Status", "F1", "Drift", "Latency", "Owner"], [
    ["churn_prediction_v3", ["Warning", "warning"], "0.872", "0.43", "96 ms", "N. Orlova"],
    ["fraud_detection_v2", ["Critical", "critical"], "0.781", "0.81", "142 ms", "M. Petrov"],
    ["demand_forecast_v1", ["Healthy", "healthy"], "0.914", "0.18", "184 ms", "A. Kim"],
    ["credit_risk_v4", ["Healthy", "healthy"], "0.928", "0.12", "88 ms", "D. Ivanov"]
  ]);
}

function modelDetail(x) {
  const f = makeFrame("02 - Model Detail", x);
  sidebar(f, "Models");
  topbar(f, "Search in churn_prediction_v3...", "Last 30 days");
  header(f, "MODEL DETAIL", "churn_prediction_v3", "Customer churn classifier monitored in production.", ["v3.8.2", "Prod", "Baseline"]);
  card(f, 296, 232, 724, 154, "Model status");
  text(f, "Quality degradation detected in the last 48 hours.", 316, 274, 500, { size: 13, color: C.muted });
  badge(f, "Warning", 888, 252, "warning", 96);
  ["Version 3.8.2", "Production", "Last deploy May 21", "Dataset retention_2026_05"].forEach((item, i) => {
    rect(f, 316 + i * 164, 318, 148, 42, C.soft, 12, C.line);
    text(f, item, 328 + i * 164, 332, 124, { size: 11, style: "Bold" });
  });
  metric(f, 296, 408, 258, "Accuracy", "0.918", "Baseline 0.936", "warning");
  metric(f, 572, 408, 258, "Precision", "0.904", "Baseline 0.908", "healthy");
  metric(f, 848, 408, 258, "Recall", "0.842", "Baseline 0.884", "warning");
  metric(f, 1124, 408, 258, "ROC-AUC", "0.931", "Baseline 0.951", "warning");
  chart(f, 296, 564, 724, 300, "line");
  card(f, 1040, 232, 342, 154, "Recommendations");
  button(f, "Start retrain", 1060, 286, 110, true);
  button(f, "Investigate", 1178, 286, 96);
  button(f, "Rollback", 1282, 286, 78);
  table(f, 1040, 564, 342, ["Metric", "Current", "Baseline"], [
    ["Accuracy", "0.918", "0.936"],
    ["Precision", "0.904", "0.908"],
    ["Recall", "0.842", "0.884"],
    ["F1-score", "0.872", "0.914"],
    ["ROC-AUC", "0.931", "0.951"]
  ]);
}

function drift(x) {
  const f = makeFrame("03 - Data Drift Monitoring", x);
  sidebar(f, "Data Drift");
  topbar(f, "Search feature drift...", "Last 24 hours");
  header(f, "DATA DRIFT MONITORING", "Train vs production distribution shift", "Feature-level drift scores with visual severity and latest checks.", ["fraud_detection_v2", "PSI + KS"]);
  metric(f, 296, 232, 258, "Overall drift", "0.62", "Threshold 0.50", "critical");
  metric(f, 572, 232, 258, "High drift features", "4", "Need investigation", "critical");
  metric(f, 848, 232, 258, "Warning features", "7", "Monitor trend", "warning");
  metric(f, 1124, 232, 258, "Last check", "5m", "Next run in 10 minutes", "info");
  chart(f, 296, 388, 724, 286, "bar");
  table(f, 296, 704, 1086, ["Feature name", "Type", "Drift score", "Status", "Last checked"], [
    ["transaction_amount", "numeric", "0.81", ["Critical", "critical"], "May 25, 14:12"],
    ["merchant_category", "categorical", "0.74", ["Critical", "critical"], "May 25, 14:12"],
    ["device_risk_score", "numeric", "0.66", ["Critical", "critical"], "May 25, 14:12"],
    ["geo_distance_km", "numeric", "0.49", ["Warning", "warning"], "May 25, 14:12"],
    ["customer_age", "numeric", "0.18", ["Healthy", "healthy"], "May 25, 14:12"]
  ]);
}

function performance(x) {
  const f = makeFrame("04 - Model Performance", x);
  sidebar(f, "Performance");
  topbar(f, "Search performance metrics...", "Last 14 days");
  header(f, "MODEL PERFORMANCE", "fraud_detection_v2 operational and business performance", "Metric dynamics, version comparison, confusion matrix, latency, throughput, and business impact.", ["Production", "v2.4.1 vs v2.3.8"]);
  metric(f, 296, 232, 200, "ROC-AUC", "0.902", "Model quality", "warning");
  metric(f, 514, 232, 200, "Latency p95", "142 ms", "Below hard limit", "warning");
  metric(f, 732, 232, 200, "Throughput", "1.8k/min", "Prediction volume", "healthy");
  metric(f, 950, 232, 200, "Error rate", "3.4%", "Above target", "critical");
  metric(f, 1168, 232, 214, "Business impact", "-$42k", "Vs baseline", "critical");
  chart(f, 296, 388, 724, 286, "line");
  table(f, 1040, 388, 342, ["Metric", "v2.4.1", "v2.3.8"], [
    ["Precision", "0.835", "0.861"],
    ["Recall", "0.881", "0.842"],
    ["F1-score", "0.857", "0.851"],
    ["False positives", "384", "311"]
  ]);
  card(f, 296, 704, 342, 206, "Confusion matrix");
  [["TN", "18,420", C.blue], ["FP", "384", "#60A5FA"], ["FN", "261", C.red], ["TP", "1,942", C.yellow]].forEach((item, i) => {
    const xx = 320 + (i % 2) * 144;
    const yy = 758 + Math.floor(i / 2) * 70;
    rect(f, xx, yy, 118, 54, item[2], 12);
    text(f, item[1], xx, yy + 10, 118, { size: 18, style: "Bold", color: "#FFFFFF", align: "CENTER" });
    text(f, item[0], xx, yy + 34, 118, { size: 10, style: "Bold", color: "#FFFFFF", align: "CENTER" });
  });
  chart(f, 660, 704, 342, 206, "bar");
  card(f, 1024, 704, 358, 206, "Business metric impact");
  text(f, "Fraud blocked: $318k", 1044, 758, 260, { size: 13, style: "Bold" });
  text(f, "False-positive cost: $71k", 1044, 792, 260, { size: 13, style: "Bold", color: C.red });
  text(f, "Missed fraud estimate: $42k", 1044, 826, 260, { size: 13, style: "Bold", color: C.red });
  badge(f, "Net -$42k", 1044, 858, "critical", 110);
}

function alerts(x) {
  const f = makeFrame("05 - Alerts and Incidents", x);
  sidebar(f, "Alerts");
  topbar(f, "Search alerts by model, owner, cause...", "Open alerts");
  header(f, "ALERTS AND INCIDENTS", "Prioritized response queue", "Severity, root cause, impacted model, status, timestamps, and quick actions.", ["New + investigating", "All owners"]);
  metric(f, 296, 232, 258, "New alerts", "7", "3 require assignment", "info");
  metric(f, 572, 232, 258, "Investigating", "5", "Avg age 38 min", "warning");
  metric(f, 848, 232, 258, "Critical", "1", "fraud_detection_v2", "critical");
  metric(f, 1124, 232, 258, "Resolved today", "12", "MTTR 26 min", "healthy");
  alertCard(f, 296, 406, 1086, "High data drift in transaction_amount", "Cause: production distribution deviates from train baseline. Model: fraud_detection_v2. Time: May 25, 14:00 UTC. Status: new.", "Critical");
  alertCard(f, 296, 510, 1086, "Recall below baseline guardrail", "Cause: model drift detected after campaign traffic change. Model: churn_prediction_v3. Time: May 25, 13:48 UTC. Status: investigating.", "Warning");
  alertCard(f, 296, 614, 1086, "Latency p95 exceeded warning threshold", "Cause: prediction service saturation during batch scoring. Model: demand_forecast_v1. Time: May 25, 13:21 UTC. Status: investigating.", "Warning");
  table(f, 296, 744, 1086, ["Incident", "Model", "Severity", "Status", "Owner"], [
    ["Feature schema mismatch", "fraud_detection_v2", ["Critical", "critical"], "New", "Unassigned"],
    ["Quality guardrail breach", "churn_prediction_v3", ["Warning", "warning"], "Investigating", "N. Orlova"],
    ["Prediction service saturation", "demand_forecast_v1", ["Warning", "warning"], "Investigating", "A. Kim"]
  ]);
}

function registry(x) {
  const f = makeFrame("06 - Model Registry", x);
  sidebar(f, "Registry");
  topbar(f, "Search registry by model, version, dataset...", "All projects");
  header(f, "MODEL REGISTRY", "Models, versions, datasets, owners, and deployment actions", "Govern lifecycle from training to deployment, rollback, and version comparison.", ["All environments", "Approved"]);
  metric(f, 296, 232, 258, "Registered models", "28", "12 active in production", "info");
  metric(f, 572, 232, 258, "Approved versions", "9", "Passing validation", "healthy");
  metric(f, 848, 232, 258, "Rollback-ready", "4", "Across critical workflows", "warning");
  metric(f, 1124, 232, 258, "Blocked", "2", "Failed drift validation", "critical");
  table(f, 296, 406, 1086, ["Model/version", "Status", "Training", "Deploy", "Dataset", "Owner"], [
    ["churn_prediction_v3 / v3.8.2", ["Warning", "warning"], "May 18", "May 21", "retention_2026_05", "N. Orlova"],
    ["fraud_detection_v2 / v2.4.1", ["Critical", "critical"], "May 12", "May 19", "fraud_txn_v9", "M. Petrov"],
    ["demand_forecast_v1 / v1.9.0", ["Healthy", "healthy"], "May 03", "May 09", "demand_weekly_v14", "A. Kim"],
    ["credit_risk_v4 / v4.2.5", ["Approved", "healthy"], "May 22", "Not deployed", "credit_app_2026_q2", "D. Ivanov"],
    ["price_elasticity_v1 / v1.3.4", ["Archived", "neutral"], "Mar 12", "Apr 01", "pricing_history_v7", "S. Lee"]
  ]);
  ["View details", "Compare versions", "Deploy", "Rollback"].forEach((label, i) => button(f, label, 296 + i * 144, 764, 130, i === 2));
}

async function main() {
  await Promise.all([
    figma.loadFontAsync({ family: "Inter", style: "Regular" }),
    figma.loadFontAsync({ family: "Inter", style: "Medium" }),
    figma.loadFontAsync({ family: "Inter", style: "Semi Bold" }),
    figma.loadFontAsync({ family: "Inter", style: "Bold" })
  ]);
  figma.currentPage.name = "ModelPulse MLOps Platform";
  screens.forEach((_, index) => {
    const x = index * 1520;
    if (index === 0) dashboard(x);
    if (index === 1) modelDetail(x);
    if (index === 2) drift(x);
    if (index === 3) performance(x);
    if (index === 4) alerts(x);
    if (index === 5) registry(x);
  });
  figma.viewport.scrollAndZoomIntoView(figma.currentPage.children);
  figma.closePlugin("Created ModelPulse MLOps monitoring dashboard screens.");
}

main();
