import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const workspaceRoot = resolve(projectRoot, "..");
const screenshotDir = resolve(workspaceRoot, "artifacts", "i18n-ru-screenshots");
const appPort = Number(process.env.PORT ?? 4173);
const debugPort = Number(process.env.CHROME_DEBUG_PORT ?? 9224);
const baseUrl = `http://127.0.0.1:${appPort}`;

const routes = [
  ["login", "#/login"],
  ["register", "#/register"],
  ["onboarding", "#/onboarding"],
  ["dashboard", "#/dashboard"],
  ["vacancies", "#/vacancies"],
  ["vacancy-detail", "#/vacancies/vacancy-analytics-crm"],
  ["candidates", "#/candidates"],
  ["candidate-detail", "#/candidates/candidate-irina-volkova"],
  ["resumes", "#/resumes"],
  ["scoring", "#/scoring"],
  ["criteria", "#/criteria"],
  ["ranking", "#/ranking"],
  ["settings", "#/settings"],
  ["profile", "#/profile"],
  ["admin", "#/admin"],
];

const allowedLatin = new Set([
  "AI",
  "API",
  "ATS",
  "ATSRanker",
  "Backend",
  "Bi",
  "Bi-Encoder",
  "Boolean",
  "CRM",
  "DOCX",
  "Docker",
  "Engine",
  "FastAPI",
  "ID",
  "JSON",
  "LinkedIn",
  "MODEL",
  "MODEL_PATH",
  "PDF",
  "POST",
  "PostgreSQL",
  "Power",
  "Python",
  "Reqcore",
  "SQL",
  "Top",
  "Top-K",
  "URL",
  "X-Model-Version",
  "X-Processing-Time-Ms",
  "anna",
  "anton",
  "candidate_id",
  "demo-password",
  "elena",
  "example",
  "irina",
  "job_description",
  "kim",
  "maria",
  "match_score",
  "ms",
  "pavel",
  "recruiter",
  "resumes",
  "ru",
  "rubert-tiny2",
  "score",
  "search",
  "sokolova",
  "vacancy_id",
  "volkova",
]);

const server = spawn(process.execPath, [resolve(projectRoot, "scripts", "serve.js")], {
  cwd: projectRoot,
  env: { ...process.env, PORT: String(appPort) },
  stdio: ["ignore", "pipe", "pipe"],
});

const chromePath = findChrome();
const chrome = spawn(chromePath, [
  "--headless=new",
  "--disable-gpu",
  "--no-sandbox",
  "--hide-scrollbars",
  `--remote-debugging-port=${debugPort}`,
  `--user-data-dir=/tmp/ats-ui-chrome-${Date.now()}`,
  "about:blank",
], {
  stdio: ["ignore", "pipe", "pipe"],
});

try {
  await mkdir(screenshotDir, { recursive: true });
  await waitForHttp(`${baseUrl}/`);
  await waitForHttp(`http://127.0.0.1:${debugPort}/json/version`);

  const results = [];

  for (const [slug, hash] of routes) {
    const page = await openPage(`${baseUrl}/${hash}`);
    const cdp = await connectCdp(page.webSocketDebuggerUrl);
    const loaded = cdp.waitForEvent("Page.loadEventFired", 5000).catch(() => null);

    await cdp.send("Page.enable");
    await cdp.send("Runtime.enable");
    await cdp.send("Emulation.setDeviceMetricsOverride", {
      width: 1440,
      height: 1200,
      deviceScaleFactor: 1,
      mobile: false,
    });
    await cdp.send("Page.navigate", { url: `${baseUrl}/${hash}` });
    await loaded;
    await delay(600);

    const textResult = await cdp.send("Runtime.evaluate", {
      expression: "document.body.innerText",
      returnByValue: true,
    });
    const text = textResult.result?.value ?? "";
    const unexpectedLatin = collectUnexpectedLatin(text);

    const screenshot = await cdp.send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: true,
      fromSurface: true,
    });
    const screenshotPath = resolve(screenshotDir, `${slug}.png`);
    await writeFile(screenshotPath, Buffer.from(screenshot.data, "base64"));
    cdp.close();

    results.push({
      route: hash,
      url: `${baseUrl}/${hash}`,
      screenshot: screenshotPath,
      unexpectedLatin,
    });
  }

  const report = {
    baseUrl,
    checkedAt: new Date().toISOString(),
    routes: results,
    technicalLatinAllowed: [...allowedLatin].sort(),
  };

  await writeFile(resolve(screenshotDir, "dom-language-report.json"), `${JSON.stringify(report, null, 2)}\n`);

  const failures = results.filter((result) => result.unexpectedLatin.length > 0);
  if (failures.length > 0) {
    console.error(JSON.stringify(failures, null, 2));
    process.exitCode = 1;
  } else {
    console.log(`Скриншоты сохранены: ${screenshotDir}`);
    console.log("DOM-проверка завершена: непереведённый пользовательский английский текст не найден.");
  }
} finally {
  chrome.kill("SIGTERM");
  server.kill("SIGTERM");
}

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    "/usr/local/bin/google-chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);

  const path = candidates.find((candidate) => existsSync(candidate));
  if (!path) {
    throw new Error("Не найден Chrome для браузерной проверки.");
  }
  return path;
}

async function waitForHttp(url, timeoutMs = 15000) {
  const startedAt = Date.now();
  let lastError;

  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (error) {
      lastError = error;
    }
    await delay(150);
  }

  throw lastError ?? new Error(`Не удалось дождаться ${url}`);
}

async function openPage(url) {
  const response = await fetch(
    `http://127.0.0.1:${debugPort}/json/new?${encodeURIComponent(url)}`,
    { method: "PUT" },
  );

  if (!response.ok) {
    throw new Error(`Chrome не открыл страницу ${url}: ${response.status}`);
  }

  return response.json();
}

async function connectCdp(webSocketDebuggerUrl) {
  const socket = new WebSocket(webSocketDebuggerUrl);
  const pending = new Map();
  const listeners = new Map();
  let id = 0;

  await new Promise((resolveOpen, rejectOpen) => {
    socket.addEventListener("open", resolveOpen, { once: true });
    socket.addEventListener("error", rejectOpen, { once: true });
  });

  socket.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.id && pending.has(payload.id)) {
      const { resolveRequest, rejectRequest } = pending.get(payload.id);
      pending.delete(payload.id);
      if (payload.error) rejectRequest(new Error(payload.error.message));
      else resolveRequest(payload.result ?? {});
      return;
    }

    const routeListeners = listeners.get(payload.method) ?? [];
    for (const listener of routeListeners) listener(payload.params ?? {});
  });

  return {
    send(method, params = {}) {
      id += 1;
      socket.send(JSON.stringify({ id, method, params }));
      return new Promise((resolveRequest, rejectRequest) => {
        pending.set(id, { resolveRequest, rejectRequest });
      });
    },
    waitForEvent(method, timeoutMs = 5000) {
      return new Promise((resolveEvent, rejectEvent) => {
        const timer = setTimeout(() => rejectEvent(new Error(`Нет события ${method}`)), timeoutMs);
        const listener = (params) => {
          clearTimeout(timer);
          listeners.set(
            method,
            (listeners.get(method) ?? []).filter((entry) => entry !== listener),
          );
          resolveEvent(params);
        };
        listeners.set(method, [...(listeners.get(method) ?? []), listener]);
      });
    },
    close() {
      socket.close();
    },
  };
}

function collectUnexpectedLatin(text) {
  const words = text.match(/[A-Za-z][A-Za-z0-9_-]*(?:-[A-Za-z0-9_]+)*/g) ?? [];
  return [...new Set(words)]
    .filter((word) => !allowedLatin.has(word))
    .filter((word) => !/^[a-z]+@/.test(word))
    .sort();
}

function delay(ms) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, ms));
}
