import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { messages } from "../src/i18n.js";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const requiredBuildFiles = ["index.html", "app.js", "i18n.js", "data.js", "styles.css"];
const requiredRouteSections = [
  "login",
  "register",
  "onboarding",
  "dashboard",
  "vacancies",
  "candidates",
  "resumes",
  "scoring",
  "criteria",
  "ranking",
  "settings",
  "profile",
  "admin",
];

for (const fileName of requiredBuildFiles) {
  await access(resolve(projectRoot, "dist", fileName));
}

const appSource = await readFile(resolve(projectRoot, "src", "app.js"), "utf8");
const missingRoutes = requiredRouteSections.filter((route) => !appSource.includes(`"${route}"`));
if (missingRoutes.length > 0) {
  throw new Error(`Не найдены маршруты: ${missingRoutes.join(", ")}`);
}

const ru = messages["ru-RU"];
const smokeStrings = [
  ru.dashboard.title,
  ru.vacancies.title,
  ru.candidates.title,
  ru.resumes.uploadResume,
  ru.scoring.scoreReady,
  ru.criteria.generatedTitle,
  ru.ranking.topCandidates,
  ru.settings.russian,
  ru.profile.title,
  ru.admin.users,
];

for (const text of smokeStrings) {
  if (!text || /Dashboard|Candidates|Vacancies|Upload resume|Match score|Ranking/.test(text)) {
    throw new Error(`Некорректная строка локализации: ${text}`);
  }
}

console.log("Smoke-test ATS UI пройден: сборка и ключевые маршруты доступны.");
