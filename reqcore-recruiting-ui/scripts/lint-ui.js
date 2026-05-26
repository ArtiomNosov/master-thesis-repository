import { messages } from "../src/i18n.js";

const allowedLatin = new Set([
  "AI",
  "API",
  "Bi-Encoder",
  "CRM",
  "DOCX",
  "Engine",
  "FastAPI",
  "ID",
  "JSON",
  "MODEL",
  "PATH",
  "PDF",
  "Reqcore",
  "Top",
  "URL",
  "demo-password",
  "endpoint",
  "example",
  "http",
  "job",
  "k",
  "password",
  "recruiter",
  "resumes",
  "ru",
  "score",
  "top-k",
]);

const requiredKeys = [
  "nav.dashboard",
  "nav.vacancies",
  "nav.candidates",
  "nav.resumes",
  "nav.scoring",
  "nav.criteria",
  "nav.ranking",
  "auth.loginTitle",
  "vacancies.jobDescriptionLabel",
  "candidates.skills",
  "candidates.experience",
  "candidates.education",
  "resumes.uploadResume",
  "scoring.scoreReady",
  "criteria.generatedTitle",
  "ranking.topCandidates",
  "common.loading",
  "common.noResults",
  "common.error",
  "common.success",
];

const failures = [];

for (const [locale, dictionary] of Object.entries(messages)) {
  for (const key of requiredKeys) {
    const value = getValue(dictionary, key);
    if (typeof value !== "string" || !value.trim()) {
      failures.push(`${locale}: отсутствует перевод ${key}`);
    }
  }

  for (const [key, value] of flatten(dictionary)) {
    const latinWords = value.match(/[A-Za-z][A-Za-z0-9-]*/g) ?? [];
    const unexpectedWords = latinWords
      .map((word) => word.replace(/^-+|-+$/g, ""))
      .filter((word) => !allowedLatin.has(word));
    if (unexpectedWords.length > 0) {
      failures.push(`${locale}.${key}: проверьте латиницу "${unexpectedWords.join(", ")}"`);
    }
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("Локализация интерфейса подбора проверена: обязательные русские строки на месте.");

function getValue(dictionary, key) {
  return key.split(".").reduce((entry, part) => entry?.[part], dictionary);
}

function flatten(value, prefix = "") {
  if (typeof value === "string") {
    return [[prefix, value]];
  }

  return Object.entries(value).flatMap(([key, entry]) =>
    flatten(entry, prefix ? `${prefix}.${key}` : key),
  );
}
