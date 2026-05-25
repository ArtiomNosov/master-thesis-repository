import { createTranslator, defaultLocale } from "./i18n.js";
import { activity, candidates, criteria, resumes, vacancies } from "./data.js";

const t = createTranslator(defaultLocale);

const state = {
  activeToast: "",
  candidateFilter: "",
  candidateStage: "all",
  rankingReady: true,
};

const navItems = [
  ["dashboard", "nav.dashboard", "#/dashboard"],
  ["vacancies", "nav.vacancies", "#/vacancies"],
  ["candidates", "nav.candidates", "#/candidates"],
  ["resumes", "nav.resumes", "#/resumes"],
  ["scoring", "nav.scoring", "#/scoring"],
  ["criteria", "nav.criteria", "#/criteria"],
  ["ranking", "nav.ranking", "#/ranking"],
  ["settings", "nav.settings", "#/settings"],
  ["profile", "nav.profile", "#/profile"],
  ["admin", "nav.admin", "#/admin"],
];

const routeTitles = {
  login: "auth.loginTitle",
  register: "auth.registerTitle",
  onboarding: "onboarding.title",
  dashboard: "dashboard.title",
  vacancies: "vacancies.title",
  "vacancy-detail": "vacancies.detailTitle",
  candidates: "candidates.title",
  "candidate-detail": "candidates.detailTitle",
  resumes: "resumes.title",
  scoring: "scoring.title",
  criteria: "criteria.title",
  ranking: "ranking.title",
  settings: "settings.title",
  profile: "profile.title",
  admin: "admin.title",
  error: "errorPage.title",
};

function routeInfo() {
  const hash = window.location.hash.replace(/^#\/?/, "") || "dashboard";
  const [section, id] = hash.split("/");

  if (section === "login") return { name: "login" };
  if (section === "register") return { name: "register" };
  if (section === "onboarding") return { name: "onboarding" };
  if (section === "dashboard") return { name: "dashboard" };
  if (section === "vacancies" && id) return { name: "vacancy-detail", id };
  if (section === "vacancies") return { name: "vacancies" };
  if (section === "candidates" && id) return { name: "candidate-detail", id };
  if (section === "candidates") return { name: "candidates" };
  if (section === "resumes") return { name: "resumes" };
  if (section === "scoring") return { name: "scoring" };
  if (section === "criteria") return { name: "criteria" };
  if (section === "ranking") return { name: "ranking" };
  if (section === "settings") return { name: "settings" };
  if (section === "profile") return { name: "profile" };
  if (section === "admin") return { name: "admin" };

  return { name: "error" };
}

function pageTitle(name) {
  return t(routeTitles[name] ?? "errorPage.title");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function layout(content, route) {
  const title = `${pageTitle(route.name)} | ${t("app.title")}`;
  document.title = title;

  return `
    <a class="skip-link" href="#main">${t("app.skipToContent")}</a>
    <div class="app-shell">
      <aside class="sidebar" aria-label="${t("nav.dashboard")}">
        <div class="brand">
          <div class="brand-mark" aria-hidden="true">Р</div>
          <div>
            <strong>${t("app.title")}</strong>
            <span>${t("app.subtitle")}</span>
          </div>
        </div>
        <nav class="nav-list">
          ${navItems.map(([key, label, href]) => navLink(key, label, href, route.name)).join("")}
        </nav>
        <div class="locale-card">
          <span>${t("settings.language")}</span>
          <strong>${t("app.currentLocale")}</strong>
        </div>
      </aside>
      <div class="content-shell">
        <header class="topbar">
          <div>
            <p class="eyebrow">${t("app.subtitle")}</p>
            <h1>${pageTitle(route.name)}</h1>
          </div>
          <div class="topbar-actions">
            <a class="ghost-button" href="#/login">${t("auth.signIn")}</a>
            <a class="primary-button" href="#/vacancies">${t("vacancies.createTitle")}</a>
          </div>
        </header>
        <main id="main" class="page" tabindex="-1">
          ${toastMarkup()}
          ${content}
        </main>
      </div>
    </div>
  `;
}

function navLink(key, label, href, activeRoute) {
  const active =
    activeRoute === key ||
    (activeRoute === "vacancy-detail" && key === "vacancies") ||
    (activeRoute === "candidate-detail" && key === "candidates");

  return `
    <a class="nav-link ${active ? "active" : ""}" href="${href}" aria-current="${active ? "page" : "false"}">
      <span class="nav-dot" aria-hidden="true"></span>
      ${t(label)}
    </a>
  `;
}

function toastMarkup() {
  if (!state.activeToast) return "";
  return `<div class="toast" role="status">${escapeHtml(state.activeToast)}</div>`;
}

function showToast(message) {
  state.activeToast = message;
  render();
  window.setTimeout(() => {
    state.activeToast = "";
    render();
  }, 2400);
}

function statusLabel(status) {
  const statusMap = {
    active: "vacancies.active",
    paused: "vacancies.paused",
    archived: "vacancies.archived",
    parsed: "resumes.parsed",
    waiting: "resumes.waiting",
    failed: "resumes.failed",
  };

  return t(statusMap[status] ?? "common.status");
}

function stageLabel(stage) {
  const stageMap = {
    newStage: "candidates.newStage",
    screeningStage: "candidates.screeningStage",
    interviewStage: "candidates.interviewStage",
    offerStage: "candidates.offerStage",
    rejectedStage: "candidates.rejectedStage",
  };

  return t(stageMap[stage] ?? "candidates.allStages");
}

function vacancyById(id) {
  return vacancies.find((vacancy) => vacancy.id === id) ?? vacancies[0];
}

function candidateById(id) {
  return candidates.find((candidate) => candidate.id === id) ?? candidates[0];
}

function scoreBadge(score) {
  return `<span class="score-badge">${score}%</span>`;
}

function statCard(label, value, detail) {
  return `
    <article class="card stat-card">
      <span>${label}</span>
      <strong>${value}</strong>
      <p>${detail}</p>
    </article>
  `;
}

function renderLogin() {
  return `
    <section class="auth-grid">
      <div class="hero-card">
        <p class="eyebrow">${t("app.title")}</p>
        <h2>${t("auth.loginTitle")}</h2>
        <p>${t("auth.loginLead")}</p>
        <div class="info-box">${t("auth.demoCredentials")}</div>
      </div>
      <form class="card form-card" data-form="login">
        <label>
          <span>${t("auth.emailLabel")}</span>
          <input type="email" name="email" autocomplete="email" placeholder="${t("auth.emailPlaceholder")}" required>
        </label>
        <label>
          <span>${t("auth.passwordLabel")}</span>
          <input type="password" name="password" autocomplete="current-password" placeholder="${t("auth.passwordPlaceholder")}" required>
        </label>
        <label class="checkbox-row">
          <input type="checkbox" checked>
          <span>${t("auth.rememberMe")}</span>
        </label>
        <button class="primary-button" type="submit">${t("auth.signIn")}</button>
        <a href="#/register">${t("auth.signUp")}</a>
        <a href="#/login">${t("auth.forgotPassword")}</a>
      </form>
    </section>
  `;
}

function renderRegister() {
  return `
    <section class="auth-grid">
      <div class="hero-card">
        <p class="eyebrow">${t("app.subtitle")}</p>
        <h2>${t("auth.registerTitle")}</h2>
        <p>${t("auth.registerLead")}</p>
      </div>
      <form class="card form-card" data-form="register">
        <label>
          <span>${t("auth.nameLabel")}</span>
          <input name="name" placeholder="Анна Смирнова" required>
        </label>
        <label>
          <span>${t("auth.companyLabel")}</span>
          <input name="company" placeholder="Reqcore Demo" required>
        </label>
        <label>
          <span>${t("auth.emailLabel")}</span>
          <input type="email" name="email" placeholder="${t("auth.emailPlaceholder")}" required>
        </label>
        <label>
          <span>${t("auth.passwordLabel")}</span>
          <input type="password" name="password" placeholder="${t("auth.passwordPlaceholder")}" required>
        </label>
        <button class="primary-button" type="submit">${t("auth.signUp")}</button>
      </form>
    </section>
  `;
}

function renderOnboarding() {
  return `
    <section class="grid two-columns">
      <article class="hero-card">
        <p class="eyebrow">${t("onboarding.title")}</p>
        <h2>${t("onboarding.lead")}</h2>
        <ol class="steps">
          <li>${t("onboarding.stepCompany")}</li>
          <li>${t("onboarding.stepVacancies")}</li>
          <li>${t("onboarding.stepCandidates")}</li>
        </ol>
      </article>
      <form class="card form-card" data-form="onboarding">
        <label>
          <span>${t("onboarding.teamSizeLabel")}</span>
          <select name="teamSize">
            <option>1-3</option>
            <option>4-10</option>
            <option>10+</option>
          </select>
        </label>
        <label>
          <span>${t("onboarding.specializationLabel")}</span>
          <input name="specialization" placeholder="${t("onboarding.specializationPlaceholder")}" required>
        </label>
        <label class="checkbox-row">
          <input type="checkbox" checked>
          <span>${t("onboarding.importDemoData")}</span>
        </label>
        <button class="primary-button" type="submit">${t("common.finish")}</button>
      </form>
    </section>
  `;
}

function renderDashboard() {
  return `
    <section class="page-intro">
      <p>${t("dashboard.lead")}</p>
    </section>
    <section class="stats-grid">
      ${statCard(t("dashboard.activeVacancies"), vacancies.filter((item) => item.status === "active").length, t("common.updated"))}
      ${statCard(t("dashboard.candidatesInPipeline"), candidates.length, t("dashboard.pipelineScreening"))}
      ${statCard(t("dashboard.resumesParsed"), resumes.filter((item) => item.status === "parsed").length, t("resumes.parsed"))}
      ${statCard(t("dashboard.averageMatch"), "88%", t("dashboard.modelReady"))}
    </section>
    <section class="grid two-columns">
      <article class="card">
        <h2>${t("dashboard.processingHeader")}</h2>
        <p>${t("dashboard.processingLead")}</p>
        <div class="pipeline">
          ${["pipelineNew", "pipelineScreening", "pipelineInterview", "pipelineOffer"]
            .map((key) => `<span>${t(`dashboard.${key}`)}</span>`)
            .join("")}
        </div>
      </article>
      <article class="card">
        <h2>${t("dashboard.modelStatus")}</h2>
        <p>${t("dashboard.modelReady")}</p>
        <dl class="details-list">
          <div><dt>${t("dashboard.modelVersion")}</dt><dd>rubert-tiny2 / Bi-Encoder</dd></div>
          <div><dt>ATSRanker</dt><dd>${t("common.enabled")}</dd></div>
          <div><dt>X-Model-Version</dt><dd>2026.05</dd></div>
        </dl>
      </article>
    </section>
    <section class="card">
      <h2>${t("dashboard.latestActivity")}</h2>
      <ul class="activity-list">
        ${activity.map((item) => `<li>${item}</li>`).join("") || `<li>${t("dashboard.emptyActivity")}</li>`}
      </ul>
    </section>
  `;
}

function renderVacancies() {
  return `
    <section class="page-intro">
      <p>${t("vacancies.lead")}</p>
    </section>
    <section class="grid two-columns">
      <form class="card form-card" data-form="vacancy">
        <h2>${t("vacancies.createTitle")}</h2>
        <label>
          <span>${t("vacancies.titleLabel")}</span>
          <input name="title" placeholder="${t("vacancies.titlePlaceholder")}" required>
        </label>
        <label>
          <span>${t("vacancies.departmentLabel")}</span>
          <input name="department" placeholder="${t("vacancies.departmentPlaceholder")}">
        </label>
        <label>
          <span>${t("vacancies.locationLabel")}</span>
          <input name="location" placeholder="${t("vacancies.locationPlaceholder")}">
        </label>
        <label>
          <span>${t("vacancies.jobDescriptionLabel")}</span>
          <textarea name="description" rows="5" placeholder="${t("vacancies.jobDescriptionPlaceholder")}" required></textarea>
        </label>
        <div class="button-row">
          <button class="secondary-button" type="button" data-action="generate-criteria">${t("vacancies.generateCriteria")}</button>
          <button class="primary-button" type="submit">${t("vacancies.publish")}</button>
        </div>
      </form>
      <article class="card table-card">
        <h2>${t("nav.vacancies")}</h2>
        ${vacancyTable(vacancies)}
      </article>
    </section>
  `;
}

function vacancyTable(items) {
  if (!items.length) return `<p class="empty-state">${t("vacancies.noVacancies")}</p>`;

  return `
    <div class="responsive-table">
      <table>
        <thead>
          <tr>
            <th>${t("vacancies.titleLabel")}</th>
            <th>${t("common.status")}</th>
            <th>${t("vacancies.candidates")}</th>
            <th>${t("vacancies.matchScore")}</th>
            <th>${t("common.actions")}</th>
          </tr>
        </thead>
        <tbody>
          ${items.map((vacancy) => `
            <tr>
              <td><strong>${vacancy.title}</strong><span>${vacancy.department}</span></td>
              <td><span class="status-pill">${statusLabel(vacancy.status)}</span></td>
              <td>${vacancy.candidates}</td>
              <td>${scoreBadge(vacancy.matchScore)}</td>
              <td><a href="#/vacancies/${vacancy.id}">${t("common.open")}</a></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderVacancyDetail(id) {
  const vacancy = vacancyById(id);
  const relatedCandidates = candidates.filter((candidate) => candidate.vacancyId === vacancy.id);

  return `
    <a class="back-link" href="#/vacancies">${t("common.back")}</a>
    <section class="hero-card">
      <p class="eyebrow">${statusLabel(vacancy.status)}</p>
      <h2>${vacancy.title}</h2>
      <p>${vacancy.description}</p>
      <div class="meta-row">
        <span>${vacancy.department}</span>
        <span>${vacancy.location}</span>
        <span>${t("vacancies.matchScore")}: ${vacancy.matchScore}%</span>
      </div>
    </section>
    <section class="grid three-columns">
      <article class="card">
        <h2>${t("vacancies.requirements")}</h2>
        <ul>${vacancy.requirements.map((item) => `<li>${item}</li>`).join("")}</ul>
      </article>
      <article class="card">
        <h2>${t("vacancies.responsibilities")}</h2>
        <ul>${vacancy.responsibilities.map((item) => `<li>${item}</li>`).join("")}</ul>
      </article>
      <article class="card">
        <h2>${t("vacancies.shortList")}</h2>
        <ol class="rank-list">
          ${relatedCandidates.map((candidate) => `<li><a href="#/candidates/${candidate.id}">${candidate.name}</a> ${scoreBadge(candidate.matchScore)}</li>`).join("")}
        </ol>
      </article>
    </section>
  `;
}

function renderCandidates() {
  const filteredCandidates = candidates.filter((candidate) => {
    const query = state.candidateFilter.toLocaleLowerCase("ru-RU");
    const matchesQuery =
      !query ||
      candidate.name.toLocaleLowerCase("ru-RU").includes(query) ||
      candidate.skills.join(" ").toLocaleLowerCase("ru-RU").includes(query) ||
      vacancyById(candidate.vacancyId).title.toLocaleLowerCase("ru-RU").includes(query);
    const matchesStage = state.candidateStage === "all" || candidate.stage === state.candidateStage;

    return matchesQuery && matchesStage;
  });

  return `
    <section class="page-intro">
      <p>${t("candidates.lead")}</p>
    </section>
    <section class="card filters-card">
      <label>
        <span>${t("common.search")}</span>
        <input data-filter="candidate-query" value="${escapeHtml(state.candidateFilter)}" placeholder="${t("candidates.searchPlaceholder")}">
      </label>
      <label>
        <span>${t("candidates.stageLabel")}</span>
        <select data-filter="candidate-stage">
          ${["all", "newStage", "screeningStage", "interviewStage", "offerStage", "rejectedStage"]
            .map((stage) => `<option value="${stage}" ${state.candidateStage === stage ? "selected" : ""}>${stage === "all" ? t("candidates.allStages") : stageLabel(stage)}</option>`)
            .join("")}
        </select>
      </label>
      <button class="secondary-button" type="button" data-action="clear-candidate-filters">${t("common.clear")}</button>
      <button class="primary-button" type="button" data-action="add-candidate">${t("candidates.addCandidate")}</button>
    </section>
    <section class="cards-list">
      ${filteredCandidates.length ? filteredCandidates.map(candidateCard).join("") : `<p class="empty-state">${t("candidates.noCandidates")}</p>`}
    </section>
  `;
}

function candidateCard(candidate) {
  return `
    <article class="card candidate-card">
      <div>
        <p class="eyebrow">${stageLabel(candidate.stage)}</p>
        <h2><a href="#/candidates/${candidate.id}">${candidate.name}</a></h2>
        <p>${candidate.desiredRole}</p>
        <p>${vacancyById(candidate.vacancyId).title}</p>
      </div>
      <div class="skills-list">
        ${candidate.skills.map((skill) => `<span>${skill}</span>`).join("")}
      </div>
      <dl class="details-list compact">
        <div><dt>${t("candidates.experience")}</dt><dd>${candidate.experience}</dd></div>
        <div><dt>${t("candidates.education")}</dt><dd>${candidate.education}</dd></div>
        <div><dt>${t("candidates.matchScore")}</dt><dd>${scoreBadge(candidate.matchScore)}</dd></div>
      </dl>
      <div class="button-row">
        <a class="secondary-button" href="#/candidates/${candidate.id}">${t("common.details")}</a>
        <button class="danger-button" type="button" data-action="delete-candidate">${t("common.delete")}</button>
      </div>
    </article>
  `;
}

function renderCandidateDetail(id) {
  const candidate = candidateById(id);

  return `
    <a class="back-link" href="#/candidates">${t("common.back")}</a>
    <section class="hero-card">
      <p class="eyebrow">${stageLabel(candidate.stage)}</p>
      <h2>${candidate.name}</h2>
      <p>${candidate.desiredRole}</p>
      <div class="meta-row">
        <span>${t("candidates.matchScore")}: ${candidate.matchScore}%</span>
        <span>${t("candidates.experience")}: ${candidate.experience}</span>
        <span>${t("candidates.source")}: ${candidate.source}</span>
      </div>
    </section>
    <section class="grid three-columns">
      <article class="card">
        <h2>${t("candidates.contacts")}</h2>
        <dl class="details-list">
          <div><dt>${t("auth.emailLabel")}</dt><dd>${candidate.email}</dd></div>
          <div><dt>${t("candidates.phoneLabel")}</dt><dd>${candidate.phone}</dd></div>
          <div><dt>${t("candidates.lastContact")}</dt><dd>${candidate.lastContact}</dd></div>
        </dl>
      </article>
      <article class="card">
        <h2>${t("candidates.topSkills")}</h2>
        <div class="skills-list">${candidate.skills.map((skill) => `<span>${skill}</span>`).join("")}</div>
        <dl class="details-list">
          <div><dt>${t("candidates.education")}</dt><dd>${candidate.education}</dd></div>
          <div><dt>${t("candidates.salaryExpectations")}</dt><dd>${candidate.salaryExpectations}</dd></div>
        </dl>
      </article>
      <article class="card">
        <h2>${t("candidates.notes")}</h2>
        <p>${candidate.notes}</p>
        <h3>${t("candidates.timeline")}</h3>
        <ul>${candidate.timeline.map((item) => `<li>${item}</li>`).join("")}</ul>
      </article>
    </section>
  `;
}

function renderResumes() {
  return `
    <section class="page-intro">
      <p>${t("resumes.lead")}</p>
    </section>
    <section class="grid two-columns">
      <form class="card form-card" data-form="resume">
        <h2>${t("resumes.uploadResume")}</h2>
        <p>${t("resumes.uploadLead")}</p>
        <label>
          <span>${t("resumes.fileLabel")}</span>
          <input type="file" name="resumeFile" accept=".pdf,.doc,.docx,.txt">
        </label>
        <label>
          <span>${t("resumes.vacancyLabel")}</span>
          <select name="vacancy">
            ${vacancies.map((vacancy) => `<option value="${vacancy.id}">${vacancy.title}</option>`).join("")}
          </select>
        </label>
        <button class="primary-button" type="submit">${t("resumes.uploadResume")}</button>
      </form>
      <article class="card table-card">
        <h2>${t("nav.resumes")}</h2>
        <div class="responsive-table">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>${t("candidates.candidate")}</th>
                <th>${t("resumes.vacancyLabel")}</th>
                <th>${t("resumes.parseStatus")}</th>
                <th>${t("resumes.parsedFields")}</th>
              </tr>
            </thead>
            <tbody>
              ${resumes.map((resume) => `
                <tr>
                  <td>${resume.id}</td>
                  <td>${resume.candidateName}</td>
                  <td>${vacancyById(resume.vacancyId).title}</td>
                  <td><span class="status-pill">${statusLabel(resume.status)}</span></td>
                  <td>${resume.parsedFields.join(", ")}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  `;
}

function renderScoring() {
  return `
    <section class="page-intro">
      <p>${t("scoring.lead")}</p>
    </section>
    <section class="grid two-columns">
      <form class="card form-card" data-form="scoring">
        <h2>${t("scoring.formTitle")}</h2>
        <label>
          <span>${t("scoring.jobDescription")}</span>
          <textarea name="jobDescription" rows="6" placeholder="${t("vacancies.jobDescriptionPlaceholder")}" required>${vacancies[0].description}</textarea>
        </label>
        <label>
          <span>${t("scoring.resumeText")}</span>
          <textarea name="resumeText" rows="6" placeholder="${t("scoring.resumeTextPlaceholder")}" required>${candidates[0].notes}</textarea>
        </label>
        <button class="primary-button" type="submit">${t("scoring.scoreCandidate")}</button>
      </form>
      <article class="card result-card">
        <h2>${t("scoring.scoreReady")}</h2>
        <div class="score-ring">94%</div>
        <p>${t("scoring.scoreExplain")}</p>
        <dl class="details-list">
          <div><dt>${t("scoring.confidence")}</dt><dd>0.91</dd></div>
          <div><dt>${t("scoring.processingTime")}</dt><dd>184 ms</dd></div>
          <div><dt>POST /submit</dt><dd>job_description, resumes</dd></div>
        </dl>
      </article>
    </section>
    <section class="card">
      <h2>${t("scoring.apiPanelTitle")}</h2>
      <p>${t("scoring.apiPanelLead")}</p>
      <code>POST /index/candidate</code>
      <code>POST /search/rank</code>
      <code>POST /score</code>
      <code>JSON: candidate_id, vacancy_id, match_score</code>
    </section>
  `;
}

function renderCriteria() {
  return `
    <section class="page-intro">
      <p>${t("criteria.lead")}</p>
    </section>
    <section class="grid two-columns">
      <form class="card form-card" data-form="criteria">
        <h2>${t("criteria.generatorTitle")}</h2>
        <label>
          <span>${t("criteria.vacancyInputLabel")}</span>
          <textarea rows="8" name="vacancyText" placeholder="${t("criteria.vacancyInputPlaceholder")}" required>${vacancies[0].description}</textarea>
        </label>
        <button class="primary-button" type="submit">${t("criteria.generate")}</button>
      </form>
      <article class="card">
        <h2>${t("criteria.generatedTitle")}</h2>
        <ul class="criteria-list">
          ${criteria.map((criterion) => `
            <li>
              <span>${criterion.label}</span>
              <strong>${criterion.weight}%</strong>
              <em>${t(`criteria.${criterion.type}`)}</em>
            </li>
          `).join("")}
        </ul>
        <label>
          <span>${t("criteria.manualCriterion")}</span>
          <input placeholder="${t("criteria.addCriterion")}">
        </label>
      </article>
    </section>
  `;
}

function renderRanking() {
  const rankedCandidates = [...candidates].sort((left, right) => right.matchScore - left.matchScore);

  return `
    <section class="page-intro">
      <p>${t("ranking.lead")}</p>
    </section>
    <section class="grid two-columns">
      <form class="card form-card" data-form="ranking">
        <h2>${t("ranking.filtersTitle")}</h2>
        <label>
          <span>${t("ranking.vacancyLabel")}</span>
          <select name="vacancy">
            ${vacancies.map((vacancy) => `<option value="${vacancy.id}">${vacancy.title}</option>`).join("")}
          </select>
        </label>
        <label>
          <span>${t("ranking.minScore")}</span>
          <input type="number" min="0" max="100" value="70">
        </label>
        <label>
          <span>${t("ranking.topK")}</span>
          <input type="number" min="1" max="10" value="5">
        </label>
        <button class="primary-button" type="submit">${t("ranking.runRanking")}</button>
      </form>
      <article class="card table-card">
        <h2>${t("ranking.topCandidates")}</h2>
        ${state.rankingReady ? rankingTable(rankedCandidates) : `<p class="empty-state">${t("ranking.noRanking")}</p>`}
      </article>
    </section>
  `;
}

function rankingTable(items) {
  return `
    <div class="responsive-table">
      <table>
        <thead>
          <tr>
            <th>${t("ranking.rank")}</th>
            <th>${t("ranking.candidate")}</th>
            <th>${t("ranking.matchScore")}</th>
            <th>${t("ranking.explanation")}</th>
          </tr>
        </thead>
        <tbody>
          ${items.map((candidate, index) => `
            <tr>
              <td>${index + 1}</td>
              <td><a href="#/candidates/${candidate.id}">${candidate.name}</a></td>
              <td>${scoreBadge(candidate.matchScore)}</td>
              <td>${candidate.skills.slice(0, 3).join(", ")}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderSettings() {
  return `
    <section class="page-intro">
      <p>${t("settings.lead")}</p>
    </section>
    <form class="card form-card wide-form" data-form="settings">
      <label>
        <span>${t("settings.language")}</span>
        <select name="language">
          <option value="ru-RU">${t("settings.russian")}</option>
        </select>
      </label>
      <label>
        <span>${t("settings.apiBaseUrl")}</span>
        <input name="apiBaseUrl" placeholder="${t("settings.apiBaseUrlPlaceholder")}" value="http://127.0.0.1:8000">
      </label>
      <label>
        <span>${t("settings.scoringThreshold")}</span>
        <input type="number" min="0" max="100" value="75">
      </label>
      <label class="checkbox-row">
        <input type="checkbox" checked>
        <span>${t("settings.notifications")}</span>
      </label>
      <section class="nested-card">
        <h2>${t("settings.modelSettings")}</h2>
        <code>MODEL_PATH</code>
        <code>ATSRanker</code>
        <code>rubert-tiny2</code>
        <code>Bi-Encoder</code>
      </section>
      <section class="nested-card">
        <h2>${t("settings.technicalHeaders")}</h2>
        <code>X-Processing-Time-Ms</code>
        <code>X-Model-Version</code>
      </section>
      <button class="primary-button" type="submit">${t("common.save")}</button>
    </form>
  `;
}

function renderProfile() {
  return `
    <section class="page-intro">
      <p>${t("profile.lead")}</p>
    </section>
    <form class="card form-card wide-form" data-form="profile">
      <label>
        <span>${t("profile.fullName")}</span>
        <input value="Анна Смирнова" required>
      </label>
      <label>
        <span>${t("profile.position")}</span>
        <input value="Ведущий рекрутер" required>
      </label>
      <label>
        <span>${t("profile.team")}</span>
        <input value="Команда подбора AI Engine">
      </label>
      <section class="nested-card">
        <h2>${t("profile.accountSecurity")}</h2>
        <button class="secondary-button" type="button" data-action="change-password">${t("profile.changePassword")}</button>
      </section>
      <button class="primary-button" type="submit">${t("common.save")}</button>
    </form>
  `;
}

function renderAdmin() {
  const users = [
    ["Анна Смирнова", t("admin.roleAdmin"), "anna@example.ru"],
    ["Павел Орлов", t("admin.roleRecruiter"), "pavel@example.ru"],
    ["Елена Морозова", t("admin.roleHiringManager"), "elena@example.ru"],
  ];

  return `
    <section class="page-intro">
      <p>${t("admin.lead")}</p>
    </section>
    <section class="grid two-columns">
      <form class="card form-card" data-form="admin">
        <h2>${t("admin.inviteUser")}</h2>
        <label>
          <span>${t("auth.emailLabel")}</span>
          <input type="email" placeholder="${t("auth.emailPlaceholder")}" required>
        </label>
        <label>
          <span>${t("common.role")}</span>
          <select>
            <option>${t("admin.roleRecruiter")}</option>
            <option>${t("admin.roleHiringManager")}</option>
            <option>${t("admin.roleAdmin")}</option>
          </select>
        </label>
        <button class="primary-button" type="submit">${t("admin.inviteUser")}</button>
      </form>
      <article class="card table-card">
        <h2>${t("admin.users")}</h2>
        <div class="responsive-table">
          <table>
            <thead>
              <tr>
                <th>${t("auth.nameLabel")}</th>
                <th>${t("common.role")}</th>
                <th>${t("auth.emailLabel")}</th>
              </tr>
            </thead>
            <tbody>
              ${users.map(([name, role, email]) => `
                <tr>
                  <td>${name}</td>
                  <td>${role}</td>
                  <td>${email}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </article>
    </section>
    <section class="card">
      <h2>${t("admin.auditLog")}</h2>
      <ul class="activity-list">
        <li>${t("admin.permissions")} ${t("common.updated").toLocaleLowerCase("ru-RU")}</li>
        <li>${t("vacancies.criteriaGenerated")}</li>
      </ul>
    </section>
  `;
}

function renderError() {
  return `
    <section class="hero-card">
      <h2>${t("errorPage.title")}</h2>
      <p>${t("errorPage.lead")}</p>
      <a class="primary-button" href="#/dashboard">${t("errorPage.action")}</a>
    </section>
  `;
}

function renderContent(route) {
  if (route.name === "login") return renderLogin();
  if (route.name === "register") return renderRegister();
  if (route.name === "onboarding") return renderOnboarding();
  if (route.name === "dashboard") return renderDashboard();
  if (route.name === "vacancies") return renderVacancies();
  if (route.name === "vacancy-detail") return renderVacancyDetail(route.id);
  if (route.name === "candidates") return renderCandidates();
  if (route.name === "candidate-detail") return renderCandidateDetail(route.id);
  if (route.name === "resumes") return renderResumes();
  if (route.name === "scoring") return renderScoring();
  if (route.name === "criteria") return renderCriteria();
  if (route.name === "ranking") return renderRanking();
  if (route.name === "settings") return renderSettings();
  if (route.name === "profile") return renderProfile();
  if (route.name === "admin") return renderAdmin();
  return renderError();
}

function render() {
  const route = routeInfo();
  document.documentElement.lang = "ru";
  document.getElementById("app").innerHTML = layout(renderContent(route), route);
  bindEvents();
}

function bindEvents() {
  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", handleSubmit);
  });

  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", handleAction);
  });

  const candidateQuery = document.querySelector("[data-filter='candidate-query']");
  if (candidateQuery) {
    candidateQuery.addEventListener("input", (event) => {
      state.candidateFilter = event.target.value;
      render();
    });
  }

  const candidateStage = document.querySelector("[data-filter='candidate-stage']");
  if (candidateStage) {
    candidateStage.addEventListener("change", (event) => {
      state.candidateStage = event.target.value;
      render();
    });
  }
}

function handleSubmit(event) {
  event.preventDefault();
  const formName = event.currentTarget.dataset.form;
  const messageMap = {
    login: "auth.loginSuccess",
    register: "auth.registerSuccess",
    onboarding: "onboarding.completed",
    vacancy: "vacancies.saved",
    resume: "resumes.uploadSuccess",
    scoring: "scoring.scoreReady",
    criteria: "criteria.success",
    ranking: "ranking.rankingReady",
    settings: "settings.saveSuccess",
    profile: "profile.saveSuccess",
    admin: "admin.inviteSuccess",
  };

  if (formName === "ranking") {
    state.rankingReady = true;
  }

  showToast(t(messageMap[formName] ?? "toast.saved"));
}

function handleAction(event) {
  const action = event.currentTarget.dataset.action;
  if (action === "clear-candidate-filters") {
    state.candidateFilter = "";
    state.candidateStage = "all";
    render();
    return;
  }

  const messageMap = {
    "generate-criteria": "vacancies.criteriaGenerated",
    "add-candidate": "candidates.addSuccess",
    "delete-candidate": "candidates.deleteConfirm",
    "change-password": "toast.saved",
  };

  showToast(t(messageMap[action] ?? "toast.saved"));
}

window.addEventListener("hashchange", render);
window.addEventListener("DOMContentLoaded", render);
