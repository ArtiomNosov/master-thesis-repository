export const vacancies = [
  {
    id: "vacancy-analytics-crm",
    title: "Старший аналитик данных в CRM",
    department: "Аналитика и CRM",
    location: "Москва / гибрид",
    status: "active",
    candidates: 34,
    matchScore: 92,
    description:
      "Развитие аналитики клиентского направления, подготовка управленческой отчётности и поиск точек роста в CRM-коммуникациях.",
    requirements: [
      "Опыт аналитики данных от 3 лет",
      "SQL, Python, BI-инструменты",
      "Понимание CRM-метрик и клиентских сегментов",
    ],
    responsibilities: [
      "Собирать требования внутренних заказчиков",
      "Готовить витрины данных и отчёты",
      "Оценивать влияние кампаний на ключевые показатели",
    ],
  },
  {
    id: "vacancy-backend-python",
    title: "Бэкенд-разработчик Python",
    department: "Платформа AI Engine",
    location: "Удалённо",
    status: "active",
    candidates: 19,
    matchScore: 87,
    description:
      "Разработка API для скоринга, интеграции с системой подбора и сервисов обработки резюме.",
    requirements: [
      "Python, FastAPI, PostgreSQL",
      "Опыт проектирования API",
      "Понимание очередей и фоновой обработки",
    ],
    responsibilities: [
      "Поддерживать сервисы индексирования кандидатов",
      "Оптимизировать обработку больших payload",
      "Участвовать в улучшении observability",
    ],
  },
  {
    id: "vacancy-recruiter-it",
    title: "IT-рекрутер",
    department: "HR-команда",
    location: "Санкт-Петербург",
    status: "paused",
    candidates: 12,
    matchScore: 78,
    description:
      "Полный цикл подбора IT-специалистов: от брифа с нанимающим менеджером до оффера.",
    requirements: [
      "Опыт IT-подбора от 2 лет",
      "Умение работать с воронкой кандидатов",
      "Навыки коммуникации с техническими командами",
    ],
    responsibilities: [
      "Вести вакансии в системе подбора",
      "Проводить первичные интервью",
      "Поддерживать кандидатов на всех этапах",
    ],
  },
];

export const candidates = [
  {
    id: "candidate-irina-volkova",
    name: "Ирина Волкова",
    desiredRole: "Старший аналитик данных",
    stage: "screeningStage",
    vacancyId: "vacancy-analytics-crm",
    matchScore: 94,
    experience: "6 лет",
    education: "НИУ ВШЭ, прикладная математика",
    skills: ["SQL", "Python", "Power BI", "CRM-аналитика"],
    salaryExpectations: "от 260 000 ₽",
    source: "Рекомендация",
    lastContact: "Сегодня",
    email: "irina.volkova@example.ru",
    phone: "+7 999 123-45-67",
    notes:
      "Сильный опыт CRM-кампаний и продуктовой аналитики. Хорошо объясняет бизнес-эффект решений.",
    timeline: [
      "Резюме загружено и распознано",
      "AI-оценка показала высокую оценку соответствия",
      "Назначен технический скрининг",
    ],
  },
  {
    id: "candidate-anton-kim",
    name: "Антон Ким",
    desiredRole: "Бэкенд-разработчик Python",
    stage: "interviewStage",
    vacancyId: "vacancy-backend-python",
    matchScore: 89,
    experience: "5 лет",
    education: "СПбГУ, программная инженерия",
    skills: ["Python", "FastAPI", "PostgreSQL", "Docker"],
    salaryExpectations: "от 300 000 ₽",
    source: "Карьерный сайт",
    lastContact: "Вчера",
    email: "anton.kim@example.ru",
    phone: "+7 999 555-17-88",
    notes:
      "Уверенно проектирует API, есть опыт производительных сервисов и асинхронной обработки.",
    timeline: [
      "Откликнулся на вакансию",
      "Прошёл скрининг рекрутера",
      "Ожидает интервью с командой",
    ],
  },
  {
    id: "candidate-maria-sokolova",
    name: "Мария Соколова",
    desiredRole: "IT-рекрутер",
    stage: "newStage",
    vacancyId: "vacancy-recruiter-it",
    matchScore: 81,
    experience: "4 года",
    education: "РГГУ, управление персоналом",
    skills: ["Поиск кандидатов", "Интервью", "Системы подбора", "Коммуникации"],
    salaryExpectations: "от 180 000 ₽",
    source: "LinkedIn",
    lastContact: "3 дня назад",
    email: "maria.sokolova@example.ru",
    phone: "+7 999 777-41-20",
    notes:
      "Хорошо знает IT-рынок и умеет выстраивать партнёрство с нанимающими менеджерами.",
    timeline: [
      "Профиль добавлен вручную",
      "Назначен первичный звонок",
    ],
  },
];

export const resumes = [
  {
    id: "resume-101",
    candidateName: "Ирина Волкова",
    vacancyId: "vacancy-analytics-crm",
    status: "parsed",
    parsedFields: ["Навыки", "Опыт", "Образование", "Контакты"],
  },
  {
    id: "resume-102",
    candidateName: "Антон Ким",
    vacancyId: "vacancy-backend-python",
    status: "parsed",
    parsedFields: ["Навыки", "Опыт", "Проекты", "Контакты"],
  },
  {
    id: "resume-103",
    candidateName: "Мария Соколова",
    vacancyId: "vacancy-recruiter-it",
    status: "waiting",
    parsedFields: ["Контакты"],
  },
];

export const criteria = [
  { label: "Профильный опыт", weight: 35, type: "mustHave" },
  { label: "Ключевые навыки по вакансии", weight: 30, type: "mustHave" },
  { label: "Релевантные проекты", weight: 20, type: "niceToHave" },
  { label: "Коммуникация с бизнес-заказчиком", weight: 15, type: "niceToHave" },
];

export const activity = [
  "Ирина Волкова перемещена на этап скрининга",
  "Для вакансии «Бэкенд-разработчик Python» обновлены критерии оценки",
  "Загружено резюме Марии Соколовой",
  "Ранжирование по вакансии «Старший аналитик данных в CRM» завершено",
];
