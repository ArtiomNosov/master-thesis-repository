Все данные собраны. У меня полный текст диссертации, полная архитектура Reqcore (scoring.ts, analyze.post.ts, DB-схема, provider.ts), и весь ML-пайплайн (17 скриптов, ATSRanker, FastAPI). Работаю по 8 секциям.

1. Аудит проблем
1.1. Прямые расхождения кода и текста
№	Что написано в диссертации	Что в коде	Критичность
1	Маршрут POST /rank (стр. 88)	В коде: POST /search/rank	Фактическая ошибка
2	Маршруты POST /embed/job, POST /embed/resume (стр. 92)	Не существуют ни в Reqcore, ни в 11_api_server.py	Вымышленный API
3	POST /index/upsert, POST /index/delete (стр. 96)	В коде: POST /index/candidate (только insert)	Вымышленный API
4	POST /monitoring/log (стр. 99)	Не существует в коде	Вымышленный API
5	batch_size = 16 (стр. 155)	09_train_biencoder.py: batch_size=8	Неверный параметр
6	Модель «~118 МБ» (стр. 114)	model.safetensors: 112 МБ	Мелкая неточность
7	«ru-en-RoSBERTa внедрена» (стр. 119)	Код использует только cointegrated/rubert-tiny2	Ложное утверждение
8	LTR/RankNet на этапе Re-ranking (стр. 125)	Код делает только cosine_similarity, без LTR	Ложная архитектура
9	Vector DB Milvus интегрирован (стр. 80)	Код: CANDIDATE_INDEX: list[dict] (in-memory list)	Ложная архитектура
10	DiskANN/SPANN «подтвердилась эффективность» (стр. 146)	Нет в коде	Выдумка
11	«Механизмы ограничений по экспозиции» (стр. 129)	Нет в коде	Выдумка
12	«Периодический аудит предвзятости» (стр. 120)	Нет в коде	Выдумка
1.2. Необоснованные утверждения
№	Утверждение	Проблема
1	«Сокращает время скрининга на порядок» (стр. 160)	Нет расчёта. Нет сравнения «до/после».
2	«Экономически оправдана» (стр. 160)	Нет таблицы затрат. Нет цифр.
3	«Экономическая модель из трёх компонентов» (стр. 153)	Чистая проза без единого числа.
4	«При масштабировании до миллионов кандидатов» (стр. 124)	Тест проведён максимум на 5000 (15_performance_benchmark.py).
5	«Валидация при участии HR-отдела» (стр. 140)	Нет протокола, нет данных, нет артефакта.
6	PROSPECT: «улучшила выдачу на 30%» (стр. 151)	Чужой результат из [3], выдан как наш.
1.3. Структурные проблемы текста
Стр. 107 — «Совокупный программный комплекс (ATS AI Engine) состоит из 17 взаимосвязанных Python-скриптов...» — именно то, что пользователь просил не писать. Общее описание вместо цитирования кода.
Нигде в тексте нет ни одного фрагмента кода — только пересказ.
Нигде нет таблиц с параметрами модели, метриками, конфигурацией.
Интеграция с Reqcore не описана — текст описывает абстрактный FastAPI-сервер, но не объясняет, как он подключается к реальному ATS.
Нет ноутбука, нет команд запуска, нет процедуры hot swap модели.
Нет A/B-теста — ни дизайна, ни метрик.
1.4. Критический разрыв: два мира не соединены
В Reqcore скоринг работает так:

analyze.post.ts → scoreApplication() → LLM (OpenAI/Anthropic) → criterionScore[] → application.score

В ML-модуле скоринг работает так:

11_api_server.py → ATSRanker.search() → cosine_similarity → JSON

В текущем виде они никак не связаны. Диссертация не описывает, как один заменяет другой.

2. План минимальных правок
2.1. Текст диссертации
Что	Действие
Вымышленные API-маршруты (стр. 88–100)	Заменить на реальные: POST /search/rank, POST /index/candidate
Вымышленная архитектура (LTR, Milvus, DiskANN, SPANN, fair ranking)	Удалить или пометить как «направление развития»
Абзац с «17 скриптов» (стр. 107)	Удалить, заменить цитатами кода с пояснениями
Экономическая оценка (стр. 148–155)	Заменить таблицей сравнения «LLM-скоринг vs Bi-Encoder»
Конфигурация модели	Вынести в таблицу: архитектура, hidden_size, слои, размер, гиперпараметры
Метрики	Вынести в таблицу: NDCG@3, P@3, R@3, MRR с указанием тестового набора
Интеграция с Reqcore	Новый подраздел: точка врезки, изменяемые файлы, контракт данных
A/B-тест	Новый подраздел: дизайн, целевая метрика, инструментация
Ноутбук	Новый подраздел: описание pipeline_prepare_model.ipynb
2.2. Код / архитектура
Файл	Изменение	Объём
ats/reqcore/server/api/applications/[id]/analyze.post.ts	Добавить ветвление: если provider === 'biencoder', вызвать микросервис	~30 строк
ats/reqcore/server/utils/ai/provider.ts	Добавить провайдер biencoder в PROVIDER_REGISTRY	~15 строк
ats/reqcore/server/database/schema/app.ts	Добавить 'biencoder' в enum provider таблицы aiConfig	1 строка
experiments/notebooks/pipeline_prepare_model.ipynb	Создать: консолидация скриптов 05–09 в один ноутбук	Новый файл
experiments/scripts/11_api_server.py	Добавить endpoint POST /score (одиночный скоринг для Reqcore)	~20 строк
2.3. A/B-тест
Минимальная инструментация: поле analysisRun.provider + analysisRun.model уже существует в Reqcore. Достаточно записывать provider: 'biencoder', model: 'rubert-tiny2-v1' при использовании альтернативного пути.

3. Исправленный текст для вставки в работу
Ниже — готовые фрагменты для замены в соответствующих местах диссертации. Каждый фрагмент самодостаточен.

3.1. Замена для раздела «Спецификация API» (вместо стр. 85–102)
Спецификация программного интерфейса модуля ранжирования

Модуль ранжирования реализован как независимый HTTP-микросервис на базе FastAPI. Спецификация включает два маршрута:

Таблица 1. Маршруты API модуля ранжирования

Маршрут	Метод	Вход	Выход
/index/candidate	POST	CandidatePayload (id, model_input, metadata)	{"status": "indexed", "total_pool_size": N}
/search/rank	POST	SearchPayload (vacancy_text, filters, top_k)	Упорядоченный JSON-список: candidate_id, match_score, summary
Контракт данных определён через Pydantic-схемы:

class CandidatePayload(BaseModel):
    id: str
    model_input: str
    metadata: Optional[Dict[str, Any]] = None

class SearchPayload(BaseModel):
    vacancy_text: str
    filters: Optional[Dict[str, str]] = None
    top_k: int = 10

Каждый ответ сервера содержит телеметрические HTTP-заголовки X-Processing-Time-Ms и X-Model-Version, обеспечивающие трассируемость результатов при A/B-тестировании моделей.

Маршруты офлайн-индексации в Vector DB (upsert/delete), мониторинга справедливости и раздельного построения эмбеддингов представляют направление развития и в текущей версии прототипа не реализованы.

3.2. Замена для раздела «Реализация модели» (вместо стр. 106–120)
Реализация и дообучение модели сопоставления «вакансия–резюме»

Таблица 2. Конфигурация модели

Параметр	Значение
Базовая модель	cointegrated/rubert-tiny2
Архитектура	BERT, 3 слоя, 12 голов внимания
Скрытая размерность	312
Размер словаря	83 828 токенов (русский BPE)
Макс. длина последовательности	2048 токенов
Пулинг	CLS-токен + L2-нормализация
Размерность эмбеддинга	312
Размер модели на диске	112 МБ (safetensors)
Таблица 3. Гиперпараметры обучения

Параметр	Значение
Функция потерь	ContrastiveLoss (margin = 0,5, cosine distance)
Batch size	8
Эпохи	3
Warmup steps	100
Learning rate	5e-5 (AdamW)
Валидация	BinaryClassificationEvaluator каждые 100 шагов
Сохранение	Лучшая модель по F1 на val-выборке
Подгрузка обучающих данных реализована потоковым классом StreamingPairDataset, наследующим torch.utils.data.IterableDataset. Тексты вакансий и резюме хранятся на диске в SQLite-индексе; при генерации каждого батча выполняется быстрый lookup по ID:

class StreamingPairDataset(IterableDataset):
    def __init__(self, split_tsv_path: str, db_path: str):
        self.split_path = split_tsv_path
        self.db_path = db_path

    def __iter__(self):
        conn = sqlite3.connect(self.db_path)
        with open(self.split_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                resume = conn.execute(
                    "SELECT model_input FROM features WHERE id=?",
                    (row['resume_id'],)
                ).fetchone()
                vacancy = conn.execute(
                    "SELECT model_input FROM features WHERE id=?",
                    (row['vacancy_id'],)
                ).fetchone()
                if resume and vacancy:
                    yield InputExample(
                        texts=[resume[0], vacancy[0]],
                        label=float(row['label'])
                    )

Данное решение обеспечивает потребление RAM порядка 80 МБ при обучении на массивах в сотни тысяч пар.

Примечание: в первой версии прототипа использовалась исключительно модель rubert-tiny2. Упоминавшаяся ранее модель ru-en-RoSBERTa рассматривалась как кандидат для будущих экспериментов, но не была интегрирована в текущий конвейер. Данное уточнение требует подтверждения.

3.3. Замена для раздела «Модуль ранжирования» (вместо стр. 122–129)
Программная реализация модуля ранжирования кандидатов

Ядро ранжирования реализовано в классе ATSRanker с двухступенчатой архитектурой:

class ATSRanker:
    def __init__(self, model_path_or_name: str):
        self.model = SentenceTransformer(model_path_or_name)

    def search(self, vacancy_text, candidate_pool, filters=None, top_k=10):
        # Этап 1: жёсткая фильтрация
        candidates = self._apply_hard_filters(candidate_pool, filters)
        # Этап 2: семантическое переранжирование
        vacancy_emb = self.model.encode(vacancy_text, convert_to_tensor=True)
        cand_texts = [c['model_input'] for c in candidates]
        cand_embs = self.model.encode(cand_texts, convert_to_tensor=True)
        scores = util.cos_sim(vacancy_emb, cand_embs)[0]
        top_indices = torch.topk(scores, k=min(top_k, len(candidates)))
        # ... формирование JSON-ответа

Первый этап (_apply_hard_filters) отсекает кандидатов, не удовлетворяющих мандаторным ограничениям (город, формат занятости), сокращая пул до управляемого размера. Второй этап вычисляет косинусное сходство между эмбеддингом вакансии и эмбеддингами оставшихся кандидатов, возвращая top-K отсортированный по убыванию.

Таблица 4. Производительность модуля (CPU, single-core, без GPU)

Размер пула	Задержка	Стоимость на документ	QPS
100	417 мс	4,1 мс	~4
1 000	2 220 мс	2,2 мс	~0,5
5 000	[требует замера]	[требует замера]	[требует замера]
Пиковое потребление RAM: 28,31 МБ (3% от ограничения в 1 ГБ).

Примечание: текущая реализация использует brute-force cosine similarity. Модули LTR-переранжирования (RankNet), ANN-поиска (DiskANN/SPANN) и Vector DB (Milvus) в прототипе не реализованы и представляют направление развития системы.

3.4. Замена для раздела «Экономическая эффективность» (вместо стр. 148–155)
Сравнительная оценка затрат: LLM-скоринг vs. семантический модуль

Таблица 5. Сравнение операционных характеристик двух подходов к ранжированию

Параметр	LLM-скоринг (текущий путь Reqcore)	Bi-Encoder (предлагаемый модуль)
Задержка на 1 оценку	2–8 сек (зависит от провайдера и модели)	~4 мс на документ (CPU)
Стоимость за 1 оценку	$0,002–0,03 (по тарифам OpenAI/Anthropic)	$0 (self-hosted, CPU)
Стоимость на 100 откликов/вакансия	$0,20–3,00	$0
Требования к инфраструктуре	Интернет + API-ключ провайдера	VPS ~$5/мес (1 vCPU, 1 GB RAM)
Пиковый RAM	N/A (облачный API)	28 МБ
Объяснимость	Покритериальная (evidence, strengths, gaps)	Только match_score (0–1)
Зависимость от внешнего API	Да	Нет
Примечание: стоимость LLM-скоринга рассчитана по публичным тарифам на апрель 2026 г. для модели gpt-4.1-mini ($0,002/запрос) и claude-sonnet-4 ($0,015/запрос). Точные цифры зависят от длины промпта и модели. Требует уточнения по фактическому token usage из таблицы analysisRun.

Семантический модуль не заменяет, а дополняет LLM-скоринг: он обеспечивает мгновенное предварительное ранжирование (pre-ranking), после которого покритериальная LLM-оценка применяется только к top-K кандидатам. При K=10 и 100 откликах на вакансию это сокращает количество LLM-вызовов на 90%.

3.5. Замена для раздела «Метрики» (уточнение стр. 133–139)
Таблица 6. Результаты офлайн-оценки качества ранжирования

Метрика	Значение	Тестовый набор
NDCG@3	0,92	5 кандидатов, K=3, синтетический
Precision@3	0,67	То же
Recall@3	1,00	То же
MRR	1,00	То же
Таблица 7. Сравнение подходов на краевом случае (синонимический перефраз)

Метод	Оценка релевантного кандидата	Оценка нерелевантного
TF-IDF	0,000	0,252
rubert-tiny2 (base, zero-shot)	0,815	0,724
rubert-tiny2 (fine-tuned)	0,865	[требует замера]
Ограничение: метрики получены на синтетическом тестовом наборе из 5 кандидатов. Для валидации на production-масштабе необходимы замеры на полном test-сплите (10% корпуса) и сравнение с LLM-скорингом Reqcore на тех же парах.

4. Изменения в коде / архитектуре
4.1. Точка врезки: один репозиторий
Обоснование: микросервис ML-модуля (11_api_server.py) уже живёт в experiments/scripts/ того же репозитория. Reqcore находится в ats/reqcore/. Разделение на два репозитория создаёт накладные расходы на синхронизацию без выгоды — оба компонента деплоятся на один сервер.

4.2. Минимальные изменения (3 файла в Reqcore + 1 в ML)
Файл 1: ats/reqcore/server/utils/ai/provider.ts

Добавить провайдер biencoder:

// В PROVIDER_REGISTRY добавить:
biencoder: {
  name: 'Bi-Encoder (Self-Hosted)',
  models: [
    { id: 'rubert-tiny2-v1', name: 'rubert-tiny2 fine-tuned' },
  ],
  defaultModel: 'rubert-tiny2-v1',
  setupUrl: null, // self-hosted, ключ не нужен
}

Файл 2: ats/reqcore/server/api/applications/[id]/analyze.post.ts

Добавить ветвление перед вызовом scoreApplication():

let compositeScore: number;
let scoringResult: ScoringResponse;

if (aiConfig.provider === 'biencoder') {
  // Альтернативный путь: вызов микросервиса
  const biEncoderUrl = process.env.BIENCODER_URL || 'http://localhost:8000';
  const response = await fetch(`${biEncoderUrl}/search/rank`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      vacancy_text: `Role: ${job.title} | Text: ${job.description}`,
      filters: {},
      top_k: 1,
      candidate_texts: [resumeText]  // одиночный скоринг
    })
  });
  const result = await response.json();
  compositeScore = Math.round(result.results[0].match_score * 100);
} else {
  // Текущий путь: LLM-скоринг
  const result = await scoreApplication(providerConfig, { ... });
  compositeScore = computeCompositeScore(criteria, result.scoring.evaluations);
}

Файл 3: experiments/scripts/11_api_server.py

Добавить endpoint для одиночного скоринга (Reqcore передаёт один текст, а не пул):

class SingleScorePayload(BaseModel):
    vacancy_text: str
    candidate_texts: list[str]
    top_k: int = 1

@app.post("/score/single")
async def score_single(payload: SingleScorePayload):
    vacancy_emb = RANKER_ENGINE.model.encode(
        payload.vacancy_text, convert_to_tensor=True
    )
    cand_embs = RANKER_ENGINE.model.encode(
        payload.candidate_texts, convert_to_tensor=True
    )
    scores = util.cos_sim(vacancy_emb, cand_embs)[0]
    return {
        "scores": [round(float(s), 4) for s in scores],
        "model_version": MODEL_VERSION
    }

Файл 4 (env): .env Reqcore — добавить BIENCODER_URL=http://localhost:8000

4.3. Схема развёртывания
┌─────────────────────────────────────────────┐
│                  VPS / Сервер               │
│                                              │
│  ┌──────────────┐    ┌────────────────────┐  │
│  │ Reqcore       │    │ ML-микросервис     │  │
│  │ (Nuxt/Nitro)  │───→│ (FastAPI :8000)    │  │
│  │ :3000         │    │ rubert-tiny2       │  │
│  └──────────────┘    │ RAM: ~30 МБ        │  │
│         │            └────────────────────┘  │
│         ▼                                    │
│  ┌──────────────┐                            │
│  │ PostgreSQL   │                            │
│  │ (application │                            │
│  │  .score)     │                            │
│  └──────────────┘                            │
└──────────────────────────────────────────────┘

5. Воспроизводимый ноутбук и команды запуска
5.1. Ноутбук: experiments/notebooks/pipeline_prepare_model.ipynb
Вход: Сырые TSV-файлы в data/sample/

Шаги (ячейки):

Ячейка	Что делает	Артефакт
1	Профилирование (из 02_profile)	profiling_reports/*.json
2	Очистка + PII-маскирование (из 03_clean)	data/clean/*.tsv
3	Сплитование по HR-меткам (из 04_build)	data/splits/train.tsv, val.tsv, test.tsv
4	Нормализация текста (из 05_text)	data/preprocessed/*.tsv
5	Извлечение сущностей (из 06_entity)	data/features/*.tsv
6	Форматирование для модели (из 07_feature)	data/unified/*.tsv
7	Построение SQLite-индекса (из 08_model)	data/splits/model_features.db
8	Обучение bi-encoder (из 09_train)	experiments/models/bi_encoder_rubert_tiny2/
9	Валидация (из 14_ranking_metrics)	NDCG@K, Precision@K, Recall@K, MRR
Выход: Директория experiments/models/bi_encoder_rubert_tiny2/ с файлами: model.safetensors, config.json, tokenizer.json, modules.json, 1_Pooling/config.json.

5.2. Команды запуска
# 1. Подготовка модели (ноутбук)
cd experiments/notebooks
jupyter nbconvert --to notebook --execute pipeline_prepare_model.ipynb

# 2. Запуск микросервиса
cd experiments/scripts
MODEL_PATH=../models/bi_encoder_rubert_tiny2 \
  uvicorn 11_api_server:app --host 0.0.0.0 --port 8000

# 3. Проверка работоспособности
curl -X POST http://localhost:8000/search/rank \
  -H 'Content-Type: application/json' \
  -d '{"vacancy_text": "Role: Backend | Skills: python, docker", "top_k": 5}'

5.3. Hot swap модели
# 1. Обучить новую модель (изменив параметры в ноутбуке)
#    Результат: experiments/models/bi_encoder_v2/

# 2. Перезапустить микросервис с новым путём
MODEL_PATH=../models/bi_encoder_v2 \
  uvicorn 11_api_server:app --host 0.0.0.0 --port 8000

# Reqcore узнаёт версию через заголовок X-Model-Version

Требует доработки: в текущем коде 11_api_server.py путь к модели захардкожен. Необходимо вынести в переменную окружения MODEL_PATH.

6. A/B-тест: как внедрить и как мерить
6.1. Дизайн
Группы:

Control (A): Текущий путь — LLM-скоринг через scoreApplication() (provider: openai/anthropic)
Treatment (B): Альтернативный путь — Bi-Encoder через микросервис (provider: biencoder)
Распределение: По jobId (все кандидаты одной вакансии оцениваются одним методом). Это исключает смешение внутри одной вакансии.

Инструментация: Уже существует — таблица analysisRun записывает provider и model для каждого скоринга.

6.2. Метрики-кандидаты
Уровень	Метрика	Как измерить	Неоднозначности
Целевая	Конверсия в найм (hired rate)	COUNT(status='hired') / COUNT(*) по группам A/B	Длинный цикл: от отклика до найма может пройти 30–90 дней. Потребуется длительный эксперимент.
Прокси	Конверсия в интервью (interview rate)	COUNT(status='interview') / COUNT(*)	Более быстрая метрика, доступна через 1–2 недели.
Прокси	Время до первого действия рекрутера	MIN(activity.createdAt) - application.createdAt для первого status != 'new'	Требует наличия activity log.
Операционная	Стоимость скоринга на вакансию	Из analysisRun: SUM(promptTokens * price) для группы A; $0 для группы B	Цены меняются; фиксировать на момент эксперимента.
Качество ранжирования	Корреляция рангов A и B	Ранговая корреляция Спирмена между score_A и score_B для одних и тех же кандидатов	Нужно прогнать оба метода на одном наборе (shadow mode).
6.3. Минимальная реализация
В Reqcore UI (страница AI Config): выбор provider = biencoder для конкретного job.
Все скоринги этой вакансии идут через микросервис.
Для контрольной группы — другие вакансии с provider = openai.
Через 2–4 недели: SQL-запрос по analysisRun + application.status, группировка по provider.
Полноценное случайное распределение (рандомизация по вакансиям) требует feature-flag системы, которой в Reqcore нет. На первом этапе достаточно ручного выбора провайдера per job.

7. Что ещё требует подтверждения
№	Вопрос	Почему важно
1	Использовалась ли модель ru-en-RoSBERTa хоть в каком-то эксперименте?	Стр. 119 утверждает «внедрена», код показывает только rubert-tiny2.
2	Проводилась ли реально валидация с HR-отделом?	Стр. 140. Если нет — удалить.
3	Существует ли аудит предвзятости в виде кода/отчёта?	Стр. 120. Если нет — удалить.
4	Какой реальный batch_size: 8 или 16?	Код: 8. Текст (стр. 155): 16.
5	Есть ли замеры на полном test-сплите (а не на 5 кандидатах)?	Все метрики (NDCG, P@K, R@K) получены на синтетическом наборе из 5 шт.
6	Какой точный token usage одного LLM-скоринга в Reqcore?	Нужно для таблицы сравнения затрат.
7	Требуется ли поддержка покритериального скоринга (evidence, strengths, gaps) в Bi-Encoder пути?	Если да — архитектура усложняется. Если нет — текущий подход достаточен.
8	Какое название раздела 1 — «Аналитическая часть» или «Анализ рекрутмента...»?	В тексте два варианта: стр. 14 и стр. 14 (мной ранее изменено). Нужно синхронизировать оглавление.
8. Самопроверка качества результата
Что сделано корректно:
Все цитаты кода — из реальных файлов (10_ranking_module.py, 08_model_dataset.py, 11_api_server.py)
Таблицы конфигурации — из config.json, 09_train_biencoder.py
Метрики — из 14_ranking_metrics.py, 15_performance_benchmark.py
Точка интеграции — из analyze.post.ts (реальный файл Reqcore)
DB-схема — из server/database/schema/app.ts
Остающиеся допущения:
Предложенный endpoint POST /score/single — не существует в текущем коде, это проект изменения
Код ветвления в analyze.post.ts — проект, не реализован
Стоимость LLM-скоринга «$0,002–0,03» — по публичным тарифам, не по фактическому usage из Reqcore
Формулировка A/B-теста предполагает, что рекрутеры будут работать параллельно с двумя методами — не проверено, удобно ли это операционно
Что не содержит вымышленных фактов:
Ни один эксперимент не описан, если его нет в коде
Ни одна метрика не приведена, если она не из скриптов 12–17
Все удалённые утверждения (Milvus, DiskANN, LTR, fair ranking) явно помечены как «направление развития»
Экономическая таблица помечена как требующая уточнения фактического token usage


№	Вопрос	Почему важно
1	Использовалась ли модель ru-en-RoSBERTa хоть в каком-то эксперименте?	Стр. 119 утверждает «внедрена», код показывает только rubert-tiny2. -- net ne ispolzovalas -- isprav text na rubert-tiny2
2	Проводилась ли реально валидация с HR-отделом?	Стр. 140. Если нет — удалить. -- da provodilas no toliko razmetka dataseta -- ne delalos potom nichego s hr
3	Существует ли аудит предвзятости в виде кода/отчёта?	Стр. 120. Если нет — удалить. -- napishi tam chto eto plan na razvitie ili kak to tak
4	Какой реальный batch_size: 8 или 16?	Код: 8. Текст (стр. 155): 16. -- sdelay chtobi sootvetstvovalo kodu
5	Есть ли замеры на полном test-сплите (а не на 5 кандидатах)?	Все метрики (NDCG, P@K, R@K) получены на синтетическом наборе из 5 шт. -- dumau chto net -- sdelay pj eti zameri i dobav v text raboti
6	Какой точный token usage одного LLM-скоринга в Reqcore?	Нужно для таблицы сравнения затрат. -- poschitay po otkritim istochnikam vozmi konstanti a potom sdelay rascheti
7	Требуется ли поддержка покритериального скоринга (evidence, strengths, gaps) в Bi-Encoder пути?	Если да — архитектура усложняется. Если нет — текущий подход достаточен. -- net ne trebuetsa -- mojesh v texte tam pometit chto eto plan razvitia
8	Какое название раздела 1 — «Аналитическая часть» или «Анализ рекрутмента...»?	В тексте два варианта: стр. 14 и стр. 14 (мной ранее изменено). Нужно синхронизировать оглавление. -- nazvanie pervogo raxdela yavlaetsia "Анализ рекрутмента, методов NLP и архитектуры семантического ранжирования"
