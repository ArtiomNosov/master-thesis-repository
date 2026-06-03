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

const BADGE = {
  healthy: "Норма",
  warning: "Предупр.",
  critical: "Критично",
  info: "Инфо",
  neutral: "Архив"
};

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
  const label = BADGE[kind] || value;
  rect(parent, x, y, w, 28, colors[0], 14);
  text(parent, value || label, x, y + 6, w, { size: 11, style: "Bold", color: colors[1], align: "CENTER" });
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
  badge(parent, BADGE[kind] || "Инфо", x + w - 104, y + 14, kind, 84);
  text(parent, value, x + 18, y + 52, w - 36, { size: 30, style: "Bold" });
  text(parent, note, x + 18, y + 94, w - 36, { size: 12, color: C.muted });
}

function sidebar(parent, active) {
  rect(parent, 0, 0, 264, 1024, C.dark, 0);
  rect(parent, 28, 28, 40, 40, C.blue, 12);
  text(parent, "МО", 28, 38, 40, { size: 13, style: "Bold", color: "#FFFFFF", align: "CENTER" });
  text(parent, "ПульсМоделей", 80, 29, 150, { size: 18, style: "Bold", color: "#FFFFFF" });
  text(parent, "Управление моделями в промышленной среде", 80, 54, 160, { size: 11, color: "#8EA2BB" });
  text(parent, "МОНИТОРИНГ", 32, 112, 160, { size: 10, style: "Bold", color: "#7890AD" });
  const items = ["Панель мониторинга", "Модели", "Дрейф данных", "Производительность", "Оповещения", "Реестр", "Настройки"];
  items.forEach((item, index) => {
    const y = 138 + index * 48 + (index === 6 ? 28 : 0);
    const isActive = item === active;
    if (isActive) rect(parent, 20, y, 224, 42, C.blue, 12);
    rect(parent, 34, y + 9, 24, 24, isActive ? "#3B82F6" : "#22334A", 8);
    text(parent, item.slice(0, 2).toUpperCase(), 34, y + 15, 24, { size: 8, style: "Bold", color: "#FFFFFF", align: "CENTER" });
    text(parent, item, 70, y + 12, 150, { size: 13, style: "Bold", color: isActive ? "#FFFFFF" : "#C4D2E5" });
  });
  rect(parent, 20, 850, 224, 118, "#18293F", 18, "#294059");
  text(parent, "Состояние промышленной среды", 38, 870, 180, { size: 13, style: "Bold", color: "#FFFFFF" });
  text(parent, "Критичные проблемы моделей показываются с контекстом дрейфа и действий.", 38, 895, 174, {
    size: 11,
    color: "#9FB1CA",
    h: 48,
    lineHeight: 16
  });
}

function topbar(parent, placeholder, right = "Последние 7 дней") {
  rect(parent, 264, 0, 1176, 80, "#FFFFFF", 0, C.line);
  rect(parent, 296, 18, 420, 44, C.soft, 14, C.line);
  text(parent, placeholder, 314, 31, 360, { size: 13, color: C.muted });
  rect(parent, 1110, 20, 110, 40, C.surface, 12, C.line);
  text(parent, right, 1124, 32, 82, { size: 12, style: "Bold" });
  rect(parent, 1232, 20, 104, 40, C.surface, 12, C.line);
  text(parent, "Промышленная среда", 1248, 32, 72, { size: 12, style: "Bold" });
  rect(parent, 1348, 20, 40, 40, C.blueSoft, 20);
  text(parent, "АК", 1348, 31, 40, { size: 12, style: "Bold", color: C.blue, align: "CENTER" });
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
  text(parent, kind === "bar" ? "Распределения признаков" : "Тренд качества", x + 20, y + 18, w - 40, { size: 17, style: "Bold" });
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
  button(parent, "Назначить", x + w - 204, y + 26, 72);
  button(parent, "Исследовать", x + w - 138, y + 26, 96, true);
  button(parent, "Закрыть", x + w - 40 - 72, y + 26, 72);
}

function dashboard(x) {
  const f = makeFrame("01 — Главная панель мониторинга", x);
  sidebar(f, "Панель мониторинга");
  topbar(f, "Поиск моделей, признаков, оповещений...");
  header(f, "ПАНЕЛЬ МОНИТОРИНГА", "Обзор состояния моделей в промышленной среде", "Отслеживание качества, дрейфа, оповещений и готовности промышленной среды.", ["«Рост»", "Все модели", "7 дней"]);
  metric(f, 296, 232, 258, "Статус системы", "92,4%", "Соответствие целевому уровню, −2,1% к прошлой неделе", "warning");
  metric(f, 572, 232, 258, "Модели в норме", "8", "Стабильное качество и дрейф", "healthy");
  metric(f, 848, 232, 258, "С предупреждением", "3", "Проверить сигналы деградации", "warning");
  metric(f, 1124, 232, 258, "Критичные модели", "1", "Требуется немедленная реакция", "critical");
  chart(f, 296, 388, 724, 286, "line");
  card(f, 1040, 388, 342, 286, "Дрейф и оповещения");
  alertCard(f, 1062, 438, 298, "Высокий дрейф данных", "обнаружение_мошенничества_2 — оценка дрейфа 0,81", "Критично");
  alertCard(f, 1062, 536, 298, "Гармоническая мера ниже базы", "прогноз_оттока_3 — текущий 0,872", "Предупр.");
  table(f, 296, 696, 1086, ["Модель", "Статус", "Гарм. мера", "Дрейф", "Задержка", "Владелец"], [
    ["прогноз_оттока_3", ["Предупр.", "warning"], "0,872", "0,43", "96 мс", "Н. Орлова"],
    ["обнаружение_мошенничества_2", ["Критично", "critical"], "0,781", "0,81", "142 мс", "М. Петров"],
    ["прогноз_спроса_1", ["Норма", "healthy"], "0,914", "0,18", "184 мс", "А. Ким"],
    ["кредитный_риск_4", ["Норма", "healthy"], "0,928", "0,12", "88 мс", "Д. Иванов"]
  ]);
}

function modelDetail(x) {
  const f = makeFrame("02 — Карточка модели", x);
  sidebar(f, "Модели");
  topbar(f, "Поиск в прогноз_оттока_3...", "Последние 30 дней");
  header(f, "КАРТОЧКА МОДЕЛИ", "прогноз_оттока_3", "Классификатор оттока клиентов в промышленной среде.", ["вер. 3.8.2", "Пром", "База"]);
  card(f, 296, 232, 724, 154, "Статус модели");
  text(f, "Деградация качества за последние 48 часов.", 316, 274, 500, { size: 13, color: C.muted });
  badge(f, "Предупр.", 888, 252, "warning", 96);
  ["Версия 3.8.2", "Промышленная", "Развёртывание 21 мая", "удержание_2026_05"].forEach((item, i) => {
    rect(f, 316 + i * 164, 318, 148, 42, C.soft, 12, C.line);
    text(f, item, 328 + i * 164, 332, 124, { size: 11, style: "Bold" });
  });
  metric(f, 296, 408, 258, "Точность", "0,918", "База 0,936", "warning");
  metric(f, 572, 408, 258, "Точность полож. класса", "0,904", "База 0,908", "healthy");
  metric(f, 848, 408, 258, "Полнота", "0,842", "База 0,884", "warning");
  metric(f, 1124, 408, 258, "Площадь под кривой", "0,931", "База 0,951", "warning");
  chart(f, 296, 564, 724, 300, "line");
  card(f, 1040, 232, 342, 154, "Рекомендации");
  button(f, "Переобучить", 1060, 286, 110, true);
  button(f, "Исследовать", 1178, 286, 96);
  button(f, "Откат", 1282, 286, 78);
  table(f, 1040, 564, 342, ["Метрика", "Текущее", "База"], [
    ["Точность", "0,918", "0,936"],
    ["Точность полож. класса", "0,904", "0,908"],
    ["Полнота", "0,842", "0,884"],
    ["Гармоническая мера", "0,872", "0,914"],
    ["Площадь под кривой", "0,931", "0,951"]
  ]);
}

function drift(x) {
  const f = makeFrame("03 — Мониторинг дрейфа данных", x);
  sidebar(f, "Дрейф данных");
  topbar(f, "Поиск дрейфа признаков...", "Последние 24 часа");
  header(f, "МОНИТОРИНГ ДРЕЙФА ДАННЫХ", "Сдвиг распределений: обучение и промышленная среда", "Оценка дрейфа по признакам с визуальной критичностью.", ["обнаружение_мошенничества_2", "ИС + КС"]);
  metric(f, 296, 232, 258, "Общий дрейф", "0,62", "Порог 0,50", "critical");
  metric(f, 572, 232, 258, "Высокий дрейф", "4", "Требуется исследование", "critical");
  metric(f, 848, 232, 258, "С предупреждением", "7", "Отслеживать тренд", "warning");
  metric(f, 1124, 232, 258, "Последняя проверка", "5 мин", "Следующий запуск через 10 мин", "info");
  chart(f, 296, 388, 724, 286, "bar");
  table(f, 296, 704, 1086, ["Признак", "Тип", "Оценка дрейфа", "Статус", "Проверка"], [
    ["сумма_операции", "числовой", "0,81", ["Критично", "critical"], "25 мая, 14:12"],
    ["категория_продавца", "категор.", "0,74", ["Критично", "critical"], "25 мая, 14:12"],
    ["риск_устройства", "числовой", "0,66", ["Критично", "critical"], "25 мая, 14:12"],
    ["расстояние_км", "числовой", "0,49", ["Предупр.", "warning"], "25 мая, 14:12"],
    ["возраст_клиента", "числовой", "0,18", ["Норма", "healthy"], "25 мая, 14:12"]
  ]);
}

function performance(x) {
  const f = makeFrame("04 — Производительность модели", x);
  sidebar(f, "Производительность");
  topbar(f, "Поиск метрик производительности...", "Последние 14 дней");
  header(f, "ПРОИЗВОДИТЕЛЬНОСТЬ МОДЕЛИ", "обнаружение_мошенничества_2 — операционные метрики", "Динамика метрик, сравнение версий, матрица ошибок и бизнес-эффект.", ["Промышленная среда", "вер. 2.4.1 и 2.3.8"]);
  metric(f, 296, 232, 200, "Площадь под кривой", "0,902", "Качество модели", "warning");
  metric(f, 514, 232, 200, "95-й перцентиль", "142 мс", "Ниже жёсткого лимита", "warning");
  metric(f, 732, 232, 200, "Пропускная способность", "1,8 тыс./мин", "Объём предсказаний", "healthy");
  metric(f, 950, 232, 200, "Доля ошибок", "3,4%", "Выше цели", "critical");
  metric(f, 1168, 232, 214, "Бизнес-эффект", "−42 тыс. ₽", "К базе", "critical");
  chart(f, 296, 388, 724, 286, "line");
  table(f, 1040, 388, 342, ["Метрика", "вер. 2.4.1", "вер. 2.3.8"], [
    ["Точность полож. класса", "0,835", "0,861"],
    ["Полнота", "0,881", "0,842"],
    ["Гармоническая мера", "0,857", "0,851"],
    ["Ложноположительные", "384", "311"]
  ]);
  card(f, 296, 704, 342, 206, "Матрица ошибок");
  [["ИН", "18 420", C.blue], ["ЛП", "384", "#60A5FA"], ["ЛН", "261", C.red], ["ИП", "1 942", C.yellow]].forEach((item, i) => {
    const xx = 320 + (i % 2) * 144;
    const yy = 758 + Math.floor(i / 2) * 70;
    rect(f, xx, yy, 118, 54, item[2], 12);
    text(f, item[1], xx, yy + 10, 118, { size: 18, style: "Bold", color: "#FFFFFF", align: "CENTER" });
    text(f, item[0], xx, yy + 34, 118, { size: 10, style: "Bold", color: "#FFFFFF", align: "CENTER" });
  });
  chart(f, 660, 704, 342, 206, "bar");
  card(f, 1024, 704, 358, 206, "Влияние на бизнес-метрики");
  text(f, "Заблокировано: 318 тыс. ₽", 1044, 758, 260, { size: 13, style: "Bold" });
  text(f, "Ложноположительные: 71 тыс. ₽", 1044, 792, 260, { size: 13, style: "Bold", color: C.red });
  text(f, "Пропущено: 42 тыс. ₽", 1044, 826, 260, { size: 13, style: "Bold", color: C.red });
  badge(f, "Чистый −42 тыс. ₽", 1044, 858, "critical", 110);
}

function alerts(x) {
  const f = makeFrame("05 — Оповещения и инциденты", x);
  sidebar(f, "Оповещения");
  topbar(f, "Поиск оповещений...", "Открытые");
  header(f, "ОПОВЕЩЕНИЯ И ИНЦИДЕНТЫ", "Очередь приоритетных реакций", "Критичность, причина, модель, статус и быстрые действия.", ["Новые + расследуются", "Все владельцы"]);
  metric(f, 296, 232, 258, "Новые", "7", "3 требуют назначения", "info");
  metric(f, 572, 232, 258, "Расследуются", "5", "Средний возраст 38 мин", "warning");
  metric(f, 848, 232, 258, "Критичные", "1", "обнаружение_мошенничества_2", "critical");
  metric(f, 1124, 232, 258, "Решено сегодня", "12", "Среднее время восстановления 26 мин", "healthy");
  alertCard(f, 296, 406, 1086, "Высокий дрейф в сумма_операции", "Причина: отклонение распределения. Модель: обнаружение_мошенничества_2. Время: 25 мая, 14:00.", "Критично");
  alertCard(f, 296, 510, 1086, "Полнота ниже порога", "Причина: дрейф модели. Модель: прогноз_оттока_3. Время: 25 мая, 13:48.", "Предупр.");
  alertCard(f, 296, 614, 1086, "Задержка 95-го перцентиля", "Причина: насыщение сервиса. Модель: прогноз_спроса_1. Время: 25 мая, 13:21.", "Предупр.");
  table(f, 296, 744, 1086, ["Инцидент", "Модель", "Критичность", "Статус", "Владелец"], [
    ["Несоответствие схемы", "обнаружение_мошенничества_2", ["Критично", "critical"], "Новый", "Не назначен"],
    ["Нарушение порога качества", "прогноз_оттока_3", ["Предупр.", "warning"], "Расследуется", "Н. Орлова"],
    ["Насыщение сервиса", "прогноз_спроса_1", ["Предупр.", "warning"], "Расследуется", "А. Ким"]
  ]);
}

function registry(x) {
  const f = makeFrame("06 — Реестр моделей", x);
  sidebar(f, "Реестр");
  topbar(f, "Поиск в реестре...", "Все проекты");
  header(f, "РЕЕСТР МОДЕЛЕЙ", "Модели, версии, наборы данных и развёртывание", "Управление жизненным циклом от обучения до отката.", ["Все среды", "Одобренные"]);
  metric(f, 296, 232, 258, "Зарегистрировано", "28", "12 активны в промышленной среде", "info");
  metric(f, 572, 232, 258, "Одобренные версии", "9", "Прошли проверку", "healthy");
  metric(f, 848, 232, 258, "Готовы к откату", "4", "Для критичных процессов", "warning");
  metric(f, 1124, 232, 258, "Заблокированы", "2", "Не прошли проверку дрейфа", "critical");
  table(f, 296, 406, 1086, ["Модель/версия", "Статус", "Обучение", "Развёртывание", "Набор данных", "Владелец"], [
    ["прогноз_оттока_3 / вер. 3.8.2", ["Предупр.", "warning"], "18 мая", "21 мая", "удержание_2026_05", "Н. Орлова"],
    ["обнаружение_мошенничества_2 / вер. 2.4.1", ["Критично", "critical"], "12 мая", "19 мая", "мошенничество_операции_9", "М. Петров"],
    ["прогноз_спроса_1 / вер. 1.9.0", ["Норма", "healthy"], "3 мая", "9 мая", "спрос_недельный_14", "А. Ким"],
    ["кредитный_риск_4 / вер. 4.2.5", ["Одобрено", "healthy"], "22 мая", "Не развёрнуто", "кредит_заявки_2026_2кв", "Д. Иванов"],
    ["эластичность_цены_1 / вер. 1.3.4", ["Архив", "neutral"], "12 мар", "1 апр", "цены_история_7", "С. Ли"]
  ]);
  ["Подробнее", "Сравнить версии", "Развернуть", "Откат"].forEach((label, i) => button(f, label, 296 + i * 144, 764, 130, i === 2));
}

async function main() {
  await Promise.all([
    figma.loadFontAsync({ family: "Inter", style: "Regular" }),
    figma.loadFontAsync({ family: "Inter", style: "Medium" }),
    figma.loadFontAsync({ family: "Inter", style: "Semi Bold" }),
    figma.loadFontAsync({ family: "Inter", style: "Bold" })
  ]);
  figma.currentPage.name = "ПульсМоделей — платформа мониторинга";
  [dashboard, modelDetail, drift, performance, alerts, registry].forEach((builder, index) => builder(index * 1520));
  figma.viewport.scrollAndZoomIntoView(figma.currentPage.children);
  figma.closePlugin("Созданы экраны платформы мониторинга ПульсМоделей.");
}

main();
