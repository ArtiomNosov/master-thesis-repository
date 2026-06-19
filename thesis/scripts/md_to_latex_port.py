#!/usr/bin/env python3
"""Port thesis markdown source to LaTeX chapter files."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MD_PATH = ROOT / "docs/obsidian/thesis/[USED] thesis_draft_final_ru.md"
ABSTRACT_PATH = ROOT / "vkr/artifacts/М24-534_НосовАИ_ВКР_реферат.md"
OUT_DIR = ROOT / "thesis/latex/chapters"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def escape_latex(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicaret{}",
    }
    out = []
    for ch in text:
        out.append(repl.get(ch, ch))
    return "".join(out)


def md_citations_to_latex(text: str) -> str:
    text = re.sub(r"\[(\d+)\s*,\s*стр\.\s*\d+\]", lambda m: f"\\cite{{src{int(m.group(1)):02d}}}", text)

    def repl(m: re.Match[str]) -> str:
        nums = [int(x.strip()) for x in m.group(1).split(",")]
        keys = ", ".join(f"src{n:02d}" for n in nums)
        return f"\\cite{{{keys}}}"

    return re.sub(r"\[(\d+(?:,\s*\d+)*)\]", repl, text)


def should_use_texttt(inner: str) -> bool:
    if inner in {"+", "#"}:
        return True
    if re.search(r"[/\\:_{}\[\]()=+*?|<>\|]", inner):
        return True
    if re.search(
        r"(?i)^(POST|GET|curl|http|provider|optional|dict|candidate|search|score|"
        r"ats|ranker|json|biencoder|model_|analysis_|input_|output_|prompt|completion|"
        r"\.ts|\.py|\.json|\.md|\.tsv|experiments/|ats/)",
        inner,
    ):
        return True
    if re.search(r"^\d+\+", inner):
        return True
    return False


def verb_delim(content: str) -> str:
    for ch in "|!@#$%^&":
        if ch not in content:
            return ch
    return "|"


def json_to_latex_texttt(inner: str) -> str:
    escaped = escape_latex(inner)
    escaped = escaped.replace("[", "{[}").replace("]", "{]}")
    return f"\\texttt{{{escaped}}}"


def protect_json_in_quotes(text: str, slots: list[str]) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith('"{"', i):
            j = i + 2
            depth = 1
            while j < len(text) and depth:
                ch = text[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                j += 1
            if depth == 0 and j < len(text) and text[j] == '"':
                inner = text[i + 1 : j]
                slots.append(json_to_latex_texttt(inner))
                out.append(f"@@P{len(slots) - 1}@@")
                i = j + 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


REGEX_SECTION_REF_SLOT = "@@REGEXREF@@"
REGEX_FIGURE_BLOCK = textwrap.dedent(
    r"""
    \begin{figure}[H]
    \centering
    \includegraphics[width=0.92\textwidth]{figures/regex-section-headers.pdf}
    \caption{Регулярное выражение для сегментации заголовков секций резюме}
    \label{fig:regex-section}
    \end{figure}
    """
).strip()


def normalize_slashes_for_texttt(body: str) -> str:
    """HTTP routes keep slashes; repository file paths use hyphens."""
    if re.search(r"(?i)^(POST|GET)\s+/", body) or body.startswith("/"):
        return body.replace("/", "{\\slash}")
    if re.search(r"\.(py|ts|json|md|tsv)\b", body, re.I):
        return body.replace("/", "-")
    if re.search(r"(?i)^(experiments|ats|data|cointegrated|models)/", body):
        return body.replace("/", "-")
    if body.count("/") >= 3:
        return body.replace("/", "-")
    return body


def protect_regex_patterns(text: str, slots: list[str]) -> str:
    def repl(_: re.Match[str]) -> str:
        return REGEX_SECTION_REF_SLOT + "@@REGEXFIG@@"

    return re.sub(
        r'"\(\?i\)\\b\([^)]+\)\\b"',
        repl,
        text,
    )


def md_quotes_to_texttt(text: str, slots: list[str] | None = None) -> str:
    if slots is None:
        slots = []
    text = protect_regex_patterns(text, slots)
    text = protect_json_in_quotes(text, slots)

    def repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        if should_use_texttt(inner):
            trailing_pct = ""
            body = inner
            if body.endswith("%"):
                trailing_pct = r"\%"
                body = body[:-1]
            escaped = escape_latex(body)
            escaped = normalize_slashes_for_texttt(escaped)
            escaped = escaped.replace("[", "{[}").replace("]", "{]}")
            return f"\\texttt{{{escaped}}}{trailing_pct}"
        return inner

    return re.sub(r'"([^"\n]+)"', repl, text)


def restore_protected_slots(text: str, slots: list[str]) -> str:
    for idx, slot in enumerate(slots):
        text = text.replace(f"@@P{idx}@@", slot)
    return text


def escape_latex_except_cite(text: str) -> str:
    chunks = re.split(r"(@@P\d+@@|\\cite\{[^}]+\})", text)
    return "".join(
        chunk
        if chunk.startswith("\\cite") or re.fullmatch(r"@@P\d+@@", chunk)
        else escape_latex(chunk)
        for chunk in chunks
    )


def md_guillemets(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("„", '"')
    text = text.replace("«", '"').replace("»", '"')
    return text


def normalize_dashes(text: str) -> str:
    text = text.replace("–", "--").replace("—", "---")
    return text


def split_latex_texttt(text: str) -> list[tuple[str, bool]]:
    result: list[tuple[str, bool]] = []
    i = 0
    marker = r"\texttt{"
    while i < len(text):
        idx = text.find(marker, i)
        if idx == -1:
            result.append((text[i:], False))
            break
        if idx > i:
            result.append((text[i:idx], False))
        j = idx + len(marker)
        depth = 1
        while j < len(text) and depth:
            c = text[j]
            if c == "\\":
                j += 1
                while j < len(text) and text[j].isalpha():
                    j += 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        result.append((text[idx:j], True))
        i = j
    return result


def convert_inline(text: str) -> str:
    slots: list[str] = []
    eqref_slots: list[str] = []

    def stash_eqref(match: re.Match[str]) -> str:
        eqref_slots.append(match.group(0))
        return f"@@EQSLOT{len(eqref_slots) - 1}@@"

    text = re.sub(r"@@EQREF:\w+@@", stash_eqref, text)
    text = md_guillemets(text)
    text = md_quotes_to_texttt(text, slots)
    text = md_citations_to_latex(text)
    text = normalize_dashes(text)
    escaped: list[str] = []
    for part, protected in split_latex_texttt(text):
        if protected:
            escaped.append(part)
        else:
            escaped.append(escape_latex_except_cite(part))
    text = restore_protected_slots("".join(escaped), slots).replace(
        REGEX_SECTION_REF_SLOT, r"(см.~\ref{fig:regex-section})"
    )
    for idx, slot in enumerate(eqref_slots):
        text = text.replace(f"@@EQSLOT{idx}@@", slot)
    return text


def format_route_path(route: str) -> str:
    route = route.strip().strip('"')
    return f"\\path{{{route}}}"


def format_table_cell(text: str, *, column: int | None = None, is_api_table: bool = False) -> str:
    raw = text.strip().strip('"')
    if is_api_table and column == 0 and raw.startswith("/"):
        return format_route_path(raw)
    return convert_inline(text)


def is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|")


def parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) >= 2 and all(set(c) <= {"-", ":"} for c in rows[1]):
        header, body = rows[0], rows[2:]
    else:
        header, body = rows[0], rows[1:]
    return header, body


def table_to_longtable(header: list[str], body: list[list[str]], caption: str, label: str) -> str:
    n = len(header)
    if n == 5:
        colspec = (
            r"|P{0.17\textwidth}|P{0.06\textwidth}|P{0.20\textwidth}|"
            r"P{0.27\textwidth}|P{0.20\textwidth}|"
        )
    elif n == 4:
        colspec = r"|P{0.22\textwidth}|P{0.28\textwidth}|P{0.28\textwidth}|P{0.16\textwidth}|"
    elif n == 3:
        colspec = r"|P{0.30\textwidth}|P{0.34\textwidth}|P{0.28\textwidth}|"
    else:
        colspec = "|" + "|".join([r"P{0.22\textwidth}"] * n) + "|"

    is_api = label == "tbl:api"

    def row(cells: list[str], *, inline: bool = False) -> str:
        if inline:
            content = (
                format_table_cell(c, column=i, is_api_table=is_api)
                for i, c in enumerate(cells)
            )
        else:
            content = (escape_latex(c) for c in cells)
        return " & ".join(content) + r" \\"

    lines = [
        r"\begin{small}",
        r"\begin{longtable}{" + colspec + "}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\hline",
        row(header) + "\n\\hline",
        r"\endfirsthead",
        r"\hline",
        row(header) + "\n\\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot",
        r"\hline",
        r"\endlastfoot",
    ]
    for idx, r in enumerate(body):
        suffix = row(r, inline=True)
        if idx < len(body) - 1:
            suffix += "\n\\hline"
        lines.append(suffix)
    lines.extend([r"\end{longtable}", r"\end{small}"])
    return "\n".join(lines)


TABLE_LABELS = iter(
    [
        ("tbl:api", "Спецификация API-маршрутов FastAPI-сервиса ранжирования"),
        ("tbl:model-params", "Параметры сохранённой модели cointegrated/rubert-tiny2"),
        ("tbl:hyperparams", "Гиперпараметры и фактические показатели обучения"),
        ("tbl:ranking", "Ranking-метрики на полном test split"),
        ("tbl:deploy-time", "Сравнение времени ручной обработки и модельного контура"),
        ("tbl:roi", "Расчётные денежные показатели внедрения"),
        ("tbl:offline", "Уровни офлайн-валидации трёх стратегий ранжирования"),
    ]
)

FORMULA_LINE_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s*(?:~?=|=)"
)
FORMULA_EXCLUDE_RE = re.compile(
    r"^(if|for|while|def|class|import|return|elif|else)\b"
)


class EquationRegistry:
    """Tracks numbered equation labels for \\eqref in variable lists."""

    def __init__(self) -> None:
        self._counter = 0
        self.by_name: dict[str, str] = {}

    def add(self, lhs_name: str) -> str:
        self._counter += 1
        slug = re.sub(r"[^\w]+", "-", lhs_name.lower()).strip("-") or "expr"
        label = f"eq:{slug}-{self._counter}"
        if lhs_name not in self.by_name:
            self.by_name[lhs_name] = label
        return label

    def label_for(self, name: str) -> str | None:
        base = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", name)
        if not base:
            return None
        return self.by_name.get(base.group(1))


def latex_subscript_name(name: str) -> str:
    if "_" not in name:
        return name
    base, rest = name.split("_", 1)
    if rest.isdigit():
        return f"{base}_{{{rest}}}"
    text = rest.replace("_", r"\_")
    return f"{base}_{{\\text{{{text}}}}}"


def latex_formula_token(token: str) -> str:
    match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)(\([^)]*\))$", token)
    if match:
        name, args = match.groups()
        inner_args = re.sub(
            r"[A-Za-z][A-Za-z0-9_]*",
            lambda m: latex_subscript_name(m.group(0)),
            args,
        )
        return latex_subscript_name(name) + inner_args
    return latex_subscript_name(token)


def wrap_cyrillic_in_text(formula: str) -> str:
    parts = re.split(r"([а-яА-ЯёЁ][а-яА-ЯёЁ\s.\-]*)", formula)
    out: list[str] = []
    for part in parts:
        if part and re.match(r"[а-яА-ЯёЁ]", part):
            out.append(f"\\text{{{part.strip()}}}")
        else:
            out.append(part)
    return "".join(out)


def md_formula_to_latex(raw: str) -> str:
    s = raw.strip()
    s = s.replace("~=", r"\approx ")
    s = re.sub(r"\bceil\(", lambda _: r"\operatorname{ceil}(", s)

    while re.search(r"(\d)_(\d)", s):
        s = re.sub(r"(\d)_(\d)", r"\1{,}\2", s)

    if s.rstrip().endswith("%"):
        s = s.rstrip()[:-1].rstrip() + r" \%"

    s = re.sub(r"\s*\*\s*", lambda _: r" \cdot ", s)
    s = re.sub(
        r"[A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?",
        lambda m: latex_formula_token(m.group(0)),
        s,
    )
    return wrap_cyrillic_in_text(s)


def format_c_ui_equation() -> str:
    return (
        r"C_{\text{ui}} = \frac{promptTokens}{1{,}000{,}000} \cdot inputPrice \\"
        "\n         + \\frac{completionTokens}{1{,}000{,}000} \\cdot outputPrice"
    )


def formula_lhs_name(raw: str) -> str:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", raw.strip())
    return match.group(1) if match else "expr"


def is_formula_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.lower().startswith("где"):
        return False
    if FORMULA_EXCLUDE_RE.match(stripped):
        return False
    if not FORMULA_LINE_RE.match(stripped):
        return False
    if any(x in stripped for x in ("self.", ".get(", "===", "==", "!==")):
        return False
    if re.search(r"\b(util|torch|self|np|math)\.", stripped):
        return False
    if re.search(r"\.(encode|topk|cos_sim)\(", stripped):
        return False
    lhs = stripped.split("=", 1)[0].strip()
    if re.search(r"[а-яА-ЯёЁ]", lhs):
        return False
    return True


PYTHON_CODE_PREFIXES = (
    "filtered_candidates",
    "candidate_texts",
    "vacancy_emb",
    "candidate_embs",
    "cos_scores",
    "top_results",
)


def looks_like_python_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(stripped.startswith(f"{prefix} ") or stripped.startswith(f"{prefix}=") for prefix in PYTHON_CODE_PREFIXES)


def emit_python_listing(code_lines: list[str]) -> str:
    body = "\n".join(line.rstrip() for line in code_lines)
    return (
        "{\\linespread{1}\\selectfont\n"
        "\\begin{lstlisting}[language=Python, numbers=left, stepnumber=1, "
        "caption={Семантический этап ранжирования в ATSRanker}, "
        "label={lst:search}]\n"
        f"{body}\n"
        "\\end{lstlisting}\n"
        "}"
    )


def format_binding_equation(raw: str, registry: EquationRegistry) -> str:
    lhs, rhs = [part.strip() for part in raw.split("=", 1)]
    label = registry.add(lhs)
    lhs_tex = rf"\text{{{lhs}}}"
    rhs_tex = rf"\text{{{escape_latex(rhs)}}}"
    return f"\\begin{{equation}}\n\\label{{{label}}}\n{lhs_tex} = {rhs_tex}\n\\end{{equation}}"


def format_equation(raw: str, registry: EquationRegistry) -> str:
    if re.fullmatch(
        r"\w+ = (analysis_run\.\w+|input_price_per_1m|output_price_per_1m)", raw.strip()
    ):
        return format_binding_equation(raw, registry)
    lhs = formula_lhs_name(raw)
    label = registry.add(lhs)
    if lhs == "C_ui":
        body = format_c_ui_equation()
    else:
        body = md_formula_to_latex(raw)
    return f"\\begin{{equation}}\n\\label{{{label}}}\n{body}\n\\end{{equation}}"


def inject_eqrefs(text: str, registry: EquationRegistry) -> str:
    def repl(match: re.Match[str]) -> str:
        label = registry.label_for(match.group(1))
        return f"\\eqref{{{label}}}" if label else match.group(0)

    return re.sub(r"@@EQREF:(\w+)@@", repl, text)


def convert_gde_line(line: str, registry: EquationRegistry) -> str:
    text = line.strip()
    segments: list[tuple[str, str]] = []
    pos = 0
    for match in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))*)"', text):
        if match.start() > pos:
            segments.append(("text", text[pos : match.start()]))
        var = match.group(1)
        base = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", var)
        math_name = base.group(1) if base else var
        math = f"${latex_subscript_name(math_name)}$"
        eq_label = registry.label_for(var)
        if eq_label:
            segments.append(("math", f"{math}~\\eqref{{{eq_label}}}"))
        else:
            segments.append(("math", math))
        pos = match.end()
    if pos < len(text):
        segments.append(("text", text[pos:]))

    out: list[str] = []
    for kind, content in segments:
        if kind == "text":
            out.append(convert_inline(content))
        else:
            out.append(content)
    return "".join(out)


def convert_block(lines: list[str]) -> str:
    out: list[str] = []
    i = 0
    in_code = False
    code_lines: list[str] = []
    registry = EquationRegistry()
    while i < len(lines):
        line = lines[i].rstrip()
        if line.strip().startswith("@startuml"):
            while i < len(lines) and not lines[i].strip().startswith("@enduml"):
                i += 1
            i += 1
            out.append(
                textwrap.dedent(
                    r"""
                    \FloatBarrier
                    \begin{figure}[H]
                    \centering
                    \includegraphics[width=\textwidth,height=0.648\textheight,keepaspectratio]{figures/application_analysis_request_flow.pdf}
                    \caption{Поток обработки запроса анализа заявки в интеграции Reqcore с модулем ранжирования}
                    \label{fig:flow}
                    \end{figure}
                    """
                ).strip()
            )
            continue
        if line.strip().startswith("filtered_candidates = "):
            in_code = True
            code_lines = []
        if in_code:
            stripped = line.strip()
            if not stripped:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and looks_like_python_code_line(lines[j].rstrip()):
                    i += 1
                    continue
                if code_lines:
                    out.append(emit_python_listing(code_lines))
                code_lines = []
                in_code = False
                continue
            if looks_like_python_code_line(line):
                code_lines.append(line.rstrip())
                i += 1
                continue
            if code_lines:
                out.append(emit_python_listing(code_lines))
            code_lines = []
            in_code = False
            continue
        if is_table_line(line):
            tbl_lines = []
            while i < len(lines) and is_table_line(lines[i]):
                tbl_lines.append(lines[i])
                i += 1
            header, body = parse_table(tbl_lines)
            try:
                label, cap = next(TABLE_LABELS)
            except StopIteration:
                label, cap = "tbl:extra", "Таблица"
            out.append(table_to_longtable(header, body, cap, label))
            continue
        stripped = line.strip()
        if stripped.lower().startswith("где "):
            out.append(convert_gde_line(stripped, registry))
            i += 1
            continue
        if is_formula_line(stripped):
            out.append(format_equation(stripped, registry))
            i += 1
            continue
        if re.match(r"^\d+\.\s+", line.strip()):
            items = []
            needs_regex_fig = False
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item = convert_inline(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                if "@@REGEXFIG@@" in item:
                    needs_regex_fig = True
                    item = item.replace("@@REGEXFIG@@", "")
                items.append(item)
                i += 1
            out.append("\\begin{compactenum}")
            for it in items:
                out.append(f"  \\item {it}")
            out.append("\\end{compactenum}")
            if needs_regex_fig:
                out.append(REGEX_FIGURE_BLOCK)
            continue
        if line.strip():
            out.append(convert_inline(line.strip()))
        i += 1
    return inject_eqrefs("\n\n".join(out), registry)


def split_sections(md: str) -> dict[str, str]:
    md = md.split("**Список литературы**")[0]
    parts = re.split(r"\n\*\*(.+?)\*\*\n", md)
    sections: dict[str, str] = {}
    title = parts[0].strip().splitlines()[0].lstrip("# ").strip()
    sections["__title__"] = title
    it = iter(parts[1:])
    for head, body in zip(it, it):
        sections[head.strip()] = body.strip()
    return sections


CHAPTER1_SECTIONS = [
    "Системный анализ предметной области подбора персонала, процессов подбора персонала и систем автоматизированного подбора персонала",
    "Аналитический обзор методов обработки естественного языка (Natural Language Processing) и методов машинного обучения для построения ранжирующей функции (Learning to Rank)",
    "Обоснование выбора архитектуры двунаправленных кодировочных представлений (Bidirectional Encoder Representations from Transformers), модели векторных представлений предложений (Sentence Embeddings using Siamese Bidirectional Encoder Representations from Transformers Networks) и методов векторизации",
]

CHAPTER2_SECTIONS = [
    "Выполнение сбора, профилирования, очистки и разметки исходных данных вакансии и резюме",
    "Проектирование архитектуры конвейера предварительной обработки данных (preprocessing pipeline) и модуля вычисления семантической близости плотных векторных представлений текста",
    "Спецификация программного интерфейса приложения (Application Programming Interface) и программных модулей интеграции модуля ранжирования соискателей с ядром системы автоматизированного подбора персонала",
    "Интеграция системы автоматизированного подбора персонала под названием \"Reqcore\" с модулем семантического ранжирования цифровых профилей соискателей",
]

CHAPTER3_SECTIONS = [
    "Разработка программного кода подготовки данных и конвейера машинного обучения",
    (
        "Реализация и доменная тонкая настройка модели семантического сопоставления текст вакансии и текст резюме",
        "Реализация и доменная тонкая настройка модели семантического сопоставления текст вакансии и текст резюме",
    ),

]

CHAPTER4_SECTIONS = [
    "Сравнение подхода на основе моделей с механизмом самовнимания с классическими системами ранжирования",
    "Проведение экспериментальной оценки качества алгоритма ранжирования по нормализованной дисконтированной кумулятивной выгоде, точности, полноте и средней обратной позиции",
    "Выполнение интеграции созданного модуля в целевую систему автоматизированного подбора персонала",
    "Экономическое обоснование внедрения модуля ранжирования",
    "Валидация трех стратегий ранжирования на отложенных данных",
]


def section_entry(entry: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return entry, entry


def section_body(sections: dict[str, str], md_key: str) -> str:
    body = sections.get(md_key, "")
    return convert_block(body.splitlines())


def write_chapter(
    path: Path,
    chapter_cmd: str,
    label: str,
    intro: str,
    section_names: list[str | tuple[str, str]],
    sections: dict[str, str],
) -> None:
    chunks = [
        chapter_cmd,
        f"\\label{{{label}}}",
        "",
        intro,
    ]
    for entry in section_names:
        latex_title, md_key = section_entry(entry)
        chunks.append(f"\\section{{{latex_title}}}")
        chunks.append("")
        chunks.append(section_body(sections, md_key))
        chunks.append("")
    path.write_text("\n".join(chunks).strip() + "\n", encoding="utf-8")


def write_abstract() -> None:
    md = read_text(ABSTRACT_PATH)
    lines = [l.strip() for l in md.splitlines() if l.strip() and not l.startswith("#")]
    keywords = ""
    body: list[str] = []
    for line in lines:
        if line.startswith("Ключевые слова:"):
            keywords = convert_inline(line.replace("Ключевые слова:", "").strip())
        else:
            body.append(convert_inline(line))
    stats = body.pop(0).replace("–", "---").replace("—", "---")
    stats = re.sub(
        r"(\d+)\s+страниц,\s+(\d+)\s+рисунка,\s+(\d+)\s+таблиц",
        r"\\pageref{end_of_main_text}~стр., 2~рис., \3~табл.",
        stats,
    )
    tex = (
        "\\chapter*{Реферат}\n"
        "\\addcontentsline{toc}{chapter}{Реферат}\n\n"
        f"{stats}\n\n"
        f"{body[0]}\n\n"
        f"\\textbf{{Ключевые слова:}} {keywords}\n\n"
        + "\n\n".join(body[1:])
        + "\n"
    )
    (OUT_DIR / "master-thesis-abstract.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    sections = split_sections(read_text(MD_PATH))
    write_abstract()

    intro = section_body(sections, "Введение")
    (OUT_DIR / "master-thesis-intro.tex").write_text(
        "\\chapter*{Введение}\n"
        "\\label{sec:intro}\n"
        "\\addcontentsline{toc}{chapter}{Введение}\n\n"
        + intro
        + "\n",
        encoding="utf-8",
    )

    write_chapter(
        OUT_DIR / "master-thesis-chapter1.tex",
        "\\chapter{Анализ сферы подбора персонала, методов обработки естественного языка и подходов семантического сопоставления вакансий и резюме}",
        "chapter:analysis",
        "В главе анализируются предметная область подбора персонала и системы автоматизированного подбора персонала, рассматриваются методы обработки естественного языка и подходы Learning to Rank, а также обосновывается выбор нейросетевых архитектур и методов векторизации для задачи семантического сопоставления текст вакансии и текст резюме.",
        CHAPTER1_SECTIONS,
        sections,
    )

    write_chapter(
        OUT_DIR / "master-thesis-chapter2.tex",
        "\\chapter{Проектирование схем данных, конвейера их обработки и интеграции модуля ранжирования цифровых профилей соискателей с системой автоматизированного подбора персонала}",
        "chapter:design",
        "В главе описываются сбор, профилирование, очистка и разметка исходных данных, проектируется архитектура конвейера предварительной обработки и модуля семантического анализа, специфицируется программный интерфейс ранжирующего сервиса и рассматривается интеграция с ядром Reqcore.",
        CHAPTER2_SECTIONS,
        sections,
    )

    write_chapter(
        OUT_DIR / "master-thesis-chapter3.tex",
        "\\chapter{Разработка и реализация модуля, который семантически сопоставляет вакансии и резюме, рассчитывает оценку соответствия в доменном векторном пространстве и ранжирует кандидатов на вакансию}",
        "chapter:impl",
        "В главе описываются программный код подготовки данных и ML-конвейера, реализация и дообучение модели сопоставления текст вакансии и текст резюме, структура сохранённого артефакта, реализация класса \\texttt{ATSRanker} и FastAPI-сервиса.",
        CHAPTER3_SECTIONS,
        sections,
    )

    write_chapter(
        OUT_DIR / "master-thesis-chapter4.tex",
        "\\chapter{Экспериментальная оценка, интеграция и анализ эффективности модуля ранжирования кандидатов}",
        "chapter:eval",
        "Экспериментальная часть подтверждает улучшение разделимости релевантных и нерелевантных пар после дообучения, измерение ranking-метрик на полном test split и работоспособность API-контура интеграции с Reqcore.",
        CHAPTER4_SECTIONS,
        sections,
    )

    conclusion = section_body(sections, "Заключение")
    (OUT_DIR / "master-thesis-conclusion.tex").write_text(
        textwrap.dedent(
            f"""
            \\chapter*{{Заключение}}
            \\addcontentsline{{toc}}{{chapter}}{{Заключение}}

            {conclusion}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    print("LaTeX chapters written to", OUT_DIR)


if __name__ == "__main__":
    main()
