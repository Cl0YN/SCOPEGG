"""Проверка faithfulness: каждое число и каждая зона в тексте совета должны быть в findings JSON."""
from __future__ import annotations

import math
import re
from collections.abc import Iterable

CACHE_CALLOUTS: list[str] = [
    "A main", "A site", "A long", "A short", "Highway", "Quad", "Forklift", "Truck", "Squeaky",
    "Toxic", "Boost", "Heaven", "Mid", "Vents", "Checkers", "Z-hall", "Z connector", "Garage",
    "White box", "Sunroom", "Sun room", "B main", "B halls", "B hall", "B site", "Headshot",
    "CT spawn", "T spawn", "Connector",
]

# Русские написания каллаутов -> латинский каллаут. Окончания перечислены явно,
# чтобы «трак» не ловил «трактовку». «хедшот» и «буст» не включены: в речи игроков
# это чаще слова «убийство в голову» и «буст рейтинга», чем зоны.
_RU_ALIASES: list[tuple[str, str]] = [
    (r"мид(?:л)?(?:а|е|у|ом)?", "Mid"),
    (r"[аa][\s\-]+(?:мейн|main)", "A main"),
    (r"[бb][\s\-]+(?:мейн|main)", "B main"),
    (r"(?:плент[\s\-]+[аa]|[аa][\s\-]+(?:плент|сайт))", "A site"),
    (r"(?:плент[\s\-]+[бb]|[бb][\s\-]+(?:плент|сайт))", "B site"),
    (r"т[\s\-]+спавн", "T spawn"),
    (r"квад(?:а|е|у|ом)?", "Quad"),
    (r"форклифт(?:а|е|у|ом)?", "Forklift"),
    (r"трак(?:а|е|у|ом)?", "Truck"),
    (r"сквики", "Squeaky"),
    (r"токсик(?:а|е|у|ом)?", "Toxic"),
    (r"х[еэ]вен(?:а|е|у|ом)?", "Heaven"),
    (r"хайве(?:й|я|е|ю|ем)", "Highway"),
    (r"вент(?:ы|ов|ах|ам|ами)", "Vents"),
    (r"чекерс(?:а|е|у|ом)?", "Checkers"),
    (r"гараж(?:а|е|у|ом)?", "Garage"),
    (r"санрум(?:а|е|у|ом)?", "Sunroom"),
    (r"коннектор(?:а|е|у|ом)?", "Connector"),
]

ALWAYS_ALLOWED: set[float] = {95.0}   # уровень ДИ: он заложен в ключах ci95_*

_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*•][ \t]*)?\**\d{1,2}[.)]\**(?=[ \t])", re.MULTILINE)
_NUMBER = re.compile(r"(?<![\w.,])([-−–]?)(\d+(?:[.,]\d+)?)(?!\d)")
_TOL = 1e-9


def _norm(name: str) -> str:
    return re.sub(r"[\s\-_]+", "", name).casefold()


def _flex(name: str) -> str:
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", name.strip()) if p]
    return r"[\s\-]?".join(parts)


def extract_numbers(text: str) -> list[float]:
    """Все числа текста: "1,9" и "1.9", "34%", "-28", "−4.9". Маркеры списков в начале строки пропускаются."""
    cleaned = _LIST_MARKER.sub(" ", text)
    out: list[float] = []
    for m in _NUMBER.finditer(cleaned):
        value = float(m.group(2).replace(",", "."))
        if m.group(1):
            value = -value
        out.append(value + 0.0)
    return out


def allowed_numbers(findings: dict) -> set[float]:
    """Модули всех числовых листьев findings (bool не считается числом).

    Знак намеренно игнорируется, а округление не допускается: LLM должна копировать числа
    так, как они записаны в JSON. "-28" в тексте при diff = -28 проходит, "5" при листе 4.9 — нет.
    """
    out: set[float] = set()

    def walk(node) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            v = float(node)
            if math.isfinite(v):
                out.add(abs(v) + 0.0)
        elif isinstance(node, dict):
            for child in node.values():
                walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(findings)
    return out


def findings_zones(findings: dict) -> set[str]:
    """Значения ключей zone_from / zone_to на любой глубине."""
    out: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key in ("zone_from", "zone_to") and isinstance(child, str):
                    out.add(child)
                else:
                    walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(findings)
    return out


def find_zone_mentions(text: str, glossary_names: Iterable[str] = ()) -> list[tuple[int, int, str]]:
    """Упоминания зон: (start, end, каноническое имя). Длинные имена раньше коротких, пересечения отбрасываются."""
    names = sorted({n.strip() for n in [*CACHE_CALLOUTS, *glossary_names] if n and n.strip()},
                   key=lambda n: (-len(n), n))
    spans: list[tuple[int, int, str]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e, _ in spans)

    for name in names:
        for m in re.finditer(rf"(?<!\w){_flex(name)}(?!\w)", text, flags=re.IGNORECASE):
            if free(m.start(), m.end()):
                spans.append((m.start(), m.end(), name))
    for pattern, canon in _RU_ALIASES:
        for m in re.finditer(rf"(?<!\w){pattern}(?!\w)", text, flags=re.IGNORECASE):
            if free(m.start(), m.end()):
                spans.append((m.start(), m.end(), canon))
    return sorted(spans)


def _zone_allowed(name: str, allowed: set[str]) -> bool:
    target = _norm(name)
    for zone in allowed:
        if _norm(zone) == target:
            return True
        if any(_norm(part) == target for part in zone.split("/")):
            return True
    return False


# --- направление разниц -------------------------------------------------------------------------
# Предложение режется по . ! ? и переводу строки, но не внутри десятичной дроби "1.9".
# Сокращения с точками сначала схлопываются, чтобы "28 п.п. меньше" осталось одним предложением.
_ABBREV = [(re.compile(r"п\.\s?п\.", re.IGNORECASE), "пп"), (re.compile(r"т\.\s?е\.", re.IGNORECASE), "то есть"),
           (re.compile(r"т\.\s?к\.", re.IGNORECASE), "так как"), (re.compile(r"т\.\s?д\.", re.IGNORECASE), "так далее"),
           (re.compile(r"(?<!\w)сек\.", re.IGNORECASE), "сек")]
_SENTENCE_SPLIT = re.compile(r"[!?\n]|(?<!\d)\.|\.(?!\d)")
_UNIT = re.compile(r"\s*(%|пп(?!\w)|процент\w*|секунд\w*|сек(?!\w)|с(?!\w))", re.IGNORECASE)
_BLIND_UNITS = re.compile(r"^(?:с|сек|секунд\w*)$", re.IGNORECASE)
_TF_UNITS = re.compile(r"^(?:%|пп|процент\w*)$", re.IGNORECASE)
_TF_WORDS = re.compile(r"(?<!\w)(?:тимфлеш\w*|своих|союзник\w*|тиммейт\w*)", re.IGNORECASE)
_UP = re.compile(r"(?<!\w)(?:больше|выше|дольше|чаще|рост\w*|прибав\w*)", re.IGNORECASE)
_DOWN = re.compile(r"(?<!\w)(?:меньше|ниже|реже|короче|сниж\w*)", re.IGNORECASE)
_SAFETY = re.compile(r"(?<!\w)(?:безопаснее|меньше\s+риск\w*|реже\s+ослеп\w*)", re.IGNORECASE)

# Слова-числа прячут числа от проверки, поэтому запрещены. Порядковые («третьем», «четвертой») не ловятся.
NUMBER_WORDS = re.compile(
    r"(?<!\w)(?:треть(?:ю|и)?|третей|половин\w*|четверт(?:ь|и|ью|ей)|вдвое|втрое|в\s+два\s+раза|в\s+три\s+раза)(?!\w)",
    re.IGNORECASE)


def _sentences(masked: str) -> list[str]:
    out = _LIST_MARKER.sub(" ", masked)
    for pattern, repl in _ABBREV:
        out = pattern.sub(repl, out)
    return [s for s in _SENTENCE_SPLIT.split(out) if s.strip()]


def _claimed(sentence: str) -> str | None:
    up, down = bool(_UP.search(sentence)), bool(_DOWN.search(sentence))
    if up == down:
        return None
    return "up" if up else "down"


def _same(x: float, d: float) -> bool:
    return abs(abs(x) - abs(d)) < _TOL


def check_directions(masked: str, findings: dict) -> list[dict]:
    """Разницы diff, упомянутые с неверным направлением.

    Текст описывает «альтернатива минус лайнап игрока», поэтому «больше» должно стоять при d > 0,
    «меньше» — при d < 0. Метрика определяется по единице сразу после числа (с/сек/секунд — ослепление;
    п.п./%/процент вместе со словом о своих — тимфлеш), направление — по сравнительным словам в том же
    предложении. Если метрику или направление определить нельзя, вердикта нет. Нулевые разницы не проверяются.
    Отдельно: dt > 0 и в предложении с этим числом слово «безопаснее»/«меньше риска»/«реже ослеп…» —
    тоже неверное направление (у альтернативы тимфлешей больше).
    """
    bad: list[dict] = []

    def add(item: dict) -> None:
        if item not in bad:
            bad.append(item)

    items = [f for f in findings.get("findings") or [] if isinstance(f, dict) and isinstance(f.get("diff"), dict)]
    for sentence in _sentences(masked):
        claimed = _claimed(sentence)
        has_tf_words = bool(_TF_WORDS.search(sentence))
        safety = bool(_SAFETY.search(sentence))
        for m in _NUMBER.finditer(sentence):
            x = float(m.group(2).replace(",", "."))
            unit_m = _UNIT.match(sentence, m.end())
            unit = unit_m.group(1) if unit_m else ""
            for f in items:
                db, dt = f["diff"].get("blind_s"), f["diff"].get("teamflash_pct")
                fid = f.get("id")
                if isinstance(dt, (int, float)) and dt > 0 and safety and _same(x, dt):
                    add({"finding": fid, "metric": "teamflash_pct", "value": x + 0.0, "claimed": "down"})
                if claimed is None:
                    continue
                if _BLIND_UNITS.match(unit):
                    metric, d = "blind_s", db
                elif _TF_UNITS.match(unit) and has_tf_words:
                    metric, d = "teamflash_pct", dt
                else:
                    continue
                if not isinstance(d, (int, float)) or d == 0 or not _same(x, d):
                    continue
                if (claimed == "up") != (d > 0):
                    add({"finding": fid, "metric": metric, "value": x + 0.0, "claimed": claimed})
    return bad


def find_number_words(text: str) -> list[str]:
    out: list[str] = []
    for m in NUMBER_WORDS.finditer(text):
        word = re.sub(r"\s+", " ", m.group(0).casefold())
        if word not in out:
            out.append(word)
    return out


def check(text: str, findings: dict, glossary_names: set[str]) -> dict:
    """Результат: ok, bad_numbers (уникальные), bad_zones, bad_directions, bad_words, n_numbers,
    n_bad_number_occurrences (каждое вхождение числа не из findings).

    Слова-числа («треть», «в два раза»…) выносятся в отдельный список bad_words, а не в bad_numbers:
    bad_numbers остаётся списком чисел. ok требует, чтобы все четыре списка были пусты.
    """
    mentions = find_zone_mentions(text, glossary_names)
    allowed_z = findings_zones(findings)
    bad_zones: list[str] = []
    for _, _, name in mentions:
        if not _zone_allowed(name, allowed_z) and name not in bad_zones:
            bad_zones.append(name)

    chars = list(text)
    for start, end, _ in mentions:
        chars[start:end] = " " * (end - start)
    masked = "".join(chars)
    numbers = extract_numbers(masked)

    allowed = allowed_numbers(findings) | ALWAYS_ALLOWED
    bad_numbers: list[float] = []
    n_bad = 0
    for x in numbers:
        if not any(abs(abs(x) - a) < _TOL for a in allowed):
            n_bad += 1
            if x not in bad_numbers:
                bad_numbers.append(x)

    bad_directions = check_directions(masked, findings)
    bad_words = find_number_words(masked)
    ok = not bad_numbers and not bad_zones and not bad_directions and not bad_words
    return {"ok": ok, "bad_numbers": bad_numbers, "bad_zones": bad_zones, "bad_directions": bad_directions,
            "bad_words": bad_words, "n_numbers": len(numbers), "n_bad_number_occurrences": n_bad}
