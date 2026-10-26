"""Сборка одного самодостаточного HTML-отчёта: inline CSS, картинки base64, без внешних ресурсов."""
from __future__ import annotations

import base64
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from utility_coach.advisor import MODE_RU
from utility_coach.config import DATA, MIN_LINEUP_N, OUT
from utility_coach.zones import load_glossary

TEMPLATES = Path(__file__).resolve().parent / "templates"
FIGURES = ("zone_map", "calibration", "importance", "rating")

SECTIONS = [
    "1. Задача и идея",
    "2. Датасет и ограничения",
    "3. Карта зон",
    "4. xFlash: валидация, калибровка, SHAP",
    "5. Гипотеза о рейтинге",
    "6. Карточки советов",
    "7. Оценка LLM-слоя",
    "8. Что дальше",
]

LIMITATIONS = [
    "Нет исхода раунда, экономики и позиций игроков вне событий гранат, поэтому нельзя оценить влияние гранат на победу в раунде.",
    "Нет тика начала раунда: время в раунде приближённое (от первой гранаты раунда).",
    "Эффект смоков не измеряется.",
    "Рейтинг есть только примерно у 24% записей «игрок × матч».",
    "Игроки почти не повторяются (96% встречаются в одном матче), поэтому историю игрока и личные тренды не построить: персональный разбор возможен только по одному матчу.",
    "Нет геометрии карты и каллаутов: зоны получены кластеризацией и подписаны вручную.",
    "Только Cache и только Premier/MM: на другие карты и уровни игры результат автоматически не переносится.",
    "Лайнапы покрывают примерно половину флешек: остальные броски попадают в клетки сетки, где меньше 30 флешек, и для них рекомендаций нет.",
    "flash_kill засчитывается, если ослеплённый враг погиб, без проверки команды убийцы (редкие тимкиллы попадают в счёт).",
    "ДИ корреляций с рейтингом считаются в предположении независимых пар «игрок × матч».",
    "Зоны (KMeans) построены по всем матчам, включая hold-out. Это не утечка таргета, но разметка зон видела и тестовые броски.",
    "Каждый совет относится к одному матчу, в котором у игрока всего несколько флешек. Статистическая опора — лайнапы с большим n, а не история игрока.",
    "Сравнение лайнапов наблюдательное: разница смешивает позицию броска, место приземления, тип броска и ситуацию в раунде; это не причинный эффект смены броска.",
    "Альтернатива выбирается в той же зоне броска, но точка броска может быть в сотнях юнитов от точки игрока; расстояние указано в карточке.",
    "Средние лайнапов у выбранной альтернативы завышены эффектом отбора лучшего из многих (winner's curse); прогноз модели xFlash показан рядом для сравнения.",
    "Стандартная ошибка лайнапа приближает кластеризацию по матчам через число матчей (n_eff); ковариация ослепления и тимфлеша не учитывается.",
    "HE и молотовы в этой версии не анализируются: урон по ним собран (damage_events.parquet), но агрегаты по лайнапам отложены.",
]

NEXT_STEPS = [
    "Проверить пользу советов на игроках: A/B-тест — растёт ли xFlash и падает ли доля тимфлешей у тех, кто получил карточку.",
    "Если в выгрузке появятся позиции игроков и исход раунда, оценивать вклад флешки в раунд, а не только секунды ослепления.",
    "Заменить кластерные зоны на геометрию карты и официальные каллауты, показывать лайнап на мини-карте.",
    "Расширить подход на другие карты и на HE и молотовы: модель урона по лайнапам.",
    "Для игроков, у которых есть несколько матчей, добавить историю и личные тренды.",
    "Использовать проверку faithfulness в продукте как фильтр: непроверенный текст не показывается, вместо него шаблон.",
]

SPLIT_TITLES = {
    "cv": "Кросс-валидация на train (GroupKFold(5) по матчам)",
    "holdout": "Hold-out: 20% матчей, не участвовавших в обучении",
}
RATING_METRIC_RU = {
    "blind_minus_x": "факт − xFlash, с",
    "tf_rate": "доля тимфлешей",
    "flash_kill_rate": "конверсия во флеш-фраг",
}


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _img(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _to_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        if abs(v) >= 1000:
            return f"{v:.0f}"
        if v == 0 or abs(v) >= 0.001:
            return f"{v:.3f}"
        return f"{v:.2e}"
    return str(v)


def _thousands(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


def _flatten(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten(v, key))
        elif isinstance(v, (bool, int, float)):
            out.append((key, v))
    return out


def _table(title: str, src: dict) -> dict | None:
    models = [m for m, v in src.items() if isinstance(v, dict)]
    per_model = {m: dict(_flatten(src[m])) for m in models}
    keys = list(dict.fromkeys(k for m in models for k in per_model[m]))
    rows = [{"metric": k, "vals": [per_model[m].get(k) for m in models]} for k in keys]
    return {"title": title, "models": models, "rows": rows} if rows else None


def _metric_tables(metrics) -> list[dict]:
    if not isinstance(metrics, dict):
        return []
    tables = []
    for split, title in SPLIT_TITLES.items():
        block = metrics.get(split)
        if not isinstance(block, dict) or not block:
            continue
        if all(isinstance(v, dict) and isinstance(v.get("lgbm"), dict) for v in block.values()):
            # реальная форма: {цель: {lgbm, baseline_*}} — отдельная таблица на цель, модели в столбцах
            parts = [_table(f"{title} — {target}", sub) for target, sub in block.items()]
        else:
            parts = [_table(title, block)]
        tables.extend(t for t in parts if t)
    if not tables:
        flat = _flatten({k: v for k, v in metrics.items() if k != "importance"})
        if flat:
            tables.append({"title": "Метрики", "models": ["значение"],
                           "rows": [{"metric": k, "vals": [v]} for k, v in flat]})
    return tables


def _importance_tables(metrics) -> list[dict]:
    imp = metrics.get("importance") if isinstance(metrics, dict) else None
    if not isinstance(imp, dict) or not imp:
        return []
    if all(isinstance(v, (int, float)) for v in imp.values()):
        groups = {"Все модели": imp}
    else:
        groups = {str(k): v for k, v in imp.items() if isinstance(v, dict)}
    tables = []
    for name, d in groups.items():
        rows = sorted(((str(f), float(v)) for f, v in d.items() if isinstance(v, (int, float))),
                      key=lambda r: -abs(r[1]))[:10]
        if rows:
            tables.append({"target": name, "rows": rows})
    return tables


def _interpret(rho: float | None, lo: float | None, hi: float | None) -> str:
    if rho is None or lo is None or hi is None:
        return "недостаточно данных"
    if lo <= 0.0 <= hi:
        return "связь не обнаружена: ДИ включает 0"
    a = abs(rho)
    strength = "очень слабая" if a < 0.1 else "слабая" if a < 0.3 else "умеренная" if a < 0.5 else "сильная"
    sign = "положительная" if rho > 0 else "отрицательная"
    return f"{strength} {sign} связь: ДИ не включает 0"


def _rating_rows(hyp) -> list[dict]:
    rows: list[dict] = []

    def walk(node: dict, prefix: str) -> None:
        for key, val in node.items():
            if not isinstance(val, dict):
                continue
            name = f"{prefix}{key}"
            if "rho" in val:
                rho, lo, hi = (_to_float(val.get(k)) for k in ("rho", "ci_low", "ci_high"))
                rows.append({"name": RATING_METRIC_RU.get(str(key), name), "rho": rho, "ci_low": lo,
                             "ci_high": hi, "p": _to_float(val.get("p")), "n": val.get("n"),
                             "interp": _interpret(rho, lo, hi)})
            else:
                walk(val, f"{name}.")

    if isinstance(hyp, dict):
        walk(hyp, "")
    return rows


def _badge(advice: dict | None) -> tuple[str, str]:
    if advice is None:
        return "none", "совет не сгенерирован"
    if advice.get("source") != "llm":
        return "tmpl", "шаблонный совет: LLM-слой отключён"
    if advice.get("verified"):
        return "ok", "проверено: числа, зоны и направление разниц совпадают с findings"
    return "bad", "не верифицировано"


def _long(v, n: int) -> bool:
    return isinstance(v, list) and len(v) >= n


def _finding_ok(f) -> bool:
    if not isinstance(f, dict):
        return False
    if not all(isinstance(f.get(k), dict) for k in ("actual", "own_lineup", "alt_lineup", "diff")):
        return False
    own, alt = f["own_lineup"], f["alt_lineup"]
    return (all(_long(d.get(k), 2) for d in (own, alt) for k in ("ci95_blind_s", "ci95_teamflash_pct"))
            and _long(alt.get("throw_pos"), 1))


def _findings_ok(fd: dict) -> bool:
    return (isinstance(fd.get("match_summary"), dict) and isinstance(fd.get("findings"), list)
            and all(_finding_ok(f) for f in fd["findings"]))


def _cards(data: Path) -> list[dict]:
    cards = []
    for path in sorted((data / "findings").glob("*.json")):
        findings = _read_json(path)
        if not isinstance(findings, dict):
            continue
        card_id = str(findings.get("card_id", path.stem))
        advice = _read_json(data / "advice" / f"{card_id}.json")
        advice = advice if isinstance(advice, dict) else None
        cards.append({"card_id": card_id, "findings": findings, "advice": advice, "badge": _badge(advice),
                      "ok": _findings_ok(findings)})
    return cards


def _llm_stats(cards: list[dict]) -> dict:
    advs = [c["advice"] for c in cards if c["advice"]]
    llm = [a for a in advs if a.get("source") == "llm"]
    checks = [a.get("check") or {} for a in llm]
    total = sum(int(c.get("n_numbers") or 0) for c in checks)
    # каждое вхождение, а не уникальные значения; старый формат check без счётчика — по уникальным
    bad = sum(int(c["n_bad_number_occurrences"]) if isinstance(c.get("n_bad_number_occurrences"), int)
              else len(c.get("bad_numbers") or []) for c in checks)
    n_verified = sum(1 for a in llm if a.get("verified"))
    return {
        "n_advice": len(advs),
        "n_llm": len(llm),
        "n_verified": n_verified,
        "verified_pct": int(round(100 * n_verified / len(llm))) if llm else None,
        "numbers_total": total,
        "numbers_ok": total - bad,
        "numbers_ok_pct": int(round(100 * (total - bad) / total)) if total else None,
        "zones_bad": sum(len(c.get("bad_zones") or []) for c in checks),
        "directions_bad": sum(len(c.get("bad_directions") or []) for c in checks),
        "words_bad": sum(len(c.get("bad_words") or []) for c in checks),
        "retries": sum(max(int(a.get("attempts") or 1) - 1, 0) for a in llm),
        "models": sorted({str(a["model"]) for a in llm if a.get("model")}),
    }


def _rating_share(data: Path) -> int | None:
    try:
        pl = pd.read_parquet(data / "players.parquet", columns=["rating"])
    except (OSError, ValueError, KeyError):
        return None
    return int(round(100 * float(pl["rating"].notna().mean()))) if len(pl) else None


def _rating_conclusion(rows: list[dict]) -> str:
    ok = [r for r in rows if None not in (r["rho"], r["ci_low"], r["ci_high"])]
    zero = [r["name"] for r in ok if r["ci_low"] <= 0.0 <= r["ci_high"]]
    nonzero = [r for r in ok if not (r["ci_low"] <= 0.0 <= r["ci_high"])]
    parts = []
    if zero:
        parts.append("Связь не обнаружена (ДИ включает 0) для: " + ", ".join(zero) + ".")
    if nonzero:
        top = max(abs(r["rho"]) for r in nonzero)
        word = "очень слабая" if top < 0.1 else "слабая" if top < 0.3 else "умеренная"
        text = "Для остальных метрик ранговая связь" if zero else "Для всех метрик ранговая связь"
        parts.append(f"{text} статистически ненулевая, но {word} (max |ρ| = {top:.3f}).")
    return " ".join(parts)


def _dataset(data: Path) -> dict | None:
    try:
        fs = pd.read_parquet(data / "flash_scored.parquet", columns=["match_id", "is_test"])
    except (OSError, ValueError, KeyError):
        return None
    info = {
        "n_flashes": _thousands(len(fs)),
        "n_matches": _thousands(fs["match_id"].nunique()),
        "n_test_matches": _thousands(fs.loc[fs["is_test"].astype(bool), "match_id"].nunique()),
        "rating_share": None,
    }
    info["rating_share"] = _rating_share(data)
    return info


def _glossary(data: Path) -> list[tuple[int, str]]:
    try:
        return sorted((int(k), str(v)) for k, v in load_glossary(data).items())
    except Exception:  # noqa: BLE001 — глоссарий опционален, без zones.json отчёт всё равно собирается
        return []


def build_report(data: Path = DATA, out: Path = OUT / "report.html") -> Path:
    data, out = Path(data), Path(out)
    metrics = _read_json(data / "model_metrics.json")
    cards = _cards(data)
    hyp = _read_json(data / "rating_hypothesis.json")
    rating_rows = _rating_rows(hyp)
    min_flashes = hyp.get("min_flashes") if isinstance(hyp, dict) else None
    ctx = {
        "min_flashes": min_flashes if isinstance(min_flashes, int) else 5,
        "rating_share_txt": _rating_share(data) or 24,
        "rating_conclusion": _rating_conclusion(rating_rows),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sections": SECTIONS,
        "limitations": LIMITATIONS,
        "next_steps": NEXT_STEPS,
        "min_n": MIN_LINEUP_N,
        "dataset": _dataset(data),
        "figures": {name: _img(data / "figures" / f"{name}.png") for name in FIGURES},
        "glossary": _glossary(data),
        "metric_tables": _metric_tables(metrics),
        "importance": _importance_tables(metrics),
        "rating_rows": rating_rows,
        "cards": cards,
        "llm": _llm_stats(cards),
        "mode_ru": MODE_RU,
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    env.filters["fmt"] = _fmt
    html = env.get_template("report.html.j2").render(**ctx)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
