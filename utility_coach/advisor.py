"""LLM-советник: findings JSON -> короткий совет. LLM только формулирует, факты и проверка — в коде."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

from utility_coach.config import DATA, OPENROUTER_MODEL_DEFAULT, ROOT
from utility_coach.faithfulness import check, findings_zones
from utility_coach.zones import load_glossary

log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
BACKOFF_S = (2, 4, 8)          # первая попытка + 3 повтора с этими паузами
_sleep = time.sleep            # подменяется в тестах
FENCE = "`" * 3

MODE_RU = {"stand": "стоя", "crouch": "из приседа", "jump": "в прыжке", "run": "с бега"}


class LLMError(RuntimeError):
    """OpenRouter не дал пригодного ответа. Сообщение никогда не содержит ключ."""


SYSTEM_PROMPT = """Ты — тренер по Counter-Strike 2. Ты разбираешь один матч рядового игрока Premier на карте Cache и пишешь ему короткий понятный совет по флешкам.

Контекст CS2:
- Флешка полезна, когда ослепляет врагов (секунды ослепления врагов) и не ослепляет своих.
- Тимфлеш — флешка, которая ослепила своего тиммейта больше чем на секунду. Тимфлеш срывает заход и отдаёт врагу дуэль.
- Сторона T атакует точки A и B, сторона CT их защищает. Поле side в JSON — сторона игрока в этом раунде.
- Тип броска (mode): stand — стоя, crouch — из приседа, jump — в прыжке, run — с бега. Один и тот же прицел с другим типом броска даёт другую траекторию.

Что лежит в JSON:
- match_summary — итоги матча игрока: число флешек, секунды ослепления врагов, тимфлеши, флеш-фраги и перцентиль игрока по секундам ослепления среди всех игроков за матч. Это контекст, а не вывод.
- findings — конкретные флешки игрока: раунд, сторона, зона броска (zone_from) и зона приземления (zone_to), тип броска, фактический результат (actual), статистика его лайнапа (own_lineup) и лучшая альтернатива из той же зоны броска (alt_lineup).
- xflash_s — среднее фактическое ослепление врагов (секунды) у бросков этого лайнапа в обучающих матчах; model_xflash_s — средний прогноз модели xFlash для тех же бросков; teamflash_pct — доля бросков с тимфлешем в процентах; n — число бросков этого лайнапа в данных; ci95_* — доверительный интервал; yaw и pitch — углы прицела; throw_pos — точка броска в координатах карты; throw_dist_u — расстояние в юнитах карты от точки броска игрока до точки броска альтернативы; diff — разница «альтернатива минус лайнап игрока»; times_in_match — сколько раз за матч игрок бросил эту флешку.

Жёсткие правила:
- Используй только числа из JSON и пиши их ровно так, как они записаны. Не вычисляй новые числа: никаких сумм, разностей, отношений и «в разы».
- Копируй каждое число ровно так, как оно записано в JSON: без округления, без пересчёта, без своих отношений, сумм и разностей.
- Не используй дроби и слова-числа («треть», «половина», «четверть», «вдвое», «втрое», «в два раза», «в три раза»): все числа пиши цифрами, копируя их из JSON.
- Разницу из diff всегда описывай со стороны альтернативы: «альтернатива даёт на … больше/меньше, чем твой лайнап». Положительное значение — «больше», отрицательное — модулем со словом «меньше».
- Разницу тимфлешей пиши только в п.п. («на 11 п.п. больше тимфлешей»), никогда не «процентов».
- Описывай компромиссы честно. Если у альтернативы teamflash_pct выше, прямо скажи, что тимфлешей у неё больше. Альтернатива рекомендуется потому, что итоговая оценка (ослепление врагов минус штраф за тимфлеши) у неё выше, а не потому, что она безопаснее.
- Альтернатива бросается из той же зоны броска, но не обязательно с той же точки. Никогда не пиши «с той же позиции» или «в той же точке»: пиши «из той же зоны броска» и укажи расстояние до точки броска альтернативы из throw_dist_u в юнитах.
- Не пиши в тексте названия полей JSON (xflash_s, model_xflash_s, ci95_blind_s, ci95_teamflash_pct, teamflash_pct, throw_pos, throw_dist_u, diff и другие): называй величины по-русски.
- Упоминай только зоны из полей zone_from и zone_to этого JSON и пиши их названия точно как в JSON: латинские каллауты вроде «A main» или «Зона N». Не переводи названия зон на русский и не заменяй их сленгом.
- Не называй броски тактическими ярлыками («поп-флешка», «флешка под заход» и подобные): в JSON таких меток нет.
- Не придумывай позиции, лайнапы, тайминги, раскидки, названия зон и числа, которых нет в JSON.
- Не упоминай другие карты.
- Не пиши номера советов внутри текста («совет 1») и не используй числа для счёта («2 раза», «3 совета»): единственные числа в ответе — из JSON и номера пунктов списка.

Формат ответа:
Первая строка — короткий вывод о флешках игрока в этом матче, одно-два предложения.
Затем от двух до четырёх советов нумерованным списком, каждый с новой строки и с номером вида «1.» в начале. Каждый совет ссылается на раунд и опирается на числа из JSON: что сделал игрок, что в среднем даёт его лайнап и что даёт альтернатива (тип броска, yaw, pitch, расстояние до её точки броска).
Пиши по-русски, обращайся на «ты», без заголовков, таблиц и markdown-разметки."""


def _dotenv() -> dict[str, str]:
    path = ROOT / ".env"
    if not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v}


def load_key() -> str | None:
    """Ключ OpenRouter из ROOT/.env (open_router_key) или из окружения. Значение нигде не логируется."""
    value = _dotenv().get("open_router_key") or os.environ.get("open_router_key") or os.environ.get("OPEN_ROUTER_KEY")
    value = (value or "").strip()
    return value or None


def load_model() -> str:
    value = _dotenv().get("OPENROUTER_MODEL") or os.environ.get("OPENROUTER_MODEL") or OPENROUTER_MODEL_DEFAULT
    return value.strip()


def build_user_prompt(findings: dict, glossary_names: list[str]) -> str:
    zones = sorted(findings_zones(findings))
    lines = [
        f"Разбор матча на Cache для игрока «{findings.get('player', '')}».",
        "Факты (JSON — единственный источник чисел и зон):",
        FENCE + "json",
        json.dumps(findings, ensure_ascii=False, indent=2),
        FENCE,
        "Зоны, которые можно упоминать: " + (", ".join(zones) if zones else "нет"),
        "Глоссарий зон Cache для понимания карты (упоминать можно только зоны из строки выше): "
        + ", ".join(sorted(set(glossary_names))),
        "Типы броска в поле mode: " + ", ".join(f"{k} — {v}" for k, v in MODE_RU.items()) + ".",
        "Напиши разбор по формату из инструкции.",
    ]
    return "\n".join(lines)


def call_openrouter(messages: list[dict], model: str, api_key: str, temperature: float = 0.2,
                    timeout: int = 60) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "X-Title": "Utility Coach PoC"}
    body = {"model": model, "messages": messages, "temperature": temperature}
    last = "нет ответа"
    for attempt in range(len(BACKOFF_S) + 1):
        if attempt:
            _sleep(BACKOFF_S[attempt - 1])
        try:
            resp = requests.post(API_URL, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as exc:
            last = type(exc).__name__
            log.warning("OpenRouter: сетевая ошибка %s (попытка %d)", last, attempt + 1)
            continue
        status = resp.status_code
        if status == 429 or status >= 500:
            last = f"HTTP {status}"
            log.warning("OpenRouter: %s (попытка %d)", last, attempt + 1)
            continue
        if status >= 400:
            raise LLMError(f"OpenRouter HTTP {status}")
        try:
            content = resp.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("неожиданный формат ответа OpenRouter") from exc
        if content.strip():
            return content.strip()
        last = "пустой ответ"
        log.warning("OpenRouter: пустой ответ (попытка %d)", attempt + 1)
    raise LLMError(f"OpenRouter недоступен после {len(BACKOFF_S) + 1} попыток: {last}")


def _mode(mode) -> str:
    return MODE_RU.get(str(mode), str(mode))


def _diff_phrase(diff: dict) -> str:
    b, t = diff["blind_s"], diff["teamflash_pct"]
    if b > 0:
        blind = f"на {b} с больше ослепления врагов"
    elif b < 0:
        blind = f"на {abs(b)} с меньше ослепления врагов"
    else:
        blind = "ослепление врагов на том же уровне"
    if t < 0:
        tf = f"на {abs(t)} п.п. меньше тимфлешей"
    elif t > 0:
        tf = f"на {t} п.п. больше тимфлешей"
    else:
        tf = "доля тимфлешей та же"
    return f"{blind}, {tf}"


def template_advice(findings: dict) -> str:
    """Детерминированный совет без LLM. Использует только числа и зоны из findings."""
    s = findings.get("match_summary", {})
    head = str(findings.get("player", "Игрок"))
    if findings.get("rating") is not None:
        head += f" (рейтинг {findings['rating']})"
    head += f": флешек за матч — {s.get('n_flashes')}, ослепление врагов — {s.get('enemy_blind_s')} с"
    if s.get("enemy_blind_pct") is not None:
        head += f" (перцентиль {s['enemy_blind_pct']} среди всех игроков за матч)"
    head += f", тимфлешей — {s.get('teamflashes')}, флеш-фрагов — {s.get('flash_kills')}."
    items = findings.get("findings") or []
    if not items:
        return head + "\nЗначимых находок по отдельным флешкам нет."
    lines = [head]
    for i, f in enumerate(items, start=1):
        own, alt, act = f["own_lineup"], f["alt_lineup"], f["actual"]
        tf_note = ", был тимфлеш" if act.get("teamflash") else ""
        times = f.get("times_in_match", 1)
        repeat = f" Таких бросков за матч: {times}." if times and times > 1 else ""
        pos = ", ".join(str(v) for v in alt["throw_pos"])
        dist = f", в {alt['throw_dist_u']} юнитах от твоей" if alt.get("throw_dist_u") is not None else ""
        lines.append(
            f"{i}. Раунд {f['round']}, сторона {f['side']}: флешка из {f['zone_from']} в {f['zone_to']} "
            f"({_mode(f['mode'])}) дала {act['enemy_blind_s']} с ослепления врагов{tf_note}.{repeat} "
            f"Этот лайнап в среднем даёт {own['xflash_s']} с и {own['teamflash_pct']}% тимфлешей "
            f"(n = {own['n']}). Из той же зоны броска лучше бросать в {alt['zone_to']} {_mode(alt['mode'])}: "
            f"yaw {alt['yaw']}, pitch {alt['pitch']}, точка броска ({pos}){dist}. Так в среднем {alt['xflash_s']} с "
            f"и {alt['teamflash_pct']}% тимфлешей (n = {alt['n']}): {_diff_phrase(f['diff'])}."
        )
    return "\n".join(lines)


def _cache_path(cache_dir: Path, model: str, messages: list[dict]) -> Path:
    payload = json.dumps({"model": model, "messages": messages}, ensure_ascii=False, sort_keys=True)
    return Path(cache_dir) / f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}.json"


def _cached_call(messages: list[dict], model: str, api_key: str, cache_dir: Path) -> str:
    path = _cache_path(cache_dir, model, messages)
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))["response"]
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("LLM-кэш: файл %s повреждён, запрос заново", path.name[:12])
        else:
            if isinstance(cached, str) and cached.strip():
                log.info("LLM-кэш: попадание %s", path.name[:12])
                return cached
            log.warning("LLM-кэш: пустой ответ в %s, запрос заново", path.name[:12])
    text = call_openrouter(messages, model, api_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "messages": messages, "response": text},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return text


def _num_str(x: float) -> str:
    return f"{x:g}"


_METRIC_RU = {"blind_s": ("ослепления врагов", "с"), "teamflash_pct": ("тимфлешей", "п.п.")}
_DIR_RU = {"up": "больше", "down": "меньше"}


def _direction_line(item: dict) -> str:
    what, unit = _METRIC_RU.get(item.get("metric"), (str(item.get("metric")), ""))
    claimed = item.get("claimed")
    actual = _DIR_RU["down" if claimed == "up" else "up"]
    return (f"находка {item.get('finding')}: у альтернативы на {_num_str(float(item.get('value', 0)))} {unit} "
            f"{actual} {what}, чем у лайнапа игрока, а в тексте сказано «{_DIR_RU.get(claimed, claimed)}»"
            + (" или «безопаснее»" if item.get("metric") == "teamflash_pct" and claimed == "down" else ""))


def _fix_message(chk: dict) -> str:
    nums = ", ".join(_num_str(x) for x in chk.get("bad_numbers") or []) or "нет"
    zones = ", ".join(chk.get("bad_zones") or []) or "нет"
    dirs = "; ".join(_direction_line(d) for d in chk.get("bad_directions") or []) or "нет"
    words = ", ".join(chk.get("bad_words") or []) or "нет"
    return ("Проверка нашла в ответе ошибки относительно JSON.\n"
            f"Числа не из JSON: {nums}.\n"
            f"Зоны не из JSON: {zones}.\n"
            f"Перепутано направление разницы: {dirs}.\n"
            f"Слова-числа (запрещены, пиши только числа из JSON цифрами): {words}.\n"
            "Перепиши разбор целиком в том же формате: убери эти числа, зоны и слова, исправь направление "
            "разниц и используй только числа и зоны из JSON.")


def advise(findings: dict, glossary_names: set[str], cache_dir: Path) -> dict:
    card_id = findings["card_id"]
    template = template_advice(findings)
    result = {
        "card_id": card_id, "text": template, "template_text": template, "verified": True,
        "attempts": 0, "check": check(template, findings, glossary_names), "source": "template", "model": None,
    }
    api_key = load_key()
    if not api_key:
        log.info("карточка %s: ключа OpenRouter нет, шаблонный совет", card_id)
        return result

    model = load_model()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(findings, sorted(glossary_names))}]
    try:
        text = _cached_call(messages, model, api_key, cache_dir)
        attempts = 1
        chk = check(text, findings, glossary_names)
        if not chk["ok"]:
            messages = messages + [{"role": "assistant", "content": text},
                                   {"role": "user", "content": _fix_message(chk)}]
            text = _cached_call(messages, model, api_key, cache_dir)
            attempts = 2
            chk = check(text, findings, glossary_names)
    except (LLMError, requests.RequestException) as exc:
        log.warning("карточка %s: LLM недоступна (%s), шаблонный совет", card_id, type(exc).__name__)
        return result

    result.update(text=text, verified=bool(chk["ok"]), attempts=attempts, check=chk, source="llm", model=model)
    return result


def run_advise(data: Path = DATA) -> None:
    data = Path(data)
    names = set(load_glossary(data).values())
    out_dir = data / "advice"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    for path in sorted((data / "findings").glob("*.json")):
        findings = json.loads(path.read_text(encoding="utf-8"))
        res = advise(findings, names, data / "llm_cache")
        (out_dir / f"{res['card_id']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
        print(f"[advise] {res['card_id']}: source={res['source']} verified={res['verified']} "
              f"attempts={res['attempts']}")
