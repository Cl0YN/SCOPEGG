import json
import logging

import pytest
import requests

from utility_coach import advisor as A
from utility_coach.config import OPENROUTER_MODEL_DEFAULT
from utility_coach.faithfulness import check

FINDINGS = {
    "card_id": "A", "player": "Игрок A", "rating": 14200,
    "match_summary": {"n_flashes": 9, "enemy_blind_s": 11.3, "teamflashes": 3, "flash_kills": 1,
                      "enemy_blind_pct": 28},
    "findings": [{
        "id": "f1", "round": 7, "side": "T", "zone_from": "A main", "zone_to": "A site", "mode": "jump",
        "times_in_match": 2,
        "actual": {"enemy_blind_s": 0.4, "teamflash": True},
        "own_lineup": {"xflash_s": 0.9, "teamflash_pct": 34, "n": 812, "ci95_blind_s": [0.8, 1.0],
                       "ci95_teamflash_pct": [31, 37]},
        "alt_lineup": {"zone_to": "A site", "mode": "stand", "xflash_s": 2.1, "teamflash_pct": 6, "n": 420,
                       "ci95_blind_s": [1.9, 2.3], "ci95_teamflash_pct": [4, 8], "yaw": 22.1, "pitch": -4.9,
                       "throw_pos": [-647, 1204, 1688]},
        "diff": {"blind_s": 1.2, "teamflash_pct": -28},
    }],
}
NAMES = {"A main", "A site", "Mid", "B main"}
SECRET = "sk-or-v1-TEST-SECRET-123"
GOOD_TEXT = ("Игрок A: в раунде 7 флешка из A main дала 0.4 с ослепления.\n"
             "1. Лайнап в A site стоя даёт 2.1 с и 6% тимфлешей, на 28 п.п. меньше.")
BAD_TEXT = "Бросай из Mid, получишь 99 с ослепления."


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ROOT", tmp_path)          # настоящий .env не читается
    for var in ("open_router_key", "OPEN_ROUTER_KEY", "OPENROUTER_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(A, "_sleep", lambda s: None)


def _fake_llm(monkeypatch, replies):
    calls = []

    def fake(messages, model, api_key, temperature=0.2, timeout=60):
        calls.append({"messages": [dict(m) for m in messages], "model": model, "api_key": api_key})
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(A, "call_openrouter", fake)
    monkeypatch.setattr(A, "load_key", lambda: SECRET)
    return calls


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


OK_PAYLOAD = {"choices": [{"message": {"content": "  готово  "}}]}


def test_system_prompt_rules():
    p = A.SYSTEM_PROMPT.lower()
    for word in ("тимфлеш", "поп-флеш", "под заход", "только", "json", "другие карты", "cache"):
        assert word in p
    assert "T" in A.SYSTEM_PROMPT and "CT" in A.SYSTEM_PROMPT


def test_load_key_from_dotenv(tmp_path):
    (tmp_path / ".env").write_text(f"open_router_key={SECRET}\n", encoding="utf-8")
    assert A.load_key() == SECRET


def test_load_key_env_fallback_and_none(monkeypatch):
    assert A.load_key() is None
    monkeypatch.setenv("open_router_key", SECRET)
    assert A.load_key() == SECRET


def test_load_model_default_and_override(tmp_path):
    assert A.load_model() == OPENROUTER_MODEL_DEFAULT
    (tmp_path / ".env").write_text("OPENROUTER_MODEL=some/model\n", encoding="utf-8")
    assert A.load_model() == "some/model"


def test_template_advice_passes_check():
    text = A.template_advice(FINDINGS)
    assert "Раунд 7" in text and "A main" in text and "в прыжке" in text
    res = check(text, FINDINGS, NAMES)
    assert res["ok"] is True, res


def test_build_user_prompt_contains_json_and_zones():
    p = A.build_user_prompt(FINDINGS, sorted(NAMES))
    assert '"xflash_s": 0.9' in p
    assert "Зоны, которые можно упоминать: A main, A site" in p


def test_no_key_uses_template(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("API не должен вызываться без ключа")

    monkeypatch.setattr(A, "call_openrouter", boom)
    res = A.advise(FINDINGS, NAMES, tmp_path / "cache")
    assert res["source"] == "template"
    assert res["verified"] is True
    assert res["text"] == res["template_text"] == A.template_advice(FINDINGS)
    assert res["model"] is None and res["attempts"] == 0
    assert set(res) == {"card_id", "text", "template_text", "verified", "attempts", "check", "source", "model"}


def test_llm_verified_first_try(tmp_path, monkeypatch):
    calls = _fake_llm(monkeypatch, [GOOD_TEXT])
    res = A.advise(FINDINGS, NAMES, tmp_path / "cache")
    assert (res["source"], res["verified"], res["attempts"]) == ("llm", True, 1)
    assert res["text"] == GOOD_TEXT
    assert res["model"] == OPENROUTER_MODEL_DEFAULT
    assert calls[0]["messages"][0] == {"role": "system", "content": A.SYSTEM_PROMPT}
    assert res["template_text"]


def test_cache_hit_skips_api(tmp_path, monkeypatch):
    calls = _fake_llm(monkeypatch, [GOOD_TEXT])
    first = A.advise(FINDINGS, NAMES, tmp_path / "cache")
    second = A.advise(FINDINGS, NAMES, tmp_path / "cache")
    assert len(calls) == 1
    assert second["text"] == first["text"]
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_failed_check_retries_with_errors(tmp_path, monkeypatch):
    calls = _fake_llm(monkeypatch, [BAD_TEXT, GOOD_TEXT])
    res = A.advise(FINDINGS, NAMES, tmp_path / "cache")
    assert len(calls) == 2
    retry = calls[1]["messages"]
    assert retry[-2] == {"role": "assistant", "content": BAD_TEXT}
    assert retry[-1]["role"] == "user"
    assert "99" in retry[-1]["content"] and "Mid" in retry[-1]["content"]
    assert (res["verified"], res["attempts"], res["source"]) == (True, 2, "llm")


def test_double_failure_marks_unverified(tmp_path, monkeypatch):
    _fake_llm(monkeypatch, [BAD_TEXT, BAD_TEXT])
    res = A.advise(FINDINGS, NAMES, tmp_path / "cache")
    assert (res["verified"], res["attempts"], res["source"]) == (False, 2, "llm")
    assert res["text"] == BAD_TEXT
    assert res["template_text"] == A.template_advice(FINDINGS)
    assert 99.0 in res["check"]["bad_numbers"] and "Mid" in res["check"]["bad_zones"]


def test_api_failure_falls_back_to_template(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise A.LLMError("OpenRouter недоступен")

    monkeypatch.setattr(A, "call_openrouter", broken)
    monkeypatch.setattr(A, "load_key", lambda: SECRET)
    res = A.advise(FINDINGS, NAMES, tmp_path / "cache")
    assert res["source"] == "template" and res["verified"] is True
    assert res["text"] == res["template_text"]


def test_key_never_leaks(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    _fake_llm(monkeypatch, [BAD_TEXT, GOOD_TEXT])
    res = A.advise(FINDINGS, NAMES, tmp_path / "cache")
    assert SECRET not in repr(res)
    assert SECRET not in json.dumps(res, ensure_ascii=False)
    assert SECRET not in caplog.text
    for path in (tmp_path / "cache").glob("*.json"):
        assert SECRET not in path.read_text(encoding="utf-8")


def test_call_openrouter_retries_then_succeeds(monkeypatch):
    sleeps, seen = [], []
    replies = [requests.ConnectionError("boom"), FakeResp(503), FakeResp(200, OK_PAYLOAD)]

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append((url, headers, json))
        r = replies[len(seen) - 1]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(A.requests, "post", fake_post)
    monkeypatch.setattr(A, "_sleep", sleeps.append)
    out = A.call_openrouter([{"role": "user", "content": "x"}], "m/x", SECRET)
    assert out == "готово"
    assert sleeps == [2, 4]
    url, headers, body = seen[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {SECRET}"
    assert body["model"] == "m/x" and body["temperature"] == 0.2


def test_call_openrouter_gives_up_after_backoff(monkeypatch):
    sleeps, n = [], []

    def fake_post(url, headers=None, json=None, timeout=None):
        n.append(1)
        return FakeResp(429)

    monkeypatch.setattr(A.requests, "post", fake_post)
    monkeypatch.setattr(A, "_sleep", sleeps.append)
    with pytest.raises(A.LLMError) as err:
        A.call_openrouter([{"role": "user", "content": "x"}], "m/x", SECRET)
    assert len(n) == 4
    assert sleeps == [2, 4, 8]
    assert SECRET not in str(err.value)


def test_call_openrouter_no_retry_on_401(monkeypatch):
    sleeps, n = [], []

    def fake_post(url, headers=None, json=None, timeout=None):
        n.append(1)
        return FakeResp(401)

    monkeypatch.setattr(A.requests, "post", fake_post)
    monkeypatch.setattr(A, "_sleep", sleeps.append)
    with pytest.raises(A.LLMError):
        A.call_openrouter([{"role": "user", "content": "x"}], "m/x", SECRET)
    assert len(n) == 1 and sleeps == []


def test_run_advise_writes_files(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "findings").mkdir(parents=True)
    (data / "findings" / "A.json").write_text(json.dumps(FINDINGS, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(A, "load_glossary", lambda d: {5: "A main", 6: "A site"})
    monkeypatch.setattr(A, "load_key", lambda: None)
    A.run_advise(data)
    adv = json.loads((data / "advice" / "A.json").read_text(encoding="utf-8"))
    assert adv["card_id"] == "A" and adv["source"] == "template" and adv["verified"] is True


def test_system_prompt_strict_copy_rules():
    p = A.SYSTEM_PROMPT
    assert "без округления" in p and "Не переводи названия зон" in p and "Зона N" in p
    # final fix F2
    assert "из той же зоны броска" in p and "throw_dist_u" in p
    assert "Никогда не пиши «с той же позиции» или «в той же точке»" in p
    assert "Не пиши в тексте названия полей JSON" in p and "ci95_blind_s" in p
    assert "только в п.п." in p and "никогда не «процентов»" in p
    assert "не потому, что она безопаснее" in p
    assert "Не используй дроби и слова-числа" in p and "в два раза" in p
    assert "Не называй броски тактическими ярлыками" in p
    assert "Поп-флешка — флешка, которая" not in p     # определение ярлыка убрано из контекста
    assert "с той же позиции (alt_lineup)" not in p


def test_template_advice_wording():
    f = json.loads(json.dumps(FINDINGS))
    f["findings"][0]["alt_lineup"]["throw_dist_u"] = 312
    text = A.template_advice(f)
    assert "той же позиции" not in text
    assert "Из той же зоны броска" in text and "312" in text
    assert check(text, f, NAMES)["ok"] is True


# --- final fix -----------------------------------------------------------------------------------
REVERSED_TEXT = ("Игрок A: в раунде 7 флешка из A main дала 0.4 с ослепления.\n"
                 "1. Альтернатива в A site стоя даёт на 28 п.п. больше тимфлешей. Своих она слепит в два раза реже.")


def test_fix_message_lists_bad_directions_and_words():
    chk = check(REVERSED_TEXT, FINDINGS, NAMES)
    assert chk["ok"] is False and chk["bad_directions"] and chk["bad_words"] == ["в два раза"]
    msg = A._fix_message(chk)
    assert "f1" in msg and "тимфлеш" in msg and "28" in msg and "меньше" in msg
    assert "в два раза" in msg
    assert "teamflash_pct" not in msg            # имена полей JSON в подсказку не попадают


def test_retry_after_reversed_direction(tmp_path, monkeypatch):
    calls = _fake_llm(monkeypatch, [REVERSED_TEXT, GOOD_TEXT])
    res = A.advise(FINDINGS, NAMES, tmp_path / "cache")
    assert len(calls) == 2
    assert "в два раза" in calls[1]["messages"][-1]["content"]
    assert (res["verified"], res["attempts"]) == (True, 2)


def test_corrupt_cache_file_is_a_miss(tmp_path, monkeypatch):
    calls = _fake_llm(monkeypatch, [GOOD_TEXT])
    cache = tmp_path / "cache"
    A.advise(FINDINGS, NAMES, cache)
    (path,) = list(cache.glob("*.json"))
    for broken in ("{not json", json.dumps({"model": "m"})):   # ValueError, KeyError
        path.write_text(broken, encoding="utf-8")
        res = A.advise(FINDINGS, NAMES, cache)
        assert res["text"] == GOOD_TEXT and res["verified"] is True
    assert len(calls) == 3
    assert json.loads(path.read_text(encoding="utf-8"))["response"] == GOOD_TEXT
