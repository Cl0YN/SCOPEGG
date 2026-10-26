import base64
import json
import re

from markupsafe import escape

from utility_coach import report as R

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
FINDINGS = {
    "card_id": "A", "player": "Игрок A", "rating": 14200,
    "match_summary": {"n_flashes": 9, "enemy_blind_s": 11.3, "teamflashes": 3, "flash_kills": 1,
                      "enemy_blind_pct": 28},
    "findings": [{
        "id": "f1", "round": 7, "side": "T", "zone_from": "A main", "zone_to": "A site", "mode": "jump",
        "times_in_match": 1,
        "actual": {"enemy_blind_s": 0.4, "teamflash": True},
        "own_lineup": {"xflash_s": 0.9, "teamflash_pct": 34, "n": 812, "ci95_blind_s": [0.8, 1.0],
                       "ci95_teamflash_pct": [31, 37]},
        "alt_lineup": {"zone_to": "A site", "mode": "stand", "xflash_s": 2.1, "teamflash_pct": 6, "n": 420,
                       "ci95_blind_s": [1.9, 2.3], "ci95_teamflash_pct": [4, 8], "yaw": 22.1, "pitch": -4.9,
                       "throw_pos": [-647, 1204, 1688]},
        "diff": {"blind_s": 1.2, "teamflash_pct": -28},
    }],
}
METRICS = {
    "cv": {
        "lgbm": {"enemy_blind_s": {"mae": 1.234, "r2": 0.081},
                 "teamflash": {"auc": 0.712, "brier": 0.151, "calibration": [[0.05, 0.04, 100]]}},
        "baseline_global": {"enemy_blind_s": {"mae": 1.402, "r2": 0.0}, "teamflash": {"auc": 0.5, "brier": 0.17}},
        "baseline_zonepair": {"enemy_blind_s": {"mae": 1.3, "r2": 0.04}, "teamflash": {"auc": 0.66, "brier": 0.16}},
    },
    "holdout": {
        "lgbm": {"enemy_blind_s": {"mae": 1.25, "r2": 0.079}, "teamflash": {"auc": 0.709, "brier": 0.152}},
        "baseline_global": {"enemy_blind_s": {"mae": 1.41, "r2": 0.0}, "teamflash": {"auc": 0.5, "brier": 0.17}},
        "baseline_zonepair": {"enemy_blind_s": {"mae": 1.31, "r2": 0.039}, "teamflash": {"auc": 0.65, "brier": 0.16}},
    },
    "importance": {"enemy_blind_s": {"to_x": 0.31, "yaw": 0.12}, "teamflash": {"zone_to": 0.2}},
}
HYP = {
    "blind_minus_x": {"rho": 0.031, "ci_low": -0.004, "ci_high": 0.066, "p": 0.08, "n": 5120},
    "tf_rate": {"rho": -0.12, "ci_low": -0.15, "ci_high": -0.09, "p": 1e-6, "n": 5120},
}
ADVICE = {
    "card_id": "A", "text": "Игрок A: в раунде 7 флешка из A main дала 0.4 с.",
    "template_text": "Шаблонный совет для Игрок A.", "verified": True, "attempts": 1,
    "check": {"ok": True, "bad_numbers": [], "bad_zones": [], "n_numbers": 2},
    "source": "llm", "model": "deepseek/deepseek-v4.1-flash",
}


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _make_artifacts(data, advice=ADVICE):
    (data / "figures").mkdir(parents=True)
    for name in ("zone_map", "calibration", "importance", "rating"):
        (data / "figures" / f"{name}.png").write_bytes(PNG_1PX)
    _write(data / "model_metrics.json", METRICS)
    _write(data / "rating_hypothesis.json", HYP)
    _write(data / "findings" / "A.json", FINDINGS)
    if advice is not None:
        _write(data / "advice" / "A.json", advice)


def _build(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(R, "load_glossary", lambda data: {5: "A main", 6: "A site"})
    data = tmp_path / "data"
    _make_artifacts(data, **kw)
    out = R.build_report(data, tmp_path / "out" / "report.html")
    return out.read_text(encoding="utf-8")


def test_report_has_all_sections_and_content(tmp_path, monkeypatch):
    html = _build(tmp_path, monkeypatch)
    for title in R.SECTIONS:
        assert title in html
    for item in R.LIMITATIONS:
        assert str(escape(item)) in html          # автоэкранирование: ' -> &#39;
    assert "data:image/png;base64," in html
    assert "Игрок A" in html and "A main" in html
    assert "enemy_blind_s.mae" in html and "baseline_zonepair" in html and "1.234" in html
    assert "связь не обнаружена" in html and "отрицательная" in html
    assert ADVICE["text"] in html and ADVICE["template_text"] in html
    assert "проверено: числа, зоны и направление разниц совпадают с findings" in html
    assert "1 из 1" in html


def test_report_has_no_external_resources_and_escapes(tmp_path, monkeypatch):
    evil = dict(ADVICE, text="<script>alert(1)</script> и <img src=http://x.test/a.png>")
    html = _build(tmp_path, monkeypatch, advice=evil)
    # только настоящие теги: экранированный "&lt;img src=http://" ресурсом не является
    assert not re.search(r"""<[^>]*\b(?:src|href)\s*=\s*["']?\s*(?:https?:)?//""", html, flags=re.IGNORECASE)
    assert "&lt;img src=http://x.test/a.png&gt;" in html
    low = html.lower()
    assert "<script" not in low and "<link" not in low and "@import" not in low and "url(" not in low
    assert "&lt;script&gt;" in html
    assert "prefers-color-scheme: dark" in html
    assert 'name="viewport"' in html


def test_report_without_advice(tmp_path, monkeypatch):
    html = _build(tmp_path, monkeypatch, advice=None)
    for title in R.SECTIONS:
        assert title in html
    assert "не сгенерировано" in html
    assert "Игрок A" in html


def test_report_template_source_says_llm_disabled(tmp_path, monkeypatch):
    tmpl = dict(ADVICE, source="template", model=None, attempts=0, text=ADVICE["template_text"])
    html = _build(tmp_path, monkeypatch, advice=tmpl)
    assert "LLM-слой отключён" in html


def test_report_empty_data_dir(tmp_path):
    out = R.build_report(tmp_path / "empty", tmp_path / "out" / "report.html")
    html = out.read_text(encoding="utf-8")
    for title in R.SECTIONS:
        assert title in html
    assert html.count("не сгенерировано") >= 5


# --- fix round 1 ---------------------------------------------------------------------------------
def _real_split(blind_mae, tf_auc):
    return {
        "xflash_blind": {"target": "enemy_blind_s",
                         "lgbm": {"mae": blind_mae, "r2": 0.14},
                         "baseline_global": {"mae": 2.2, "r2": 0.0},
                         "baseline_zonepair": {"mae": 2.1, "r2": 0.05}},
        "xflash_tf": {"target": "teamflash",
                      "lgbm": {"auc": tf_auc, "brier": 0.144, "calibration": [[0.1, 0.1, 10]]},
                      "baseline_global": {"auc": 0.5, "brier": 0.164},
                      "baseline_zonepair": {"auc": 0.64, "brier": 0.156}},
    }


def test_metric_tables_real_shape():
    tables = R._metric_tables({"cv": _real_split(1.92, 0.73), "holdout": _real_split(1.91, 0.72)})
    assert len(tables) == 4  # 2 сплита x 2 цели
    blind = tables[0]
    assert blind["models"] == ["lgbm", "baseline_global", "baseline_zonepair"]
    assert [r["metric"] for r in blind["rows"]] == ["mae", "r2"]
    assert blind["rows"][0]["vals"] == [1.92, 2.2, 2.1]
    tf = tables[1]
    assert [r["metric"] for r in tf["rows"]] == ["auc", "brier"]
    for t in tables:
        assert "target" not in [r["metric"] for r in t["rows"]]
        assert not any("calibration" in r["metric"] for r in t["rows"])
        assert "xflash" in t["title"]


def test_metric_tables_skip_empty_target():
    block = {"xflash_blind": {"target": "x", "lgbm": {}}, "xflash_tf": _real_split(1, 0.7)["xflash_tf"]}
    tables = R._metric_tables({"cv": block})
    assert len(tables) == 1 and "xflash_tf" in tables[0]["title"]


def test_damaged_cards_do_not_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "load_glossary", lambda data: {})
    data = tmp_path / "data"
    _make_artifacts(data)
    no_summary = {k: v for k, v in FINDINGS.items() if k != "match_summary"}
    _write(data / "findings" / "B.json", dict(no_summary, card_id="B", player="Игрок B"))
    f_bad = {k: v for k, v in FINDINGS["findings"][0].items() if k != "alt_lineup"}
    _write(data / "findings" / "C.json", dict(FINDINGS, card_id="C", player="Игрок C", findings=[f_bad]))
    _write(data / "advice" / "A.json", dict(ADVICE, check={"n_numbers": None}, attempts=None))
    html = R.build_report(data, tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert "карточка повреждена: B" in html and "карточка повреждена: C" in html
    assert "карточка повреждена: A" not in html
    assert ADVICE["text"] in html and "Игрок A" in html


HYP_ALL_NONZERO = {
    "min_flashes": 7, "n_pairs": 100,
    "metrics": {
        "blind_minus_x": {"rho": 0.069, "ci_low": 0.051, "ci_high": 0.084, "p": 1e-15, "n": 13767},
        "tf_rate": {"rho": -0.065, "ci_low": -0.081, "ci_high": -0.047, "p": 1e-14, "n": 13767},
    },
}


def test_rating_conclusion_all_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "load_glossary", lambda data: {})
    data = tmp_path / "data"
    _make_artifacts(data)
    _write(data / "rating_hypothesis.json", HYP_ALL_NONZERO)
    html = R.build_report(data, tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert "ранговая связь статистически ненулевая, но очень слабая (max |ρ| = 0.069)" in html
    assert "ρ²" not in html and "почти не объясняет" not in html
    assert "не меньше чем 7 флешками" in html
    assert "не дублирует рейтинг" not in html


def test_rating_conclusion_ci_includes_zero(tmp_path, monkeypatch):
    html = _build(tmp_path, monkeypatch)  # HYP: blind_minus_x включает 0
    assert "Связь не обнаружена (ДИ включает 0) для: факт" in html
    assert "Для остальных метрик ранговая связь статистически ненулевая, но слабая (max |ρ| = 0.120)" in html


def test_rating_share_from_artifact(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(R, "load_glossary", lambda data: {})
    data = tmp_path / "data"
    _make_artifacts(data)
    pd.DataFrame({"rating": [1.0, None, None, None]}).to_parquet(data / "players.parquet")
    html = R.build_report(data, tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert "примерно у 25%" in html
    assert "примерно у 24%" in html  # статичный список LIMITATIONS не меняется
    fallback = _build(tmp_path / "x", monkeypatch)
    assert "Рейтинг есть только примерно у 24% записей, а" in fallback


def test_fixed_claims_text(tmp_path, monkeypatch):
    html = _build(tmp_path, monkeypatch)
    assert "зоны KMeans построены по всем матчам без таргета" in html
    assert "утечки нет" not in html
    assert "тип броска у альтернативы может отличаться" in html


# --- final fix F4 --------------------------------------------------------------------------------
FINDINGS_F4 = json.loads(json.dumps(FINDINGS))
FINDINGS_F4["findings"][0]["own_lineup"]["model_xflash_s"] = 1.1
FINDINGS_F4["findings"][0]["alt_lineup"]["model_xflash_s"] = 1.8
FINDINGS_F4["findings"][0]["alt_lineup"]["throw_dist_u"] = 312
UNVERIFIED = dict(ADVICE, text="Непроверенный текст LLM про 99 с.", verified=False, attempts=2,
                  check={"ok": False, "bad_numbers": [99.0], "bad_zones": [], "n_numbers": 3,
                         "bad_directions": [{"finding": "f1", "metric": "teamflash_pct", "value": 28.0,
                                             "claimed": "up"}],
                         "bad_words": ["треть"], "n_bad_number_occurrences": 1})


def _build_f4(tmp_path, monkeypatch, advice=ADVICE):
    monkeypatch.setattr(R, "load_glossary", lambda data: {5: "A main", 6: "A site"})
    data = tmp_path / "data"
    _make_artifacts(data, advice=advice)
    _write(data / "findings" / "A.json", FINDINGS_F4)
    return R.build_report(data, tmp_path / "out" / "report.html").read_text(encoding="utf-8")


def test_card_table_rows_relabelled(tmp_path, monkeypatch):
    html = _build_f4(tmp_path, monkeypatch)
    assert "Среднее ослепление врагов в лайнапе (train), с" in html
    assert "<td>xFlash, с" not in html
    assert re.search(r"Прогноз модели xFlash, с</td><td class=\"num\">1\.1</td><td class=\"num\">1\.8</td>", html)
    assert re.search(r"Расстояние до точки броска альтернативы, юнитов</td><td>—</td><td class=\"num\">312</td>", html)


def test_card_table_tolerates_old_findings(tmp_path, monkeypatch):
    html = _build(tmp_path, monkeypatch)        # FINDINGS без model_xflash_s и throw_dist_u
    assert "Прогноз модели xFlash, с" in html and "карточка повреждена" not in html


def test_wording_zone_not_position(tmp_path, monkeypatch):
    html = _build_f4(tmp_path, monkeypatch)
    assert "той же позиции" not in html and "из той же зоны броска" in html
    tmpl = (R.TEMPLATES / "report.html.j2").read_text(encoding="utf-8")
    assert "той же позиции" not in tmpl
    assert "медиана — 6 флешек" in html and "в среднем всего 6 флешек" not in html


def test_verification_paragraph_states_exact_rule(tmp_path, monkeypatch):
    html = _build_f4(tmp_path, monkeypatch)
    s7 = html[html.index('<section id="s7">'):html.index('<section id="s8">')]
    assert "округлением до целого" not in s7
    for phrase in ("без учёта знака", "без округления", "зоны этой карточки", "направление каждой",
                   "слова-числа", "Качественные объяснения", "не проверяются", "Что проверка не покрывает"):
        assert phrase in s7, phrase


def test_unverified_card_shows_template_as_main_advice(tmp_path, monkeypatch):
    html = _build_f4(tmp_path, monkeypatch, advice=UNVERIFIED)
    card = html[html.index('<section id="s6">'):html.index('<section id="s7">')]
    note = "ответ LLM не прошёл проверку — показан шаблонный совет"
    assert note in card
    i_tmpl = card.index('<div class="advice">' + UNVERIFIED["template_text"])
    i_details = card.index("<details><summary>непроверенный ответ LLM</summary>")
    i_llm = card.index(UNVERIFIED["text"])
    assert i_tmpl < i_details < i_llm                  # LLM-текст только внутри свёрнутого блока
    assert "<details open" not in card
    assert "направление: f1" in card and "треть" in card


def test_verified_card_shows_llm_text_with_template_collapsed(tmp_path, monkeypatch):
    html = _build_f4(tmp_path, monkeypatch)
    card = html[html.index('<section id="s6">'):html.index('<section id="s7">')]
    assert card.index('<div class="advice">' + ADVICE["text"]) < card.index("<summary>Шаблонный совет (без LLM)")
    assert "не прошёл проверку" not in card


def test_llm_stats_count_every_bad_occurrence():
    adv = dict(ADVICE, verified=False, check={"ok": False, "bad_numbers": [99.0], "bad_zones": [],
                                              "bad_directions": [], "bad_words": [], "n_numbers": 10,
                                              "n_bad_number_occurrences": 5})
    stats = R._llm_stats([{"advice": adv}])
    assert (stats["numbers_total"], stats["numbers_ok"], stats["numbers_ok_pct"]) == (10, 5, 50)
    legacy = dict(ADVICE, check={"bad_numbers": [99.0, 7.0], "n_numbers": 4})   # старый формат check
    assert R._llm_stats([{"advice": legacy}])["numbers_ok"] == 2


def test_llm_stats_count_directions_and_words(tmp_path, monkeypatch):
    stats = R._llm_stats([{"advice": UNVERIFIED}])
    assert stats["directions_bad"] == 1 and stats["words_bad"] == 1
    html = _build_f4(tmp_path, monkeypatch, advice=UNVERIFIED)
    assert "Разниц с неверным направлением" in html and "Слов-чисел" in html


def test_new_limitations_present(tmp_path, monkeypatch):
    html = _build(tmp_path, monkeypatch)
    for text in (
        "Сравнение лайнапов наблюдательное: разница смешивает позицию броска, место приземления, тип броска и ситуацию в раунде; это не причинный эффект смены броска.",
        "Альтернатива выбирается в той же зоне броска, но точка броска может быть в сотнях юнитов от точки игрока; расстояние указано в карточке.",
        "Средние лайнапов у выбранной альтернативы завышены эффектом отбора лучшего из многих (winner's curse); прогноз модели xFlash показан рядом для сравнения.",
        "Стандартная ошибка лайнапа приближает кластеризацию по матчам через число матчей (n_eff); ковариация ослепления и тимфлеша не учитывается.",
        "HE и молотовы в этой версии не анализируются: урон по ним собран (damage_events.parquet), но агрегаты по лайнапам отложены.",
    ):
        assert text in R.LIMITATIONS
        assert str(escape(text)) in html
