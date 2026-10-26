import copy

from utility_coach import faithfulness as FA

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
NAMES = {"A main", "A site", "Mid", "B main", "Зона 12"}

CORRECT = (
    "Игрок A, в раунде 7 твоя флешка из A main в A site дала 0,4 с ослепления и тимфлеш.\n"
    "1. Этот лайнап в среднем даёт 0.9 с и 34% тимфлешей.\n"
    "2. Брось стоя в A site: yaw 22.1, pitch -4.9. Так выходит 2.1 с (ДИ от 1,9 с) и 6% тимфлешей, "
    "на 28 п.п. меньше.\n"
)


def test_extract_numbers_formats():
    assert FA.extract_numbers("было 1,9 с, стало 2.1 с, тимфлешей 34%, разница -28") == [1.9, 2.1, 34.0, -28.0]


def test_extract_numbers_ignores_list_markers():
    text = "12. Первый совет\n13) Второй совет\n- 14. Третий\n**15.** Четвёртый"
    assert FA.extract_numbers(text) == []
    assert FA.extract_numbers("в раунде 12.") == [12.0]


def test_extract_numbers_range_and_unicode_minus():
    assert FA.extract_numbers("ДИ 1.9–2.3, pitch −4.9") == [1.9, 2.3, -4.9]


def test_allowed_numbers_contains_abs_values_without_rounding():
    allowed = FA.allowed_numbers(FINDINGS)
    for v in (0.9, 34.0, 812.0, 28.0, 2.1, 4.9, 14200.0, 1204.0, 647.0):
        assert v in allowed
    assert -28.0 not in allowed   # sign is ignored: only abs() values are stored
    assert 2.0 not in allowed     # no rounding of 2.1
    assert 99.0 not in allowed
    assert 3.7 not in allowed


def test_check_rejects_rounded_integer_of_fraction():
    assert FA.check("5 с", {"a": 4.9}, set())["bad_numbers"] == [5.0]


def test_check_rejects_integer_when_only_fractions_exist():
    res = FA.check("1 с", {"a": 0.6, "b": 1.2}, set())
    assert res["bad_numbers"] == [1.0]


def test_check_sign_insensitive_minus_passes():
    assert FA.check("откат -1.2 с", {"a": 1.2}, set())["ok"] is True


def test_check_decimal_comma_passes():
    assert FA.check("4,9 с", {"a": 4.9}, set())["ok"] is True


def test_check_connector_not_matched_by_longer_zone():
    f = copy.deepcopy(FINDINGS)
    f["findings"][0]["zone_from"] = "Z connector"
    res = FA.check("Из Connector", f, NAMES)
    assert res["bad_zones"] == ["Connector"]
    assert res["ok"] is False


def test_russian_alias_midl_is_mid():
    res = FA.check("кидай из мидла", FINDINGS, NAMES)
    assert res["bad_zones"] == ["Mid"]
    assert res["ok"] is False


def test_russian_aliases_for_labeled_zones():
    cases = {
        "с мидл": "Mid", "а мейн": "A main", "а-мейн": "A main", "а main": "A main",
        "б мейн": "B main", "б-мейн": "B main", "б main": "B main",
        "плент а": "A site", "плент a": "A site", "а плент": "A site", "а сайт": "A site",
        "плент б": "B site", "плент b": "B site", "б плент": "B site", "б сайт": "B site",
        "т спавн": "T spawn", "т-спавн": "T spawn",
    }
    for text, canon in cases.items():
        found = [c for _, _, c in FA.find_zone_mentions(text)]
        assert found == [canon], (text, found)


def test_check_passes_correct_text():
    res = FA.check(CORRECT, FINDINGS, NAMES)
    assert res == {"ok": True, "bad_numbers": [], "bad_zones": [], "bad_directions": [], "bad_words": [],
                   "n_numbers": 10, "n_bad_number_occurrences": 0}


def test_check_catches_invented_number():
    res = FA.check("Этот лайнап даёт 0.9 с, а альтернатива 3.7 с.", FINDINGS, NAMES)
    assert res["ok"] is False
    assert res["bad_numbers"] == [3.7]


def test_check_catches_invented_zone():
    res = FA.check("Попробуй флешку из Mid в A site.", FINDINGS, NAMES)
    assert res["ok"] is False
    assert res["bad_zones"] == ["Mid"]


def test_check_catches_russian_alias():
    res = FA.check("Лучше кинь с мида.", FINDINGS, NAMES)
    assert res["bad_zones"] == ["Mid"]


def test_check_negative_diff_modulus():
    text = "Альтернатива даёт на 28 п.п. меньше тимфлешей и на 1.2 с больше ослепления."
    assert FA.check(text, FINDINGS, NAMES)["ok"] is True


def test_check_glossary_zone_and_its_digits_are_masked():
    res = FA.check("Брось в Зона 12.", FINDINGS, NAMES)
    assert res["bad_zones"] == ["Зона 12"]
    assert res["bad_numbers"] == []


def test_check_longest_callout_wins():
    f2 = copy.deepcopy(FINDINGS)
    f2["findings"][0]["alt_lineup"]["zone_to"] = "Z connector"
    res = FA.check("Брось через Z connector в A site.", f2, NAMES)
    assert res["ok"] is True, res


def test_check_composite_glossary_name():
    f3 = copy.deepcopy(FINDINGS)
    f3["findings"][0]["zone_from"] = "A long/Highway"
    res = FA.check("Из Highway в A site.", f3, NAMES | {"A long/Highway"})
    assert res["ok"] is True, res


def test_callouts_cover_cache():
    for name in ("A main", "A site", "A long", "Highway", "Quad", "Forklift", "Truck", "Squeaky", "Toxic",
                 "Boost", "Heaven", "Mid", "Vents", "Checkers", "Z-hall", "Z connector", "Garage",
                 "White box", "Sunroom", "B main", "B halls", "B site", "Headshot", "CT spawn", "T spawn",
                 "Connector"):
        assert name in FA.CACHE_CALLOUTS


# --- final fix F1: direction of differences, number words, occurrence counts ----------------------
CARD_D = {
    "card_id": "D", "player": "Игрок D", "rating": None,
    "match_summary": {"n_flashes": 11, "enemy_blind_s": 35.0, "teamflashes": 3, "flash_kills": 1,
                      "enemy_blind_pct": 92},
    "findings": [{
        "id": "f1", "round": 12, "side": "T", "zone_from": "Зона 19", "zone_to": "Зона 9", "mode": "stand",
        "times_in_match": 1,
        "actual": {"enemy_blind_s": 0.0, "teamflash": False},
        "own_lineup": {"xflash_s": 0.4, "teamflash_pct": 3, "n": 34, "ci95_blind_s": [0.0, 0.9],
                       "ci95_teamflash_pct": [0, 15]},
        "alt_lineup": {"zone_to": "Зона 3", "mode": "stand", "xflash_s": 3.5, "teamflash_pct": 14, "n": 288,
                       "ci95_blind_s": [3.2, 3.9], "ci95_teamflash_pct": [9, 18], "yaw": 152.6,
                       "pitch": -30.1, "throw_pos": [643, -1450, 1613]},
        "diff": {"blind_s": 3.1, "teamflash_pct": 11},
    }],
}
NAMES_D = {"Зона 19", "Зона 9", "Зона 3"}
# дословно второй пункт реального ответа LLM по карточке D (data/advice/D.json до исправления)
REAL_D_SENTENCE = (
    "2. По тимфлешам та же замена безопаснее: у твоего лайнапа teamflash_pct 3 при интервале [0, 15], "
    "у альтернативы в Зона 3 — 14 при интервале [9, 18], то есть на 11 меньше риска ослепить своего."
)


def test_direction_real_card_d_safer_claim_is_rejected():
    res = FA.check(REAL_D_SENTENCE, CARD_D, NAMES_D)
    assert res["bad_numbers"] == [] and res["bad_zones"] == []
    assert res["ok"] is False
    assert res["bad_directions"] == [{"finding": "f1", "metric": "teamflash_pct", "value": 11.0, "claimed": "down"}]


def test_direction_short_card_d_pattern_is_rejected():
    res = FA.check("Та же замена безопаснее… на 11 меньше риска ослепить своего.", CARD_D, NAMES_D)
    assert res["ok"] is False
    assert res["bad_directions"][0]["metric"] == "teamflash_pct"


def test_direction_blind_up_matches_positive_diff():
    res = FA.check("Альтернатива даёт на 3.1 с больше ослепления врагов.", CARD_D, NAMES_D)
    assert res["ok"] is True, res
    assert res["bad_directions"] == []


def test_direction_blind_reversed_is_rejected():
    res = FA.check("Альтернатива даёт на 3.1 с меньше ослепления врагов.", CARD_D, NAMES_D)
    assert res["ok"] is False
    assert res["bad_directions"] == [{"finding": "f1", "metric": "blind_s", "value": 3.1, "claimed": "down"}]


def test_direction_teamflash_units_and_words():
    ok = FA.check("Зато у альтернативы на 11 п.п. больше тимфлешей.", CARD_D, NAMES_D)
    assert ok["ok"] is True, ok
    bad = FA.check("У альтернативы на 11 п.п. меньше тимфлешей.", CARD_D, NAMES_D)
    assert bad["bad_directions"] == [{"finding": "f1", "metric": "teamflash_pct", "value": 11.0,
                                      "claimed": "down"}]
    pct = FA.check("Своих она слепит на 11% реже.", CARD_D, NAMES_D)
    assert pct["ok"] is False
    # без слова о своих единица % не определяет метрику -> вердикта нет
    assert FA.check("Это на 11% реже.", CARD_D, NAMES_D)["bad_directions"] == []


def test_direction_negative_diff_down_passes_and_up_fails():
    good = FA.check("Альтернатива даёт на 28 п.п. меньше тимфлешей.", FINDINGS, NAMES)
    assert good["ok"] is True, good
    bad = FA.check("Альтернатива даёт на 28 п.п. больше тимфлешей.", FINDINGS, NAMES)
    assert bad["ok"] is False
    assert bad["bad_directions"][0]["claimed"] == "up"


def test_direction_skips_when_unclear():
    # нет единицы, обе стороны сравнения, нет сравнительных слов -> вердикта нет
    for text in ("На 3.1 больше.", "На 3.1 с больше, но выйти чуть ниже.", "Разница 3.1 с."):
        assert FA.check(text, CARD_D, NAMES_D)["bad_directions"] == [], text


def test_direction_sentence_split_keeps_decimals_and_pp():
    text = "Ослепление 3.5 с.\nНа 3.1 с больше. Тимфлеши: на 11 п.п. больше тимфлешей!"
    res = FA.check(text, CARD_D, NAMES_D)
    assert res["ok"] is True, res


def test_direction_safety_word_ok_when_teamflash_drops():
    res = FA.check("Альтернатива безопаснее: на 28 п.п. меньше тимфлешей.", FINDINGS, NAMES)
    assert res["ok"] is True, res


def test_number_words_rejected():
    res = FA.check("Альтернатива ослепляет своих в два раза реже.", CARD_D, NAMES_D)
    assert res["ok"] is False
    assert res["bad_words"] == ["в два раза"]
    for text, word in (("треть бросков", "треть"), ("половина флешек", "половина"), ("вдвое дольше", "вдвое"),
                       ("в три раза чаще", "в три раза"), ("четверть", "четверть"), ("втрое", "втрое")):
        assert FA.check(text, CARD_D, NAMES_D)["bad_words"] == [word], text


def test_number_words_do_not_catch_ordinals():
    assert FA.check("в третьем раунде и в четвертой попытке", CARD_D, NAMES_D)["bad_words"] == []


def test_bad_number_occurrences_counted_per_occurrence():
    text = "Получишь 99 с. Потом 99 с. И снова 99 с, 99 с и 99 с."
    res = FA.check(text, CARD_D, NAMES_D)
    assert res["bad_numbers"] == [99.0]
    assert res["n_numbers"] == 5
    assert res["n_bad_number_occurrences"] == 5
