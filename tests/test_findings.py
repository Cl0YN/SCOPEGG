import json

import pandas as pd

from utility_coach import findings as F
from utility_coach.config import TF_PENALTY_S

P = 76561198011111111   # игрок, для которого есть карточка
Q = 76561198022222222   # в test-матче, но без лайнапов
S = 76561198033333333   # в test-матче, значимая находка, но мало флешек
R = 76561198044444444   # значимая находка, но train-матч
M_TEST = 900000001
M_TRAIN = 900000002
GLOSSARY = {5: "A main", 6: "A site", 7: "Mid", 8: "B main", 9: "Quad"}


def _lineup(lid, zone_from, mode, n, blind, blind_se, tf, tf_se):
    return {
        "lineup_id": lid, "team": 2, "zone_from": zone_from, "zone_to": 6, "mode": mode, "n": n,
        "xflash_mean": blind, "blind_mean": blind, "blind_se": blind_se,
        "tf_rate": tf, "tf_se": tf_se, "blind_eb": blind, "tf_eb": tf,
        "score": blind - TF_PENALTY_S * tf,
        "throw_x": -647.4, "throw_y": 1203.6, "throw_z": 1688.2, "yaw": 22.14, "pitch": -4.93,
        "to_x": 0.0, "to_y": 0.0, "to_z": 0.0,
    }


def make_lineups() -> pd.DataFrame:
    return pd.DataFrame([
        # зона 5: значимая находка, выгода 1.98 - 0.22 = 1.76
        _lineup("2-5-jump-0", 5, "jump", 812, 0.9, 0.05, 0.34, 0.015),
        _lineup("2-5-stand-0", 5, "stand", 420, 2.1, 0.10, 0.06, 0.01),
        # зона 7: свой лайнап с малым n -> находка отбрасывается
        _lineup("2-7-jump-0", 7, "jump", 12, 0.5, 0.05, 0.30, 0.02),
        _lineup("2-7-stand-0", 7, "stand", 500, 2.0, 0.05, 0.05, 0.01),
        # зона 8: альтернатива лучше по score, но ДИ разницы включает 0 -> незначимо
        _lineup("2-8-jump-0", 8, "jump", 300, 1.0, 0.50, 0.10, 0.10),
        _lineup("2-8-stand-0", 8, "stand", 300, 1.1, 0.50, 0.09, 0.10),
        # зона 9: значимая находка, выгода 1.9 - 1.4 = 0.5
        _lineup("2-9-crouch-0", 9, "crouch", 100, 1.5, 0.05, 0.05, 0.01),
        _lineup("2-9-stand-0", 9, "stand", 200, 2.0, 0.05, 0.05, 0.01),
    ])


def make_data():
    flashes, links = [], []

    def add(uid, match_id, thrower, rnd, zone_from, blind, lineup_id, mode, tf=False, fk=False, is_test=True):
        flashes.append({
            "throw_uid": uid, "match_id": match_id, "thrower_id": thrower, "team": 2, "round": rnd,
            "zone_from": zone_from, "zone_to": 6, "enemy_blind_s": blind, "teamflash": tf,
            "flash_kill": fk, "xflash_blind": 1.0, "xflash_tf": 0.1, "is_test": is_test,
        })
        links.append({"throw_uid": uid, "lineup_id": lineup_id, "mode": mode})

    # round во входе 0-based (как RoundIndexMatch из ingest); в findings выводится round + 1
    add(0, M_TEST, P, 6, 5, 0.4, "2-5-jump-0", "jump", tf=True)
    add(1, M_TEST, P, 8, 7, 1.0, "2-7-jump-0", "jump")
    add(2, M_TEST, P, 10, 8, 0.5, "2-8-jump-0", "jump")
    add(3, M_TEST, P, 2, 9, 1.2, "2-9-crouch-0", "crouch", fk=True)
    add(4, M_TEST, P, 14, 5, 0.3, "2-5-jump-0", "jump", tf=True)
    add(5, M_TEST, P, 15, 5, 0.0, None, "stand")
    add(6, M_TEST, P, 17, 5, 0.0, None, "stand")
    for i in range(7):
        add(10 + i, M_TEST, Q, i, 5, 1.0, None, "stand")
    for i in range(3):
        add(20 + i, M_TEST, S, i, 5, 0.2, "2-5-jump-0", "jump")
    for i in range(7):
        add(30 + i, M_TRAIN, R, i, 5, 0.2, "2-5-jump-0", "jump", is_test=False)

    scored = pd.DataFrame(flashes)
    flash_lineups = pd.DataFrame(links)
    player_match = pd.DataFrame({
        "match_id": [M_TEST] * 6,
        "player_id": [1, 2, 3, 4, 5, P],
        "rating": [float("nan")] * 6,
        "n_flashes": [6] * 6,
        "enemy_blind_s_sum": [0.5, 1.0, 2.0, 3.0, 10.0, 3.4],
        "blind_minus_x": [0.0] * 6,
        "tf_rate": [0.0] * 6,
        "flash_kill_rate": [0.0] * 6,
    })
    players = pd.DataFrame({
        "match_id": [M_TEST, M_TEST],
        "player_id": [P, Q],
        "rating": [15432.0, float("nan")],
    })
    return scored, flash_lineups, make_lineups(), player_match, players


def _build(**kw):
    scored, fl, lu, pm, pl = make_data()
    return F.build_findings(M_TEST, P, "Игрок A", scored, fl, lu, pm, pl, GLOSSARY, **kw)


def _numbers(node):
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _numbers(v)
    elif isinstance(node, list):
        for v in node:
            yield from _numbers(v)


def test_side_mapping():
    assert F._side(2) == "T"
    assert F._side(3) == "CT"


def test_select_cards_only_test_matches_with_significant_finding():
    scored, fl, lu, _, _ = make_data()
    assert F.select_cards(scored, fl, lu) == [(M_TEST, P)]


def test_select_cards_respects_n_cards_and_is_deterministic():
    scored, fl, lu, _, _ = make_data()
    assert F.select_cards(scored, fl, lu, n_cards=0) == []
    assert F.select_cards(scored, fl, lu, seed=1) == F.select_cards(scored, fl, lu, seed=1)


def test_build_findings_schema_and_values():
    f = _build()
    assert list(f) == ["card_id", "player", "rating", "match_summary", "findings"]
    assert f["card_id"] == "A"
    assert f["player"] == "Игрок A"
    assert f["rating"] == 15432
    assert f["match_summary"] == {
        "n_flashes": 7, "enemy_blind_s": 3.4, "teamflashes": 2, "flash_kills": 1, "enemy_blind_pct": 83,
    }
    assert len(f["findings"]) == 2
    assert f["findings"][0] == {
        "id": "f1", "round": 7, "side": "T", "zone_from": "A main", "zone_to": "A site", "mode": "jump",
        "times_in_match": 2,
        "actual": {"enemy_blind_s": 0.4, "teamflash": True},
        "own_lineup": {"xflash_s": 0.9, "model_xflash_s": 0.9, "teamflash_pct": 34, "n": 812,
                       "ci95_blind_s": [0.8, 1.0], "ci95_teamflash_pct": [31, 37]},
        "alt_lineup": {"zone_to": "A site", "mode": "stand", "xflash_s": 2.1, "model_xflash_s": 2.1,
                       "teamflash_pct": 6, "n": 420, "ci95_blind_s": [1.9, 2.3], "ci95_teamflash_pct": [4, 8],
                       "yaw": 22.1, "pitch": -4.9, "throw_pos": [-647, 1204, 1688], "throw_dist_u": 0},
        "diff": {"blind_s": 1.2, "teamflash_pct": -28},
    }
    f2 = f["findings"][1]
    assert (f2["id"], f2["round"], f2["zone_from"], f2["mode"]) == ("f2", 3, "Quad", "crouch")
    assert f2["own_lineup"]["ci95_blind_s"] == [1.4, 1.6]
    assert f2["diff"] == {"blind_s": 0.5, "teamflash_pct": 0}


def test_filter_drops_small_n_and_insignificant():
    rounds = [x["round"] for x in _build()["findings"]]
    assert 9 not in rounds    # свой лайнап n=12 < MIN_LINEUP_N
    assert 11 not in rounds   # ДИ разницы включает 0
    assert rounds == [7, 3]


def test_filter_respects_compare_significance(monkeypatch):
    def never_significant(own, alt):
        return {"blind_diff": 1.0, "blind_ci": [-0.1, 2.0], "tf_diff": 0.0, "tf_ci": [-0.1, 0.1],
                "significant": False}

    monkeypatch.setattr(F, "compare", never_significant)
    assert _build()["findings"] == []
    scored, fl, lu, _, _ = make_data()
    assert F.select_cards(scored, fl, lu) == []


def test_max_findings_keeps_biggest_gain():
    f = _build(max_findings=1)
    assert [x["round"] for x in f["findings"]] == [7]


def test_anonymization_no_real_ids():
    f = _build()
    text = json.dumps(f, ensure_ascii=False)
    for real in (P, M_TEST):
        assert str(real) not in text
    nums = list(_numbers(f))
    assert nums
    assert all(v not in (P, M_TEST, Q, S, R, M_TRAIN) for v in nums)
    assert "player_id" not in text and "match_id" not in text and "throw_uid" not in text


def test_number_formatting():
    f = _build()
    s = f["match_summary"]
    assert isinstance(s["enemy_blind_pct"], int)
    assert round(s["enemy_blind_s"], 1) == s["enemy_blind_s"]
    for x in f["findings"]:
        assert x["side"] in {"T", "CT"}
        for block in (x["own_lineup"], x["alt_lineup"]):
            assert isinstance(block["teamflash_pct"], int)
            assert all(isinstance(v, int) for v in block["ci95_teamflash_pct"])
            assert isinstance(block["n"], int)
            assert round(block["xflash_s"], 1) == block["xflash_s"]
            assert all(round(v, 1) == v for v in block["ci95_blind_s"])
        alt = x["alt_lineup"]
        assert all(isinstance(v, int) for v in alt["throw_pos"])
        assert round(alt["yaw"], 1) == alt["yaw"] and round(alt["pitch"], 1) == alt["pitch"]
        assert isinstance(x["diff"]["teamflash_pct"], int)
        assert round(x["diff"]["blind_s"], 1) == x["diff"]["blind_s"]


def test_run_findings_writes_cards(tmp_path, monkeypatch):
    scored, fl, lu, pm, pl = make_data()
    scored.to_parquet(tmp_path / "flash_scored.parquet")
    fl.to_parquet(tmp_path / "flash_lineups.parquet")
    lu.to_parquet(tmp_path / "lineups.parquet")
    pm.to_parquet(tmp_path / "player_match.parquet")
    pl.to_parquet(tmp_path / "players.parquet")
    monkeypatch.setattr(F, "load_glossary", lambda data: GLOSSARY)
    stale = tmp_path / "findings" / "Z.json"
    stale.parent.mkdir()
    stale.write_text("{}", encoding="utf-8")

    assert F.run_findings(tmp_path) == ["A"]
    assert not stale.exists()
    text = (tmp_path / "findings" / "A.json").read_text(encoding="utf-8")
    card = json.loads(text)
    assert card["player"] == "Игрок A"
    assert str(P) not in text and str(M_TEST) not in text


def test_lineup_block_shows_raw_values_not_eb():
    row = pd.Series({"blind_mean": 2.0, "blind_eb": 1.0, "blind_se": 0.1, "xflash_mean": 1.73,
                     "tf_rate": 0.20, "tf_eb": 0.10, "tf_se": 0.05, "n": 50})
    b = F._lineup_block(row)
    assert b["xflash_s"] == 2.0                      # raw, not blind_eb=1.0
    assert b["model_xflash_s"] == 1.7                # model mean prediction, 1 decimal
    assert b["teamflash_pct"] == 20                  # raw, not tf_eb=10
    assert b["ci95_blind_s"] == [1.8, 2.2]           # 2.0 +- 1.96*0.1
    assert b["ci95_teamflash_pct"] == [10, 30]       # 100*(0.20 +- 1.96*0.05)
    assert b["n"] == 50


def test_lineup_block_clips_ci():
    row = pd.Series({"blind_mean": 0.1, "blind_eb": 0.5, "blind_se": 0.5, "xflash_mean": 0.2,
                     "tf_rate": 0.9, "tf_eb": 0.5, "tf_se": 0.2, "n": 40})
    b = F._lineup_block(row)
    assert b["ci95_blind_s"][0] == 0.0               # 0.1 - 0.98 < 0 -> 0
    assert b["ci95_teamflash_pct"][1] == 100         # 0.9 + 0.392 > 1 -> 100


# --- final fix F3 --------------------------------------------------------------------------------
def test_findings_have_model_xflash_and_throw_distance():
    scored, fl, lu, pm, pl = make_data()
    lu.loc[lu["lineup_id"] == "2-5-jump-0", "xflash_mean"] = 1.04
    lu.loc[lu["lineup_id"] == "2-5-stand-0", "xflash_mean"] = 1.86
    alt = lu["lineup_id"] == "2-5-stand-0"
    lu.loc[alt, "throw_x"] = lu.loc[alt, "throw_x"] + 300.0
    lu.loc[alt, "throw_y"] = lu.loc[alt, "throw_y"] - 400.4
    lu.loc[alt, "throw_z"] = lu.loc[alt, "throw_z"] + 999.0     # высота в расстояние XY не входит
    f = F.build_findings(M_TEST, P, "Игрок A", scored, fl, lu, pm, pl, GLOSSARY)["findings"][0]
    assert f["own_lineup"]["model_xflash_s"] == 1.0
    assert f["alt_lineup"]["model_xflash_s"] == 1.9
    assert f["own_lineup"]["xflash_s"] == 0.9 and f["alt_lineup"]["xflash_s"] == 2.1   # raw train mean
    assert f["alt_lineup"]["throw_dist_u"] == 500
    assert isinstance(f["alt_lineup"]["throw_dist_u"], int)
    assert "throw_dist_u" not in f["own_lineup"]
