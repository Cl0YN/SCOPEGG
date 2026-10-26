import numpy as np
import pandas as pd
import pytest

from utility_coach.features import CATEGORICAL, FEATURES, flash_targets, round_start_ticks

THROW_DEFAULTS = dict(match_id=1, grenade_type="flash", round=1, thrower_id=10, team=2,
                      start_tick=1000, end_tick=1100, throw_x=0.0, throw_y=0.0, throw_z=0.0,
                      yaw=0.0, pitch=0.0, vel_x=0.0, vel_y=0.0, vel_z=0.0,
                      to_x=100.0, to_y=0.0, to_z=0.0, crouch=False, airborne=False, inferno_ok=True)


def make_throws(rows: list[dict]) -> pd.DataFrame:
    """throws.parquet-подобный фрейм: throw_uid = номер строки, остальное по умолчанию."""
    recs = []
    for i, r in enumerate(rows):
        d = dict(THROW_DEFAULTS, throw_uid=i, projectile_id=i + 1)
        d.update(r)
        recs.append(d)
    df = pd.DataFrame(recs)
    dtypes = {"throw_uid": "int64", "match_id": "int64", "projectile_id": "int64", "round": "int16",
              "thrower_id": "int64", "team": "int8", "start_tick": "int32", "end_tick": "int32",
              "crouch": bool, "airborne": bool, "inferno_ok": bool}
    for c in ("throw_x", "throw_y", "throw_z", "yaw", "pitch", "vel_x", "vel_y", "vel_z", "to_x", "to_y", "to_z"):
        dtypes[c] = "float32"
    return df.astype(dtypes)


def make_victims(rows: list[tuple]) -> pd.DataFrame:
    """rows: (throw_uid, victim_id, victim_team, flash_duration, killed)."""
    df = pd.DataFrame(rows, columns=["throw_uid", "victim_id", "victim_team", "flash_duration", "killed"])
    df["match_id"] = 1
    df["killed_by"] = np.where(df["killed"], 99, -1)
    df["killed_tick"] = np.where(df["killed"], 1200, -1)
    return df.astype({"throw_uid": "int64", "match_id": "int64", "victim_id": "int64", "victim_team": "int8",
                      "flash_duration": "float32", "killed": bool, "killed_by": "int64", "killed_tick": "int32"})


@pytest.fixture
def targets() -> pd.DataFrame:
    # бросающий 10 (T=2), тиммейт 11 (T), враги 20, 21 (CT=3); uid 7 бросает CT-игрок 30
    throws = make_throws([{}, {}, {}, {}, {}, {}, {}, {"thrower_id": 30, "team": 3}])
    victims = make_victims([
        (0, 20, 3, 2.0, False), (0, 21, 3, 0.5, False), (0, 11, 2, 0.8, False), (0, 10, 2, 1.5, False),
        (1, 11, 2, 1.2, False), (1, 10, 2, 3.0, False),
        (2, 10, 2, 4.0, False),
        (3, 20, 3, 1.5, True),
        (4, 11, 2, 2.0, True),
        # uid 5 — без жертв
        (6, 11, 2, 1.0, False),
        (7, 10, 2, 2.0, False), (7, 30, 3, 0.3, False),
    ])
    return flash_targets(throws, victims)


def test_round_start_ticks_min_over_all_grenades():
    throws = make_throws([
        {"grenade_type": "smoke", "round": 1, "start_tick": 900},
        {"grenade_type": "flash", "round": 1, "start_tick": 1000},
        {"grenade_type": "flash", "round": 2, "start_tick": 5000},
        {"grenade_type": "he", "match_id": 2, "round": 1, "start_tick": 300},
    ])
    rs = round_start_ticks(throws)
    assert list(rs.columns) == ["match_id", "round", "round_start_tick"]
    got = {(int(r.match_id), int(r.round)): int(r.round_start_tick) for r in rs.itertuples()}
    assert got == {(1, 1): 900, (1, 2): 5000, (2, 1): 300}


def test_enemy_blind_sums_only_enemies(targets):
    assert targets.loc[0, "enemy_blind_s"] == pytest.approx(2.5)   # 2.0 + 0.5, без тиммейта и себя
    assert targets.loc[1, "enemy_blind_s"] == 0.0
    assert targets.loc[7, "enemy_blind_s"] == pytest.approx(2.0)   # для CT врагом считается T


def test_teamflash_ignores_thrower_and_short_blinds(targets):
    assert not targets.loc[0, "teamflash"]   # тиммейт 0.8 с <= 1 с, сам бросающий не в счёт
    assert targets.loc[1, "teamflash"]       # тиммейт 1.2 с
    assert not targets.loc[2, "teamflash"]   # ослеплён только сам бросающий
    assert not targets.loc[6, "teamflash"]   # ровно 1.0 с — ещё не тимфлеш
    assert targets.loc[4, "teamflash"]


def test_self_flash(targets):
    assert targets.loc[0, "self_flash"]
    assert targets.loc[2, "self_flash"]
    assert not targets.loc[3, "self_flash"]
    assert targets.loc[7, "self_flash"]       # бросающий 30 ослепил себя на 0.3 с (> 0)


def test_flash_kill_only_enemy(targets):
    assert targets.loc[3, "flash_kill"]
    assert not targets.loc[4, "flash_kill"]   # убит тиммейт — не флеш-фраг
    assert not targets.loc[0, "flash_kill"]


def test_n_enemies_blinded_counts_over_threshold(targets):
    assert targets.loc[0, "n_enemies_blinded"] == 1   # 2.0 > 1, 0.5 нет
    assert targets.loc[3, "n_enemies_blinded"] == 1
    assert targets.loc[1, "n_enemies_blinded"] == 0


def test_flash_without_victims_is_zero(targets):
    row = targets.loc[5]
    assert row["enemy_blind_s"] == 0.0 and row["n_enemies_blinded"] == 0
    assert not row["teamflash"] and not row["self_flash"] and not row["flash_kill"]


def test_targets_index_and_dtypes(targets):
    assert targets.index.name == "throw_uid"
    assert list(targets.index) == list(range(8))
    assert targets["enemy_blind_s"].dtype == np.float32
    assert targets["n_enemies_blinded"].dtype == np.int16
    for c in ("teamflash", "self_flash", "flash_kill"):
        assert targets[c].dtype == bool


import json  # noqa: E402

from utility_coach.features import TARGETS, flash_features, run_features  # noqa: E402

CENTERS = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]])


@pytest.fixture
def feats() -> pd.DataFrame:
    throws = make_throws([
        # uid 0: старт раунда 1000, бросок на 1064, полёт 128 тиков; to (600,800) ближе к центру 2
        {"start_tick": 1064, "end_tick": 1192, "vel_x": 3.0, "vel_y": 4.0, "vel_z": 50.0,
         "to_x": 600.0, "to_y": 800.0, "crouch": True},
        # uid 1: раунд 7 без строки в round_starts -> t_in_round_s = 0; бросок из зоны 1
        {"round": 7, "throw_x": 990.0, "throw_y": 10.0, "to_x": 20.0, "to_y": 10.0, "airborne": True},
    ])
    round_starts = pd.DataFrame({"match_id": [1], "round": [1], "round_start_tick": [1000]})
    return flash_features(throws, round_starts, CENTERS)


def test_feature_columns_and_order(feats):
    assert list(feats.columns) == ["throw_uid", "match_id", "thrower_id"] + FEATURES
    assert list(feats["throw_uid"]) == [0, 1]


def test_ticks_converted_with_tickrate_64(feats):
    assert feats.loc[0, "flight_s"] == pytest.approx(2.0)       # 128 / 64
    assert feats.loc[0, "t_in_round_s"] == pytest.approx(1.0)   # (1064 - 1000) / 64
    assert feats.loc[1, "t_in_round_s"] == 0.0


def test_speed_is_horizontal_and_dist_is_3d(feats):
    assert feats.loc[0, "speed"] == pytest.approx(5.0)          # hypot(3, 4), vel_z не учитывается
    assert feats.loc[0, "dist"] == pytest.approx(1000.0)        # (0,0,0) -> (600,800,0)


def test_zones_and_flags(feats):
    assert int(feats.loc[0, "zone_from"]) == 0 and int(feats.loc[0, "zone_to"]) == 2
    assert int(feats.loc[1, "zone_from"]) == 1 and int(feats.loc[1, "zone_to"]) == 0
    assert bool(feats.loc[0, "crouch"]) and not bool(feats.loc[0, "airborne"])
    assert bool(feats.loc[1, "airborne"])


def test_categorical_dtypes(feats):
    for c in CATEGORICAL:
        assert isinstance(feats[c].dtype, pd.CategoricalDtype), c
    assert list(feats["team"].cat.categories) == [2, 3]
    assert list(feats["zone_from"].cat.categories) == [0, 1, 2]
    assert feats["speed"].dtype == np.float32 and feats["round"].dtype == np.int16


def test_run_features_end_to_end(tmp_path):
    throws = make_throws([
        {"grenade_type": "smoke", "start_tick": 900},
        {"grenade_type": "flash", "start_tick": 964, "end_tick": 1028},
        {"grenade_type": "flash", "start_tick": 1100, "end_tick": 1164, "thrower_id": 20, "team": 3},
        {"grenade_type": "he", "start_tick": 1200},
    ])
    throws.to_parquet(tmp_path / "throws.parquet", index=False)
    make_victims([(1, 20, 3, 2.5, True), (1, 11, 2, 1.5, False), (2, 10, 2, 0.7, False)]).to_parquet(
        tmp_path / "flash_victims.parquet", index=False)
    (tmp_path / "zones.json").write_text(json.dumps({"centers": CENTERS.tolist(), "z_scale": 3.0}), encoding="utf-8")

    run_features(tmp_path)

    ff = pd.read_parquet(tmp_path / "flash_features.parquet")
    assert list(ff["throw_uid"]) == [1, 2]                       # только флешки
    assert set(FEATURES + TARGETS) <= set(ff.columns)
    r1 = ff.set_index("throw_uid").loc[1]
    assert r1["enemy_blind_s"] == pytest.approx(2.5)
    assert bool(r1["teamflash"]) and bool(r1["flash_kill"])
    assert r1["t_in_round_s"] == pytest.approx(1.0)              # (964 - 900) / 64: старт раунда по смоку
    r2 = ff.set_index("throw_uid").loc[2]
    assert r2["enemy_blind_s"] == pytest.approx(0.7) and not bool(r2["teamflash"])
