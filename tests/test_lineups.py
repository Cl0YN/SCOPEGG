import numpy as np
import pandas as pd
import pytest

from utility_coach.lineups import (LINEUP_COLUMNS, assign_lineups, best_alternative, build_lineups, compare,
                                   eb_shrink, pooled_eb, run_lineups, throw_mode)


def test_throw_mode_priority():
    crouch = pd.Series([True, True, True, False, True, False])
    airborne = pd.Series([True, False, False, False, False, False])
    speed = pd.Series([200.0, 150.0, 50.0, 0.0, 100.0, 99.9])
    got = throw_mode(crouch, airborne, speed)
    assert got.tolist() == ["jump", "run", "crouch", "stand", "run", "stand"]
    assert got.index.equals(crouch.index)


def test_eb_shrink_pulls_small_n_harder_towards_prior():
    mean = pd.Series([3.0, 3.0, 1.0, 1.0])
    n = pd.Series([10, 1000, 1000, 1000])
    within_var = pd.Series([4.0, 4.0, 4.0, 4.0])
    eb = eb_shrink(mean, n, within_var, prior=2.0)
    # tau2 = var_ddof1([3,3,1,1]) - mean(4/n) = 1.3333 - 0.103 = 1.2303; B0 = 0.4 / (0.4 + 1.2303)
    b0 = 0.4 / (0.4 + 1.2303)
    assert eb[0] == pytest.approx(b0 * 2.0 + (1 - b0) * 3.0, abs=1e-3)
    assert abs(eb[0] - 2.0) < abs(eb[1] - 2.0)       # малое n ближе к prior
    assert 2.0 < eb[0] < eb[1] < 3.0                  # сжатие не перескакивает prior
    assert eb[1] == pytest.approx(3.0, abs=0.01)      # большое n почти не трогаем


def test_eb_shrink_collapses_to_prior_when_means_are_pure_noise():
    mean = pd.Series([2.01, 1.99])
    eb = eb_shrink(mean, pd.Series([30, 30]), pd.Series([4.0, 4.0]), prior=2.0)
    assert eb.tolist() == pytest.approx([2.0, 2.0], abs=1e-4)   # var(mean) << se2 -> tau2 = 1e-6


def make_scored(points: list[tuple[float, float]], **cols) -> pd.DataFrame:
    """flash_scored-подобный фрейм: по умолчанию T, зона 0, бросок стоя, train-матч."""
    n = len(points)
    df = pd.DataFrame({
        "throw_uid": np.arange(n, dtype="int64"), "match_id": np.ones(n, dtype="int64"),
        "team": pd.Categorical([2] * n, categories=[2, 3]),
        "zone_from": pd.Categorical([0] * n, categories=list(range(4))),
        "zone_to": pd.Categorical([1] * n, categories=list(range(4))),
        "crouch": np.zeros(n, dtype=bool), "airborne": np.zeros(n, dtype=bool),
        "speed": np.zeros(n, dtype="float32"),
        "throw_x": np.full(n, -500.0, dtype="float32"), "throw_y": np.full(n, 200.0, dtype="float32"),
        "throw_z": np.zeros(n, dtype="float32"), "yaw": np.full(n, 45.0, dtype="float32"),
        "pitch": np.full(n, -5.0, dtype="float32"),
        "to_x": np.array([p[0] for p in points], dtype="float32"),
        "to_y": np.array([p[1] for p in points], dtype="float32"),
        "to_z": np.zeros(n, dtype="float32"),
        "enemy_blind_s": np.ones(n, dtype="float32"), "teamflash": np.zeros(n, dtype=bool),
        "xflash_blind": np.ones(n, dtype="float32"), "is_test": np.zeros(n, dtype=bool),
    })
    for k, v in cols.items():
        df[k] = v
    return df


def cloud(rng, cx, cy, n, sd=12.0):
    return list(zip(rng.normal(cx, sd, n), rng.normal(cy, sd, n)))


@pytest.fixture
def two_clouds():
    """Облака в центрах ячеек 120x120 (60,60) и (1020,1020); каждая флешка из своего матча."""
    rng = np.random.default_rng(0)
    pts = (cloud(rng, 60, 60, 60) + cloud(rng, 1020, 1020, 50)
           + [(-3000, 3000), (3000, -3000), (-3000, -3000), (3000, 3000), (0, 2500)]   # шум
           + cloud(rng, 2100, -1860, 40))                                               # test-матч
    df = make_scored(pts)
    df["match_id"] = np.arange(len(df), dtype="int64")
    blind = np.r_[1.0 + rng.normal(0, 0.5, 60), 3.0 + rng.normal(0, 0.5, 50), np.zeros(5), np.full(40, 9.0)]
    df["enemy_blind_s"] = blind.astype("float32")
    df["teamflash"] = np.r_[np.arange(60) < 30, np.zeros(50, bool), np.zeros(5, bool), np.zeros(40, bool)]
    df["is_test"] = np.r_[np.zeros(115, bool), np.ones(40, bool)]
    return df


def test_build_lineups_finds_two_clouds_and_marks_noise(two_clouds):
    lineups, labels = build_lineups(two_clouds)
    assert list(lineups.columns) == LINEUP_COLUMNS
    assert len(lineups) == 2
    assert sorted(lineups["n"].tolist()) == [50, 60]
    assert all(lid.startswith("2-0-stand-") for lid in lineups["lineup_id"])

    assert labels.index.equals(two_clouds.index)
    assert labels.iloc[:60].nunique() == 1 and labels.iloc[60:110].nunique() == 1
    assert labels.iloc[0] != labels.iloc[60]
    assert labels.iloc[110:115].isna().all()          # шум
    assert labels.iloc[115:].isna().all()             # test-матчи в лайнапы не входят

    a = lineups.set_index("n").loc[60]
    assert a["blind_mean"] == pytest.approx(1.0, abs=0.2) and a["tf_rate"] == pytest.approx(0.5)
    assert a["blind_se"] == pytest.approx(two_clouds["enemy_blind_s"].iloc[:60].std() / np.sqrt(60), rel=1e-4)
    assert a["tf_se"] == pytest.approx(np.sqrt(0.25 / 64))  # Агрести–Коулл: p~=(30+2)/(60+4)=0.5
    assert a["to_x"] == pytest.approx(60, abs=15) and a["to_y"] == pytest.approx(60, abs=15)
    assert a["n_matches"] == 60 and (a["cell_lx"], a["cell_ly"]) == (0, 0) and (a["cell_tx"], a["cell_ty"]) == (-5, 1)
    assert a["yaw"] == pytest.approx(45.0) and a["zone_to"] == 1
    b = lineups.set_index("n").loc[50]
    # prior зоны = среднее по 115 train-флешкам (~1.8 с): оба лайнапа сдвинуты к нему, порядок сохранён
    assert a["blind_mean"] < a["blind_eb"] and b["blind_eb"] < b["blind_mean"]
    assert b["blind_eb"] > a["blind_eb"]
    assert b["score"] == pytest.approx(b["blind_eb"] - 2.0 * b["tf_eb"])
    # Review Focus: в облаке b нет ни одного тимфлеша — se всё равно > 0, ДИ разницы не вырождается в точку
    assert b["tf_rate"] == 0 and b["tf_se"] == pytest.approx(np.sqrt((2 / 54) * (52 / 54) / 54))


def test_se_uses_number_of_matches_not_flashes(two_clouds):
    shared = two_clouds.copy()
    shared["match_id"] = (np.arange(len(shared)) % 6).astype("int64")   # 60 флешек облака a -> 6 матчей
    lineups, _ = build_lineups(shared)
    a = lineups.set_index("n").loc[60]
    assert a["n_matches"] == 6
    assert a["blind_se"] == pytest.approx(shared["enemy_blind_s"].iloc[:60].std() / np.sqrt(6), rel=1e-4)
    p_ac = (0.5 * 6 + 2) / (6 + 4)
    assert a["tf_se"] == pytest.approx(np.sqrt(p_ac * (1 - p_ac) / (6 + 4)))


def test_cells_are_bounded_no_chain_merging():
    # цепочка точек с шагом 50: DBSCAN склеил бы её в один кластер, сетка режет на ячейки по 120
    pts = [(float(x), 0.0) for x in range(0, 1200, 50) for _ in range(30)]
    lineups, labels = build_lineups(make_scored(pts))
    assert len(lineups) > 5
    assert (lineups["cell_lx"] == np.floor(lineups["to_x"] / 120)).all()
    assert labels.notna().all()


def test_assign_equals_build_label_for_train_and_hits_cells_for_others(two_clouds):
    lineups, labels = build_lineups(two_clouds)
    assigned = assign_lineups(lineups, two_clouds)
    train = ~two_clouds["is_test"]
    assert assigned[train].tolist() == labels[train].tolist()
    assert assigned.iloc[110:].isna().all()                       # шум и test-облако вне ячеек лайнапов
    probe = make_scored([(70, 70), (400, 400)])
    probe["is_test"] = True
    assert assign_lineups(lineups, probe).tolist() == [labels.iloc[0], None]


def test_pooled_eb_single_lineup_group_keeps_signal():
    mean = pd.Series([3.0, 1.0, 5.0, 2.0])
    prior = pd.Series([2.0, 2.0, 2.0, 2.0])
    n = pd.Series([400, 400, 400, 400])
    wv = pd.Series([1.0, 1.0, 1.0, 1.0])
    eb = pooled_eb(mean, n, wv, prior)
    # tau2 большая -> почти нет сжатия; одиночная "группа" (элемент 2) не схлопывается в prior
    assert abs(eb[2] - 2.0) > 2.5 and eb[2] < 5.0
    assert eb.index.equals(mean.index)


def test_build_lineups_splits_by_mode_and_side():
    rng = np.random.default_rng(1)
    pts = cloud(rng, 60, 60, 40) + cloud(rng, 60, 60, 40)
    df = make_scored(pts)
    df.loc[40:, "airborne"] = True                     # та же точка, но прыжком
    lineups, _ = build_lineups(df)
    assert sorted(lineups["mode"]) == ["jump", "stand"]
    small = make_scored(cloud(rng, 60, 60, 20))           # меньше MIN_LINEUP_N -> лайнапов нет
    lineups_small, labels_small = build_lineups(small)
    assert lineups_small.empty and labels_small.isna().all()
    assert list(lineups_small.columns) == LINEUP_COLUMNS


def test_assign_lineups_exact_cell_match_only():
    lineups = pd.DataFrame({"lineup_id": ["2-0-stand-0", "2-0-stand-1", "2-0-jump-0"],
                            "team": [2, 2, 2], "zone_from": [0, 0, 0], "mode": ["stand", "stand", "jump"],
                            "cell_tx": [-5, -5, -5], "cell_ty": [1, 1, 1],
                            "cell_lx": [0, 8, 0], "cell_ly": [0, 0, 0]})
    scored = make_scored([(10, 10), (1000, 0), (500, 2000), (0, 0), (0, 0), (125, 10)])
    scored.loc[3, "airborne"] = True                    # прыжок -> лайнап jump
    scored["zone_from"] = pd.Categorical([0, 0, 0, 0, 1, 0], categories=list(range(4)))   # другая зона -> None
    scored["is_test"] = True
    got = assign_lineups(lineups, scored, max_dist=1e9)    # max_dist игнорируется
    # (125,10) всего в 5 единицах от ячейки 0, но в соседней ячейке -> None
    assert got.tolist() == ["2-0-stand-0", "2-0-stand-1", None, "2-0-jump-0", None, None]
    assert got.index.equals(scored.index)


def lineup(lid, score, n=100, team=2, zone_from=0, blind_mean=1.0, blind_se=0.05, tf_rate=0.2, tf_se=0.02):
    return {"lineup_id": lid, "team": team, "zone_from": zone_from, "n": n, "score": score,
            "blind_mean": blind_mean, "blind_se": blind_se, "tf_rate": tf_rate, "tf_se": tf_se}


def test_compare_significant_when_alt_clearly_better():
    own = pd.Series(lineup("own", 0.0, blind_mean=1.0, blind_se=0.05, tf_rate=0.3, tf_se=0.02))
    alt = pd.Series(lineup("alt", 0.0, blind_mean=2.0, blind_se=0.05, tf_rate=0.1, tf_se=0.02))
    r = compare(own, alt)
    assert r["blind_diff"] == pytest.approx(1.0) and r["tf_diff"] == pytest.approx(-0.2)
    half = 1.959964 * np.hypot(0.05, 0.05)
    assert r["blind_ci"] == pytest.approx([1.0 - half, 1.0 + half], abs=1e-6)
    assert r["score_diff"] == pytest.approx(1.0 + 2.0 * 0.2)
    se_s = np.sqrt(0.05 ** 2 * 2 + 4.0 * (0.02 ** 2 * 2))
    assert r["score_ci"] == pytest.approx([1.4 - 1.959964 * se_s, 1.4 + 1.959964 * se_s], abs=1e-5)
    assert r["significant"] is True


def test_compare_mixed_sign_is_judged_by_score_not_or_of_tests():
    # blind заметно лучше (+0.5, ДИ > 0), но тимфлешей на 0.3 больше: score_diff = 0.5 - 0.6 < 0
    own = pd.Series(lineup("own", 0.0, blind_mean=1.0, blind_se=0.05, tf_rate=0.1, tf_se=0.02))
    alt = pd.Series(lineup("alt", 0.0, blind_mean=1.5, blind_se=0.05, tf_rate=0.4, tf_se=0.02))
    r = compare(own, alt)
    assert r["blind_ci"][0] > 0
    assert r["score_ci"][0] < 0 and r["significant"] is False


def test_compare_not_significant_when_ci_cover_zero():
    own = pd.Series(lineup("own", 0.0, blind_mean=1.0, blind_se=0.2, tf_rate=0.3, tf_se=0.1))
    alt = pd.Series(lineup("alt", 0.0, blind_mean=1.05, blind_se=0.2, tf_rate=0.29, tf_se=0.1))
    r = compare(own, alt)
    assert r["score_ci"][0] < 0 < r["score_ci"][1]
    assert r["significant"] is False


def test_compare_alt_worse_is_never_significant():
    own = pd.Series(lineup("own", 0.0, blind_mean=2.0, tf_rate=0.1))
    alt = pd.Series(lineup("alt", 0.0, blind_mean=1.0, tf_rate=0.3))
    assert compare(own, alt)["significant"] is False
    assert compare(own, alt, n_candidates=1000)["significant"] is False


def test_compare_bonferroni_over_candidates():
    own = pd.Series(lineup("own", 0.0, blind_mean=1.0, blind_se=0.05, tf_rate=0.2, tf_se=0.0))
    alt = pd.Series(lineup("alt", 0.0, blind_mean=1.2, blind_se=0.05, tf_rate=0.2, tf_se=0.0))
    # score_diff = 0.2, se = 0.0707: z-stat 2.83 > 1.96 (K=1), но < 3.14 (K=30)
    assert compare(own, alt, n_candidates=1)["significant"] is True
    assert compare(own, alt, n_candidates=30)["significant"] is False
    alt_k = alt.copy()
    alt_k["n_candidates"] = 30                       # K читается из alt, если не передан явно
    assert compare(own, alt_k)["significant"] is False


def test_best_alternative_rules():
    lineups = pd.DataFrame([
        lineup("own", 1.0),
        lineup("alt1", 2.0),
        lineup("small", 5.0, n=20),                 # n < 30 — не рекомендуем
        lineup("other_zone", 9.0, zone_from=3),     # другая зона броска
        lineup("other_side", 9.0, team=3),          # другая сторона
    ])
    own = lineups.iloc[0]
    best = best_alternative(lineups, own)
    assert best is not None and best["lineup_id"] == "alt1"
    assert best["n_candidates"] == 1                 # подходящих кандидатов: только alt1

    top = lineups.copy()
    top.loc[0, "score"] = 3.0                        # own уже лучший -> себя не возвращаем, None
    assert best_alternative(top, top.iloc[0]) is None
    alone = lineups[lineups["lineup_id"].isin(["own", "small"])]
    assert best_alternative(alone, alone.iloc[0]) is None


def test_run_lineups_writes_both_files(tmp_path, two_clouds):
    two_clouds.to_parquet(tmp_path / "flash_scored.parquet", index=False)
    run_lineups(tmp_path)
    lineups = pd.read_parquet(tmp_path / "lineups.parquet")
    fl = pd.read_parquet(tmp_path / "flash_lineups.parquet")
    assert list(lineups.columns) == LINEUP_COLUMNS and len(lineups) == 2
    assert list(fl.columns) == ["throw_uid", "lineup_id", "mode"]
    assert len(fl) == len(two_clouds)                 # все флешки, включая test
    assert fl["lineup_id"].iloc[:110].notna().all()
    assert fl["lineup_id"].iloc[110:].isna().all()    # шум и далёкое test-облако без лайнапа
    assert set(fl["mode"]) == {"stand"}
