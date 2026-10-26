import numpy as np
import pandas as pd
import pytest

from utility_coach.stats import PM_COLUMNS, player_match_metrics, spearman_bootstrap


def make_scored_for_stats() -> pd.DataFrame:
    return pd.DataFrame({
        "throw_uid": np.arange(6, dtype="int64"),
        "match_id": np.array([1, 1, 1, 1, 2, 2], dtype="int64"),
        "thrower_id": np.array([10, 10, 10, 11, 10, 10], dtype="int64"),
        "enemy_blind_s": np.array([2.0, 0.0, 1.0, 4.0, 1.0, 1.0], dtype="float32"),
        "xflash_blind": np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0], dtype="float32"),
        "teamflash": np.array([True, False, False, False, True, True]),
        "flash_kill": np.array([True, True, False, False, False, False]),
    })


def make_players() -> pd.DataFrame:
    return pd.DataFrame({"match_id": np.array([1, 2], dtype="int64"), "player_id": np.array([10, 10], dtype="int64"),
                         "rating": np.array([15000.0, np.nan], dtype="float32")})


def test_player_match_metrics_by_hand():
    pm = player_match_metrics(make_scored_for_stats(), make_players())
    assert list(pm.columns) == PM_COLUMNS
    assert len(pm) == 3
    r = pm.set_index(["match_id", "player_id"])
    a = r.loc[(1, 10)]
    assert a["n_flashes"] == 3 and a["enemy_blind_s_sum"] == pytest.approx(3.0)
    assert a["blind_minus_x"] == pytest.approx(0.0)                  # mean(2-1, 0-1, 1-1)
    assert a["tf_rate"] == pytest.approx(1 / 3) and a["flash_kill_rate"] == pytest.approx(2 / 3)
    assert a["rating"] == pytest.approx(15000.0)
    b = r.loc[(1, 11)]
    assert b["n_flashes"] == 1 and b["blind_minus_x"] == pytest.approx(3.0) and np.isnan(b["rating"])
    c = r.loc[(2, 10)]                                               # тот же игрок, другой матч — отдельная пара
    assert c["blind_minus_x"] == pytest.approx(-1.0) and c["tf_rate"] == pytest.approx(1.0)
    assert np.isnan(c["rating"])


def test_spearman_bootstrap_monotone_gives_rho_one_and_narrow_ci():
    x = np.arange(200, dtype=float)
    r = spearman_bootstrap(x, x ** 3, n_boot=300)
    assert r["rho"] == pytest.approx(1.0)
    assert r["ci_low"] > 0.99 and r["ci_high"] == pytest.approx(1.0)
    assert r["p"] < 1e-10 and r["n"] == 200
    assert set(r) == {"rho", "ci_low", "ci_high", "p", "n"}


def test_spearman_bootstrap_independent_ci_covers_zero():
    rng = np.random.default_rng(3)
    r = spearman_bootstrap(rng.normal(size=400), rng.normal(size=400), n_boot=500)
    assert r["ci_low"] < 0 < r["ci_high"]
    assert abs(r["rho"]) < 0.1


def test_spearman_bootstrap_drops_nan_and_handles_tiny_n():
    r = spearman_bootstrap(np.array([1.0, 2.0, np.nan, 4.0, 5.0]), np.array([1.0, 2.0, 3.0, np.nan, 6.0]), n_boot=50)
    assert r["n"] == 3
    tiny = spearman_bootstrap(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert tiny["n"] == 2 and np.isnan(tiny["rho"])


import json  # noqa: E402

from utility_coach.stats import run_rating  # noqa: E402


def test_run_rating_on_synthetic(tmp_path):
    rng = np.random.default_rng(0)
    n_pairs, per = 600, 6
    match = np.repeat(np.arange(n_pairs), per).astype("int64")
    player = (match * 10 + 1).astype("int64")
    rating = rng.uniform(5000, 25000, n_pairs)
    # чем выше рейтинг, тем меньше тимфлешей — rho для tf_rate должен быть отрицательным
    p_tf = np.repeat(np.clip(0.6 - rating / 50000, 0.05, 0.95), per)
    scored = pd.DataFrame({
        "throw_uid": np.arange(n_pairs * per, dtype="int64"), "match_id": match, "thrower_id": player,
        "enemy_blind_s": rng.uniform(0, 4, n_pairs * per).astype("float32"),
        "xflash_blind": np.full(n_pairs * per, 2.0, dtype="float32"),
        "teamflash": rng.random(n_pairs * per) < p_tf,
        "flash_kill": rng.random(n_pairs * per) < 0.1,
    })
    # одна пара с 3 флешками (ниже min_flashes) и половина пар без рейтинга
    scored = pd.concat([scored, scored.iloc[:3].assign(match_id=9999, thrower_id=1)], ignore_index=True)
    rating_col = np.where(np.arange(n_pairs) % 2 == 0, rating, np.nan)
    players = pd.DataFrame({"match_id": np.r_[np.arange(n_pairs), 9999].astype("int64"),
                            "player_id": np.r_[np.arange(n_pairs) * 10 + 1, 1].astype("int64"),
                            "rating": np.r_[rating_col, 20000.0].astype("float32")})
    scored.to_parquet(tmp_path / "flash_scored.parquet", index=False)
    players.to_parquet(tmp_path / "players.parquet", index=False)

    res = run_rating(tmp_path)

    assert res["n_pairs"] == n_pairs // 2
    assert set(res["metrics"]) == {"blind_minus_x", "tf_rate", "flash_kill_rate"}
    assert res["metrics"]["tf_rate"]["ci_high"] < 0
    pm = pd.read_parquet(tmp_path / "player_match.parquet")
    assert len(pm) == n_pairs + 1                       # все пары, включая без рейтинга и с 3 флешками
    on_disk = json.loads((tmp_path / "rating_hypothesis.json").read_text(encoding="utf-8"))
    assert on_disk["metrics"]["tf_rate"]["n"] == n_pairs // 2
    assert on_disk["bins"] and {"rating_bin", "n", "tf_rate", "tf_rate_ci"} <= set(on_disk["bins"][0])
    assert (tmp_path / "figures" / "rating.png").exists()
