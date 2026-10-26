"""Гипотеза о рейтинге: метрики игрок x матч и корреляция Спирмена с бутстрепом."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from utility_coach.config import DATA, SEED  # noqa: E402
from utility_coach.model import to_jsonable  # noqa: E402

METRICS: dict[str, str] = {
    "blind_minus_x": "факт − xFlash, с на флешку",
    "tf_rate": "доля тимфлешей",
    "flash_kill_rate": "доля флешек с флеш-фрагом",
}
RATING_BIN = 2500
MIN_BIN_N = 20
Z95 = 1.959963984540054
PM_COLUMNS: list[str] = ["match_id", "player_id", "rating", "n_flashes", "enemy_blind_s_sum",
                         "blind_minus_x", "tf_rate", "flash_kill_rate"]


def player_match_metrics(scored: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на (match_id, бросающий). blind_minus_x = среднее (enemy_blind_s - xflash_blind) по флешкам."""
    s = pd.DataFrame({
        "match_id": scored["match_id"].to_numpy(dtype="int64"),
        "player_id": scored["thrower_id"].to_numpy(dtype="int64"),
        "blind": scored["enemy_blind_s"].to_numpy(dtype="float64"),
        "resid": scored["enemy_blind_s"].to_numpy(dtype="float64") - scored["xflash_blind"].to_numpy(dtype="float64"),
        "tf": scored["teamflash"].to_numpy(dtype="float64"),
        "kill": scored["flash_kill"].to_numpy(dtype="float64"),
    })
    pm = s.groupby(["match_id", "player_id"], as_index=False, sort=True).agg(
        n_flashes=("blind", "size"), enemy_blind_s_sum=("blind", "sum"), blind_minus_x=("resid", "mean"),
        tf_rate=("tf", "mean"), flash_kill_rate=("kill", "mean"))
    pl = (players[["match_id", "player_id", "rating"]]
          .astype({"match_id": "int64", "player_id": "int64", "rating": "float64"})
          .drop_duplicates(["match_id", "player_id"]))
    pm = pm.merge(pl, on=["match_id", "player_id"], how="left")
    pm = pm.astype({"n_flashes": "int32", "rating": "float32", "enemy_blind_s_sum": "float32",
                    "blind_minus_x": "float32", "tf_rate": "float32", "flash_kill_rate": "float32"})
    return pm[PM_COLUMNS]


def spearman_bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = SEED) -> dict:
    """rho и p из scipy.stats.spearmanr; 95% ДИ — перцентильный бутстреп по парам."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(len(x))
    nan = float("nan")
    if n < 3:
        return {"rho": nan, "ci_low": nan, "ci_high": nan, "p": nan, "n": n}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # ConstantInputWarning на вырожденных выборках
        rho, p = spearmanr(x, y)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            boots[i] = spearmanr(x[idx], y[idx])[0]  # NaN, если выборка вышла константной
    boots = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (nan, nan))
    return {"rho": float(rho), "ci_low": float(lo), "ci_high": float(hi), "p": float(p), "n": n}


def _rating_bins(sub: pd.DataFrame) -> pd.DataFrame:
    """Среднее и 95% ДИ каждой метрики по бинам рейтинга шириной RATING_BIN (бины с n >= MIN_BIN_N)."""
    b = (np.floor(sub["rating"].to_numpy(dtype="float64") / RATING_BIN) * RATING_BIN).astype("int64")
    rows = []
    for start, g in sub.groupby(b, sort=True):
        if len(g) < MIN_BIN_N:
            continue
        row = {"rating_bin": int(start), "n": int(len(g))}
        for m in METRICS:
            v = g[m].to_numpy(dtype="float64")
            half = Z95 * v.std(ddof=1) / np.sqrt(len(v))
            row[m] = float(v.mean())
            row[f"{m}_ci"] = [float(v.mean() - half), float(v.mean() + half)]
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_rating(bins: pd.DataFrame, results: dict[str, dict], out_png: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    for ax, (m, label) in zip(np.atleast_1d(axes), METRICS.items()):
        r = results[m]
        if len(bins):
            x = bins["rating_bin"].to_numpy() + RATING_BIN / 2
            mean = bins[m].to_numpy()
            lo = np.array([c[0] for c in bins[f"{m}_ci"]])
            hi = np.array([c[1] for c in bins[f"{m}_ci"]])
            ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="o-", capsize=3, color="#4C72B0")
        ax.set_xlabel("Premier rating (бины по 2500)")
        ax.set_ylabel(label)
        ax.set_title(f"rho={r['rho']:.3f} [{r['ci_low']:.3f}; {r['ci_high']:.3f}], n={r['n']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_rating(data: Path = DATA, min_flashes: int = 5) -> dict:
    """player_match.parquet (все пары), rating_hypothesis.json и figures/rating.png (пары с рейтингом и >= min_flashes)."""
    data = Path(data)
    scored = pd.read_parquet(data / "flash_scored.parquet",
                             columns=["match_id", "thrower_id", "enemy_blind_s", "xflash_blind",
                                      "teamflash", "flash_kill"])
    players = pd.read_parquet(data / "players.parquet", columns=["match_id", "player_id", "rating"])
    pm = player_match_metrics(scored, players)
    pm.to_parquet(data / "player_match.parquet", index=False)

    sub = pm[(pm["n_flashes"] >= min_flashes) & pm["rating"].notna()]
    results = {m: spearman_bootstrap(sub["rating"].to_numpy(), sub[m].to_numpy()) for m in METRICS}
    bins = _rating_bins(sub) if len(sub) else pd.DataFrame()
    out = {
        "min_flashes": int(min_flashes),
        "n_player_match": int(len(pm)),
        "n_with_rating": int(pm["rating"].notna().sum()),
        "n_pairs": int(len(sub)),
        "metrics": results,
        "bins": bins.to_dict(orient="records") if len(bins) else [],
    }
    (data / "rating_hypothesis.json").write_text(json.dumps(to_jsonable(out), ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    fig_dir = data / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_rating(bins, results, fig_dir / "rating.png")
    for m, r in results.items():
        print(f"{m}: rho={r['rho']:.4f} CI=[{r['ci_low']:.4f}; {r['ci_high']:.4f}] p={r['p']:.3g} n={r['n']}")
    return out
