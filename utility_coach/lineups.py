"""Лайнапы: (сторона, зона броска, тип броска) + ячейка сетки в пространстве (точка броска, точка приземления).

Ячейка: floor(throw_x/E), floor(throw_y/E), floor(to_x/E), floor(to_y/E), E = LINEUP_EPS. Лайнап = ячейка с
>= MIN_LINEUP_N train-флешками. DBSCAN по точкам приземления отвергнут: он сцеплял приземления цепочкой в кластеры
до 41k флешек, в которых нет одного воспроизводимого броска. Сжатие: empirical Bayes с общей tau2 на сторону.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from utility_coach.config import DATA, LINEUP_EPS, MIN_LINEUP_N, TF_PENALTY_S

RUN_SPEED = 100.0          # горизонтальная скорость, начиная с которой бросок считается "с бега"
LINEUP_COLUMNS: list[str] = ["lineup_id", "team", "zone_from", "zone_to", "mode", "n", "xflash_mean",
                             "blind_mean", "blind_se", "tf_rate", "tf_se", "blind_eb", "tf_eb", "score",
                             "throw_x", "throw_y", "throw_z", "yaw", "pitch", "to_x", "to_y", "to_z",
                             "n_matches", "cell_tx", "cell_ty", "cell_lx", "cell_ly"]
SCORED_COLS: list[str] = ["throw_uid", "match_id", "team", "zone_from", "zone_to", "crouch", "airborne", "speed",
                          "throw_x", "throw_y", "throw_z", "yaw", "pitch", "to_x", "to_y", "to_z",
                          "enemy_blind_s", "teamflash", "xflash_blind", "is_test"]
CELL_COLS = ["cell_tx", "cell_ty", "cell_lx", "cell_ly"]
KEY_COLS = ["team", "zone_from", "mode"] + CELL_COLS


def throw_mode(crouch: pd.Series, airborne: pd.Series, speed: pd.Series) -> pd.Series:
    """Приоритет: "jump" > "run" (speed >= RUN_SPEED) > "crouch" > "stand"."""
    a = airborne.to_numpy(dtype=bool)
    c = crouch.to_numpy(dtype=bool)
    s = speed.to_numpy(dtype="float64")
    mode = np.select([a, s >= RUN_SPEED, c], ["jump", "run", "crouch"], default="stand")
    return pd.Series(mode.astype(object), index=crouch.index, name="mode")


def eb_shrink(mean: pd.Series, n: pd.Series, within_var: pd.Series, prior: float) -> pd.Series:
    """Сжатие средних к prior методом моментов.

    se2_i = within_var_i / n_i; tau2 = max(var_ddof1(mean) - mean(se2), 1e-6);
    B_i = se2_i / (se2_i + tau2); eb_i = B_i * prior + (1 - B_i) * mean_i.
    """
    m = mean.to_numpy(dtype="float64")
    se2 = np.nan_to_num(within_var.to_numpy(dtype="float64"), nan=0.0) / n.to_numpy(dtype="float64")
    tau2 = max(float(np.var(m, ddof=1 if len(m) > 1 else 0)) - float(np.mean(se2)), 1e-6)
    b = se2 / (se2 + tau2)
    return pd.Series(b * prior + (1.0 - b) * m, index=mean.index)


def pooled_eb(mean: pd.Series, n: pd.Series, within_var: pd.Series, prior: pd.Series) -> pd.Series:
    """EB с ОДНОЙ tau2 на все переданные лайнапы; каждый сжимается к своему prior (Series на индексе mean).

    tau2 = max(var_ddof1(mean - prior) - mean(within_var/n), 1e-6); B = se2/(se2+tau2).
    """
    m = mean.to_numpy(dtype="float64")
    pr = prior.to_numpy(dtype="float64")
    se2 = np.nan_to_num(within_var.to_numpy(dtype="float64"), nan=0.0) / n.to_numpy(dtype="float64")
    tau2 = max(float(np.var(m - pr, ddof=1 if len(m) > 1 else 0)) - float(np.mean(se2)), 1e-6)
    b = se2 / (se2 + tau2)
    return pd.Series(b * pr + (1.0 - b) * m, index=mean.index)


def _circular_median_deg(deg: np.ndarray) -> float:
    """Медиана угла с учётом перехода через ±180 (yaw). Возвращает угол в исходном диапазоне."""
    rad = np.deg2rad(deg)
    ref = np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
    diff = (deg - ref + 180.0) % 360.0 - 180.0
    med = (ref + float(np.median(diff)) + 180.0) % 360.0 - 180.0
    if (deg >= 0).all() and med < 0:
        med += 360.0                     # данные в [0, 360) -> ответ тоже в [0, 360)
    return float(med)


def _cell_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Ключ лайнапа каждой флешки: (team, zone_from, mode, ячейка броска, ячейка приземления)."""
    e = LINEUP_EPS
    return pd.DataFrame({
        "team": df["team"].astype("int64").to_numpy(),
        "zone_from": df["zone_from"].astype("int64").to_numpy(),
        "mode": throw_mode(df["crouch"], df["airborne"], df["speed"]).to_numpy(),
        "cell_tx": np.floor(df["throw_x"].to_numpy(dtype="float64") / e).astype("int64"),
        "cell_ty": np.floor(df["throw_y"].to_numpy(dtype="float64") / e).astype("int64"),
        "cell_lx": np.floor(df["to_x"].to_numpy(dtype="float64") / e).astype("int64"),
        "cell_ly": np.floor(df["to_y"].to_numpy(dtype="float64") / e).astype("int64"),
    })


def build_lineups(train_scored: pd.DataFrame, eps: float = LINEUP_EPS,
                  min_samples: int = MIN_LINEUP_N) -> tuple[pd.DataFrame, pd.Series]:
    """Лайнапы по train-флешкам (строки с is_test=True игнорируются, если колонка есть).

    eps не используется (размер ячейки задаёт LINEUP_EPS), оставлен ради сигнатуры контракта.
    se считаются по n_eff = n_matches (число различных матчей лайнапа): консервативная оценка
    design effect при зависимости флешек внутри матча.
    Возвращает (lineups с колонками LINEUP_COLUMNS, Series lineup_id на индексе train_scored; шум/test -> None).
    """
    df = train_scored
    use = ~df["is_test"].to_numpy(dtype=bool) if "is_test" in df.columns else np.ones(len(df), dtype=bool)
    keys_all = _cell_keys(df)
    zone_to = df["zone_to"].astype("int64").to_numpy()
    match = df["match_id"].to_numpy()
    blind = df["enemy_blind_s"].to_numpy(dtype="float64")
    tf = df["teamflash"].to_numpy(dtype="float64")
    xflash = (df["xflash_blind"].to_numpy(dtype="float64") if "xflash_blind" in df.columns
              else np.full(len(df), np.nan))
    coords = {c: df[c].to_numpy(dtype="float64") for c in ("throw_x", "throw_y", "throw_z", "yaw", "pitch",
                                                            "to_x", "to_y", "to_z")}

    labels = np.full(len(df), None, dtype=object)
    pos_all = np.flatnonzero(use)
    keys = keys_all[use].reset_index(drop=True)
    rows = []
    counter: dict = {}
    for key, idx in keys.groupby(KEY_COLS, sort=True).indices.items():
        if len(idx) < min_samples:
            continue
        members = pos_all[idx]
        t, z, m = int(key[0]), int(key[1]), key[2]
        k = counter.get((t, z, m), 0)
        counter[(t, z, m)] = k + 1
        n = len(members)
        mm = len(np.unique(match[members]))
        lid = f"{t}-{z}-{m}-{k}"
        labels[members] = lid
        p = float(tf[members].mean())
        # Агрести–Коулл по числу матчей: se не схлопывается в 0 при p=0 или p=1
        p_ac = (p * mm + 2.0) / (mm + 4.0)
        row = {
            "lineup_id": lid, "team": t, "zone_from": z,
            "zone_to": int(np.bincount(zone_to[members]).argmax()), "mode": m, "n": n,
            "xflash_mean": float(np.mean(xflash[members])),
            "blind_mean": float(blind[members].mean()),
            "blind_var": float(blind[members].var(ddof=1)),
            "blind_se": float(blind[members].std(ddof=1) / np.sqrt(mm)),
            "tf_rate": p, "tf_se": float(np.sqrt(p_ac * (1.0 - p_ac) / (mm + 4.0))),
            "n_matches": mm,
            "cell_tx": int(key[3]), "cell_ty": int(key[4]), "cell_lx": int(key[5]), "cell_ly": int(key[6]),
        }
        for c, arr in coords.items():
            row[c] = _circular_median_deg(arr[members]) if c == "yaw" else float(np.median(arr[members]))
        rows.append(row)

    label_series = pd.Series(labels, index=df.index, name="lineup_id")
    if not rows:
        return pd.DataFrame(columns=LINEUP_COLUMNS), label_series

    lineups = pd.DataFrame(rows)
    # prior = среднее по ВСЕМ train-флешкам той же (team, zone_from), включая шум и все типы броска
    zone_prior = (keys_all[use][["team", "zone_from"]].assign(prior_blind=blind[use], prior_tf=tf[use])
                  .groupby(["team", "zone_from"])[["prior_blind", "prior_tf"]].mean())
    lineups = lineups.join(zone_prior, on=["team", "zone_from"])
    lineups["blind_eb"] = np.nan
    lineups["tf_eb"] = np.nan
    for _, g in lineups.groupby("team"):          # одна tau2 на сторону, не на (team, zone_from)
        lineups.loc[g.index, "blind_eb"] = pooled_eb(g["blind_mean"], g["n"], g["blind_var"], g["prior_blind"])
        # для доли берём биномиальную дисперсию prior: у лайнапа с 0 тимфлешей p(1-p)=0 и сжатия бы не было
        tf_var = g["prior_tf"] * (1.0 - g["prior_tf"])
        lineups.loc[g.index, "tf_eb"] = pooled_eb(g["tf_rate"], g["n"], tf_var, g["prior_tf"])
    lineups["score"] = lineups["blind_eb"] - TF_PENALTY_S * lineups["tf_eb"]
    lineups = lineups.astype({"team": "int8", "zone_from": "int16", "zone_to": "int16", "n": "int32",
                              "n_matches": "int32", "cell_tx": "int32", "cell_ty": "int32",
                              "cell_lx": "int32", "cell_ly": "int32"})
    return lineups[LINEUP_COLUMNS].reset_index(drop=True), label_series


def assign_lineups(lineups: pd.DataFrame, scored: pd.DataFrame, max_dist: float = 2 * LINEUP_EPS) -> pd.Series:
    """Флешка получает лайнап, если её (team, zone_from, mode, ячейка) совпадает с ключом лайнапа, иначе None.

    max_dist больше не используется (оставлен ради сигнатуры): правило членства здесь и в build_lineups одно.
    """
    if len(lineups) == 0 or len(scored) == 0:
        return pd.Series(np.full(len(scored), None, dtype=object), index=scored.index, name="lineup_id")
    lu = pd.DataFrame({"team": lineups["team"].astype("int64").to_numpy(),
                       "zone_from": lineups["zone_from"].astype("int64").to_numpy(),
                       "mode": lineups["mode"].astype(str).to_numpy(),
                       **{c: lineups[c].astype("int64").to_numpy() for c in CELL_COLS},
                       "lineup_id": lineups["lineup_id"].to_numpy(dtype=object)})
    merged = _cell_keys(scored).merge(lu, on=KEY_COLS, how="left")
    ids = merged["lineup_id"].to_numpy(dtype=object)
    ids[pd.isna(ids)] = None
    return pd.Series(ids, index=scored.index, name="lineup_id")


def compare(own: pd.Series, alt: pd.Series, n_candidates: int | None = None) -> dict:
    """Разница alt - own по СЫРЫМ оценкам (blind_mean, tf_rate); EB используется только для ранжирования.

    score_diff = blind_diff - TF_PENALTY_S * tf_diff, se = sqrt(se_b^2 + TF_PENALTY_S^2 * se_t^2)
    (ковариация blind/tf игнорируется). z = norm.ppf(1 - 0.025 / K), K = n_candidates (по умолчанию
    alt["n_candidates"], иначе 1): поправка Бонферрони на выбор лучшего из K (winner's curse).
    significant = нижняя граница ДИ score_diff > 0.
    """
    k = n_candidates if n_candidates is not None else (alt["n_candidates"] if "n_candidates" in alt.index else 1)
    z = float(norm.ppf(1.0 - 0.025 / max(1, int(k))))
    blind_diff = float(alt["blind_mean"] - own["blind_mean"])
    se_b = float(np.hypot(alt["blind_se"], own["blind_se"]))
    tf_diff = float(alt["tf_rate"] - own["tf_rate"])
    se_t = float(np.hypot(alt["tf_se"], own["tf_se"]))
    score_diff = blind_diff - TF_PENALTY_S * tf_diff
    se_s = float(np.sqrt(se_b ** 2 + TF_PENALTY_S ** 2 * se_t ** 2))
    score_ci = [score_diff - z * se_s, score_diff + z * se_s]
    return {"blind_diff": blind_diff, "blind_ci": [blind_diff - z * se_b, blind_diff + z * se_b],
            "tf_diff": tf_diff, "tf_ci": [tf_diff - z * se_t, tf_diff + z * se_t],
            "score_diff": score_diff, "score_ci": score_ci, "significant": bool(score_ci[0] > 0)}


def best_alternative(lineups: pd.DataFrame, own: pd.Series) -> pd.Series | None:
    """Лучший по score (EB) лайнап той же (team, zone_from), не own, с n >= MIN_LINEUP_N и score выше own.

    В результат добавлено n_candidates = K числа подходящих кандидатов (для поправки Бонферрони в compare).
    """
    cand = lineups[(lineups["team"].astype("int64") == int(own["team"]))
                   & (lineups["zone_from"].astype("int64") == int(own["zone_from"]))
                   & (lineups["lineup_id"] != own["lineup_id"])
                   & (lineups["n"] >= MIN_LINEUP_N)]
    if cand.empty:
        return None
    best = cand.loc[cand["score"].idxmax()].copy()
    if not float(best["score"]) > float(own["score"]):
        return None
    best["n_candidates"] = len(cand)
    return best


def run_lineups(data: Path = DATA) -> None:
    """flash_scored.parquet -> lineups.parquet (по train) + flash_lineups.parquet (все флешки)."""
    data = Path(data)
    scored = pd.read_parquet(data / "flash_scored.parquet", columns=SCORED_COLS)
    lineups, _ = build_lineups(scored)
    lineups.to_parquet(data / "lineups.parquet", index=False)

    mode = throw_mode(scored["crouch"], scored["airborne"], scored["speed"])
    lineup_id = assign_lineups(lineups, scored)
    flash_lineups = pd.DataFrame({"throw_uid": scored["throw_uid"].to_numpy(dtype="int64"),
                                  "lineup_id": lineup_id.to_numpy(dtype=object),
                                  "mode": mode.to_numpy(dtype=object)})
    flash_lineups.to_parquet(data / "flash_lineups.parquet", index=False)

    assigned = lineup_id.notna().to_numpy()
    is_test = scored["is_test"].to_numpy(dtype=bool)
    print(f"lineups: {len(lineups)} лайнапов, медиана n={int(lineups['n'].median()) if len(lineups) else 0}, "
          f"покрытие train={assigned[~is_test].mean():.3f}, test={assigned[is_test].mean():.3f}")
    print("режимы бросков:", mode.value_counts(normalize=True).round(3).to_dict())
