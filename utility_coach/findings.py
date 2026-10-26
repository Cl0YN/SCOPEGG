"""Факты по одному игроку в одном матче -> findings JSON.

Числа сразу в виде для показа: секунды с 1 знаком, доли в целых процентах, углы с 1 знаком,
координаты целыми. Реальные PlayerID / MatchID в JSON не попадают: игрок называется
псевдонимом «Игрок A..E», матч не называется вовсе.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from utility_coach.config import DATA, MIN_LINEUP_N, SEED, TEAM_CT, TEAM_T
from utility_coach.lineups import best_alternative, compare
from utility_coach.zones import load_glossary

CARD_LETTERS = "ABCDE"
Z95 = 1.96
SCORED_COLUMNS = ["throw_uid", "match_id", "thrower_id", "team", "round", "zone_from", "zone_to",
                  "enemy_blind_s", "teamflash", "flash_kill", "is_test"]


def _r1(x) -> float:
    """Секунды и углы: 1 знак. `+ 0.0` убирает -0.0."""
    return round(float(x), 1) + 0.0


def _pct(x) -> int:
    """Доля -> целые проценты."""
    return int(round(100.0 * float(x)))


def _int(x) -> int:
    """Координаты -> целые."""
    return int(round(float(x)))


def _side(team) -> str:
    return {TEAM_T: "T", TEAM_CT: "CT"}.get(int(team), str(int(team)))


def _zone(glossary: dict[int, str], zone) -> str:
    z = int(zone)
    return glossary.get(z, f"Зона {z}")


def _valid_lid(lineup_id) -> bool:
    return isinstance(lineup_id, str) and lineup_id != ""


def _lineup_rows(lineups: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["lineup_id"]): row for _, row in lineups.iterrows()}


def _with_lineups(scored: pd.DataFrame, flash_lineups: pd.DataFrame) -> pd.DataFrame:
    left = scored.drop(columns=[c for c in ("lineup_id", "mode") if c in scored.columns])
    return left.merge(flash_lineups[["throw_uid", "lineup_id", "mode"]], on="throw_uid", how="left")


def _recommendation(rows: dict[str, pd.Series], lineups: pd.DataFrame, lineup_id: str,
                    cache: dict[str, dict | None]) -> dict | None:
    """Значимая альтернатива для лайнапа игрока или None. Зависит только от лайнапа, поэтому кэшируется."""
    if lineup_id in cache:
        return cache[lineup_id]
    rec = None
    own = rows.get(lineup_id)
    if own is not None and int(own["n"]) >= MIN_LINEUP_N:
        alt = best_alternative(lineups, own)
        if alt is not None and int(alt["n"]) >= MIN_LINEUP_N:
            cmp = compare(own, alt)
            if bool(cmp["significant"]):
                rec = {"own": own, "alt": alt, "cmp": cmp,
                       "gain": float(alt["score"]) - float(own["score"])}
    cache[lineup_id] = rec
    return rec


def select_cards(scored: pd.DataFrame, flash_lineups: pd.DataFrame, lineups: pd.DataFrame,
                 n_cards: int = 5, min_flashes: int = 6, seed: int = SEED) -> list[tuple[int, int]]:
    """Пары (match_id, player_id) только из test-матчей, с >= min_flashes флешками и >= 1 значимой находкой."""
    test = scored[scored["is_test"].astype(bool)]
    t = _with_lineups(test, flash_lineups)
    counts = t.groupby(["match_id", "thrower_id"]).size()
    eligible = {(int(m), int(p)) for (m, p), k in counts.items() if int(k) >= min_flashes}

    rows = _lineup_rows(lineups)
    cache: dict[str, dict | None] = {}
    good = [lid for lid in t["lineup_id"].dropna().unique()
            if _valid_lid(lid) and _recommendation(rows, lineups, lid, cache) is not None]
    hits = t[t["lineup_id"].isin(good)]
    with_hit = {(int(m), int(p)) for m, p in zip(hits["match_id"], hits["thrower_id"])}

    candidates = sorted(eligible & with_hit)
    if len(candidates) <= n_cards:
        return candidates
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(candidates), size=n_cards, replace=False).tolist())
    return [candidates[i] for i in idx]


def _lineup_block(row: pd.Series) -> dict:
    """Показываем сырые оценки (по ним же идёт проверка значимости в compare); EB только ранжирует.

    xflash_s — среднее фактическое ослепление врагов у бросков лайнапа в train;
    model_xflash_s — средний прогноз модели xFlash по тем же броскам (xflash_mean).
    """
    mean, se = float(row["blind_mean"]), float(row["blind_se"])
    tf, tf_se = float(row["tf_rate"]), float(row["tf_se"])
    return {
        "xflash_s": _r1(mean),
        "model_xflash_s": _r1(row["xflash_mean"]),
        "teamflash_pct": _pct(tf),
        "n": int(row["n"]),
        "ci95_blind_s": [_r1(max(0.0, mean - Z95 * se)), _r1(mean + Z95 * se)],
        "ci95_teamflash_pct": [_pct(max(0.0, tf - Z95 * tf_se)), _pct(min(1.0, tf + Z95 * tf_se))],
    }


def _finding(index: int, flash: dict, rec: dict, glossary: dict[int, str], times: int) -> dict:
    own, alt = rec["own"], rec["alt"]
    own_block = _lineup_block(own)
    alt_block = {
        "zone_to": _zone(glossary, alt["zone_to"]),
        "mode": str(alt["mode"]),
        **_lineup_block(alt),
        "yaw": _r1(alt["yaw"]),
        "pitch": _r1(alt["pitch"]),
        "throw_pos": [_int(alt["throw_x"]), _int(alt["throw_y"]), _int(alt["throw_z"])],
        # расстояние по XY между медианными точками броска своего лайнапа и альтернативы
        "throw_dist_u": _int(math.hypot(float(alt["throw_x"]) - float(own["throw_x"]),
                                        float(alt["throw_y"]) - float(own["throw_y"]))),
    }
    return {
        "id": f"f{index}",
        "round": int(flash["round"]) + 1,  # RoundIndexMatch с 0, игроку показываем с 1
        "side": _side(flash["team"]),
        "zone_from": _zone(glossary, flash["zone_from"]),
        "zone_to": _zone(glossary, flash["zone_to"]),
        "mode": str(flash["mode"]),
        "times_in_match": int(times),
        "actual": {"enemy_blind_s": _r1(flash["enemy_blind_s"]), "teamflash": bool(flash["teamflash"])},
        "own_lineup": own_block,
        "alt_lineup": alt_block,
        "diff": {
            "blind_s": _r1(alt_block["xflash_s"] - own_block["xflash_s"]),
            "teamflash_pct": int(alt_block["teamflash_pct"] - own_block["teamflash_pct"]),
        },
    }


def _percentile(player_match: pd.DataFrame, value: float) -> int | None:
    col = player_match["enemy_blind_s_sum"].dropna().to_numpy(dtype=float)
    if col.size == 0:
        return None
    return int(round(100.0 * float(np.mean(col <= value + 1e-9))))


def _rating(players: pd.DataFrame, match_id: int, player_id: int) -> int | None:
    sel = players.loc[(players["match_id"] == match_id) & (players["player_id"] == player_id), "rating"]
    if sel.empty or pd.isna(sel.iloc[0]):
        return None
    return _int(sel.iloc[0])


def build_findings(match_id: int, player_id: int, alias: str, scored: pd.DataFrame, flash_lineups: pd.DataFrame,
                   lineups: pd.DataFrame, player_match: pd.DataFrame, players: pd.DataFrame,
                   glossary: dict[int, str], max_findings: int = 4) -> dict:
    mine = scored[(scored["match_id"] == match_id) & (scored["thrower_id"] == player_id)]
    pf = _with_lineups(mine, flash_lineups).sort_values(["round", "throw_uid"]).reset_index(drop=True)

    blind_sum = float(pf["enemy_blind_s"].sum())
    summary = {
        "n_flashes": int(len(pf)),
        "enemy_blind_s": _r1(blind_sum),
        "teamflashes": int(pf["teamflash"].astype(bool).sum()),
        "flash_kills": int(pf["flash_kill"].astype(bool).sum()),
        "enemy_blind_pct": _percentile(player_match, blind_sum),
    }

    rows = _lineup_rows(lineups)
    cache: dict[str, dict | None] = {}
    candidates = []
    for flash in pf.to_dict("records"):
        lid = flash.get("lineup_id")
        if not _valid_lid(lid):
            continue
        rec = _recommendation(rows, lineups, lid, cache)
        if rec is not None:
            candidates.append((rec, flash))
    candidates.sort(key=lambda c: (-c[0]["gain"], int(c[1]["round"]), int(c[1]["throw_uid"])))

    picked: list[dict] = []
    seen: set[str] = set()
    for rec, flash in candidates:
        lid = flash["lineup_id"]
        if lid in seen:
            continue
        seen.add(lid)
        times = int((pf["lineup_id"] == lid).sum())
        picked.append(_finding(len(picked) + 1, flash, rec, glossary, times))
        if len(picked) >= max_findings:
            break

    return {
        "card_id": alias.split()[-1],
        "player": alias,
        "rating": _rating(players, match_id, player_id),
        "match_summary": summary,
        "findings": picked,
    }


def run_findings(data: Path = DATA) -> list[str]:
    data = Path(data)
    scored = pd.read_parquet(data / "flash_scored.parquet", columns=SCORED_COLUMNS)
    flash_lineups = pd.read_parquet(data / "flash_lineups.parquet", columns=["throw_uid", "lineup_id", "mode"])
    lineups = pd.read_parquet(data / "lineups.parquet")
    player_match = pd.read_parquet(data / "player_match.parquet", columns=["enemy_blind_s_sum"])
    players = pd.read_parquet(data / "players.parquet", columns=["match_id", "player_id", "rating"])
    glossary = load_glossary(data)

    pairs = select_cards(scored, flash_lineups, lineups, n_cards=len(CARD_LETTERS))
    out_dir = data / "findings"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()

    card_ids: list[str] = []
    for letter, (match_id, player_id) in zip(CARD_LETTERS, pairs):
        card = build_findings(match_id, player_id, f"Игрок {letter}", scored, flash_lineups, lineups,
                              player_match, players, glossary)
        (out_dir / f"{letter}.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        card_ids.append(letter)
    print(f"[findings] карточек: {len(card_ids)}")
    if len(card_ids) < 3:
        print("[findings] ВНИМАНИЕ: меньше 3 карточек со значимыми находками")
    return card_ids
