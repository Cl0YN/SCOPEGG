"""Зоны карты: KMeans по точкам броска и приземления + глоссарий каллаутов."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from utility_coach.config import DATA, N_ZONES, SEED, TEAM_CT, TEAM_T, TICK_RATE  # noqa: E402

CALLOUTS = Path(__file__).resolve().parent / "zone_callouts.json"
MAX_FIT_POINTS = 500_000
EARLY_ROUND_S = 10.0


def _scaled(xyz: np.ndarray, z_scale: float) -> np.ndarray:
    a = np.asarray(xyz, dtype=np.float64).reshape(-1, 3).copy()
    a[:, 2] *= z_scale
    return a


def fit_zones(points: np.ndarray, n_zones: int = N_ZONES, seed: int = SEED, z_scale: float = 3.0) -> np.ndarray:
    """KMeans по [x, y, z*z_scale]; центры (k,3) в исходных координатах, отсортированы по (x, y)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > MAX_FIT_POINTS:
        idx = np.random.default_rng(seed).choice(len(pts), MAX_FIT_POINTS, replace=False)
        pts = pts[idx]
    km = KMeans(n_clusters=n_zones, n_init=4, random_state=seed).fit(_scaled(pts, z_scale))
    centers = km.cluster_centers_.copy()
    centers[:, 2] /= z_scale
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return centers[order]


def assign_zones(centers: np.ndarray, xyz: np.ndarray, z_scale: float = 3.0) -> np.ndarray:
    """Номер ближайшего центра в масштабированном пространстве, int16 (n,)."""
    c = _scaled(centers, z_scale)
    a = _scaled(xyz, z_scale)
    out = np.empty(len(a), dtype=np.int16)
    step = 200_000
    for i in range(0, len(a), step):
        chunk = a[i:i + step]
        d2 = ((chunk[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        out[i:i + step] = d2.argmin(axis=1)
    return out


def save_zones(centers: np.ndarray, path: Path, z_scale: float = 3.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"centers": [[round(float(v), 2) for v in row] for row in np.asarray(centers)],
               "z_scale": float(z_scale)}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_centers(path: Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(payload["centers"], dtype=np.float64).reshape(-1, 3)


def load_glossary(data: Path = DATA, callouts: Path = CALLOUTS) -> dict[int, str]:
    """id -> каллаут из zone_callouts.json или "Зона {id}" для неподписанных."""
    zones_path = Path(data) / "zones.json"
    k = len(load_centers(zones_path)) if zones_path.exists() else N_ZONES
    names = {}
    if Path(callouts).exists():
        names = json.loads(Path(callouts).read_text(encoding="utf-8"))
    out = {}
    for i in range(k):
        name = str(names.get(str(i), "")).strip()
        out[i] = name if name else f"Зона {i}"
    return out


def plot_zone_map(throws: pd.DataFrame, centers: np.ndarray, out_png: Path) -> None:
    """Плотность приземлений (hexbin, log), центры с номерами, броски T/CT в первые 10 с раунда."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    xs = np.concatenate([throws["to_x"].to_numpy(), throws["throw_x"].to_numpy()])
    ys = np.concatenate([throws["to_y"].to_numpy(), throws["throw_y"].to_numpy()])
    x0, x1 = np.nanpercentile(xs, [0.2, 99.8])
    y0, y1 = np.nanpercentile(ys, [0.2, 99.8])
    pad = 0.04 * max(x1 - x0, y1 - y0, 1.0)

    fig, ax = plt.subplots(figsize=(14, 12), dpi=110)
    ax.set_facecolor("#111111")
    hb = ax.hexbin(throws["to_x"], throws["to_y"], gridsize=220, bins="log", cmap="magma", mincnt=1,
                   linewidths=0, extent=(x0 - pad, x1 + pad, y0 - pad, y1 + pad))
    fig.colorbar(hb, ax=ax, shrink=0.6, label="число приземлений (лог. шкала)")

    start = throws.groupby(["match_id", "round"])["start_tick"].transform("min")
    early = throws[(throws["start_tick"] - start) <= EARLY_ROUND_S * TICK_RATE]
    for team, color, label in ((TEAM_T, "#ffb000", "T: точки броска в первые 10 с раунда"),
                               (TEAM_CT, "#33a8ff", "CT: точки броска в первые 10 с раунда")):
        pts = early[early["team"] == team]
        if len(pts) > 20_000:
            pts = pts.sample(20_000, random_state=SEED)
        ax.scatter(pts["throw_x"], pts["throw_y"], s=3, c=color, alpha=0.5, label=label, linewidths=0)

    halo = [patheffects.withStroke(linewidth=3, foreground="black")]
    c = np.asarray(centers)
    ax.scatter(c[:, 0], c[:, 1], s=40, marker="x", c="white", linewidths=1.5)
    for i, (x, y, _z) in enumerate(c):
        ax.text(x, y + 0.012 * (y1 - y0), str(i), color="white", fontsize=13, fontweight="bold",
                ha="center", va="bottom", path_effects=halo)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.set_title("Cache: плотность приземлений гранат, центры зон (номера), ранние броски T/CT")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="lower right", markerscale=5, facecolor="#222222", labelcolor="white")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def run_zones(data: Path = DATA, n_zones: int = N_ZONES) -> None:
    data = Path(data)
    cols = ["match_id", "round", "team", "start_tick", "throw_x", "throw_y", "throw_z",
            "to_x", "to_y", "to_z"]
    throws = pq.read_table(data / "throws.parquet", columns=cols).to_pandas()
    points = np.concatenate([
        throws[["throw_x", "throw_y", "throw_z"]].to_numpy(np.float32),
        throws[["to_x", "to_y", "to_z"]].to_numpy(np.float32),
    ])
    centers = fit_zones(points, n_zones=n_zones)
    save_zones(centers, data / "zones.json")
    plot_zone_map(throws, centers, data / "figures" / "zone_map.png")
    sizes = np.bincount(assign_zones(centers, throws[["to_x", "to_y", "to_z"]].to_numpy()),
                        minlength=len(centers))
    print(f"zones: {len(centers)} centers -> {data / 'zones.json'}; landings per zone: {sizes.tolist()}")
