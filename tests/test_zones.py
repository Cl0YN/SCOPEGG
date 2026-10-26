import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utility_coach import zones
from utility_coach.ingest import SCHEMAS
from utility_coach.zones import (assign_zones, fit_zones, load_centers, load_glossary, plot_zone_map,
                                 run_zones, save_zones)

BLOBS = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]])


def _blobs(n=300, seed=0, centers=BLOBS, spread=30.0):
    rng = np.random.default_rng(seed)
    pts = np.concatenate([c + rng.normal(0, spread, size=(n, 3)) for c in centers])
    labels = np.repeat(np.arange(len(centers)), n)
    return pts, labels


def test_fit_zones_finds_blobs():
    pts, _ = _blobs()
    centers = fit_zones(pts, n_zones=3, seed=1)
    assert centers.shape == (3, 3)
    for c in BLOBS:
        assert np.min(np.linalg.norm(centers - c, axis=1)) < 15


def test_fit_zones_is_deterministic():
    pts, _ = _blobs()
    a = fit_zones(pts, n_zones=3, seed=7)
    b = fit_zones(pts, n_zones=3, seed=7)
    np.testing.assert_allclose(a, b)


def test_assign_zones_stable_within_blob():
    pts, labels = _blobs()
    centers = fit_zones(pts, n_zones=3, seed=1)
    z = assign_zones(centers, pts)
    assert z.dtype == np.int16 and z.shape == (len(pts),)
    for k in range(3):
        assert len(set(z[labels == k].tolist())) == 1
    assert len(set(z.tolist())) == 3
    np.testing.assert_array_equal(assign_zones(centers, pts), z)


def test_z_scale_separates_levels():
    # Одна точка XY, два уровня по Z (разница 120): z_scale=3 даёт 360 юнитов — больше разброса XY.
    centers_in = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 120.0]])
    pts, labels = _blobs(centers=centers_in, spread=20.0)
    pts[:, 2] = np.where(labels == 0, 0.0, 120.0) + np.random.default_rng(3).normal(0, 5, len(pts))
    centers = fit_zones(pts, n_zones=2, seed=1, z_scale=3.0)
    z = assign_zones(centers, pts, z_scale=3.0)
    assert len(set(z[labels == 0].tolist())) == 1 and len(set(z[labels == 1].tolist())) == 1
    assert z[labels == 0][0] != z[labels == 1][0]


def test_save_load_roundtrip(tmp_path):
    centers = np.array([[1.234, -5.5, 1600.0], [300.0, 200.0, 1700.25]])
    path = tmp_path / "zones.json"
    save_zones(centers, path, z_scale=3.0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["z_scale"] == 3.0 and len(payload["centers"]) == 2
    np.testing.assert_allclose(load_centers(path), centers, atol=0.01)


def test_load_glossary_fills_unlabeled(tmp_path):
    save_zones(BLOBS, tmp_path / "zones.json")
    callouts = tmp_path / "callouts.json"
    callouts.write_text(json.dumps({"1": "Mid", "2": ""}, ensure_ascii=False), encoding="utf-8")
    assert load_glossary(tmp_path, callouts=callouts) == {0: "Зона 0", 1: "Mid", 2: "Зона 2"}


def test_load_glossary_default_callouts_file_is_valid_json():
    names = json.loads(zones.CALLOUTS.read_text(encoding="utf-8"))
    assert isinstance(names, dict)
    assert all(k.isdigit() and isinstance(v, str) for k, v in names.items())


def _synthetic_throws(n_per_blob=200):
    pts, labels = _blobs(n=n_per_blob)
    to = pts + 50.0
    n = len(pts)
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "throw_uid": np.arange(n, dtype=np.int64),
        "match_id": np.repeat([1, 2], n // 2).astype(np.int64),
        "grenade_type": "flash",
        "projectile_id": np.arange(n, dtype=np.int64),
        "round": (np.arange(n) % 5).astype(np.int16),
        "thrower_id": (labels + 10).astype(np.int64),
        "team": np.where(labels == 0, 2, 3).astype(np.int8),
        "start_tick": rng.integers(1000, 3000, n).astype(np.int32),
        "end_tick": rng.integers(3000, 3100, n).astype(np.int32),
        "throw_x": pts[:, 0].astype(np.float32), "throw_y": pts[:, 1].astype(np.float32),
        "throw_z": pts[:, 2].astype(np.float32),
        "yaw": np.zeros(n, np.float32), "pitch": np.zeros(n, np.float32),
        "vel_x": np.zeros(n, np.float32), "vel_y": np.zeros(n, np.float32), "vel_z": np.zeros(n, np.float32),
        "to_x": to[:, 0].astype(np.float32), "to_y": to[:, 1].astype(np.float32),
        "to_z": to[:, 2].astype(np.float32),
        "crouch": False, "airborne": False, "inferno_ok": True,
    })


def test_plot_zone_map_writes_png(tmp_path):
    throws = _synthetic_throws()
    out = tmp_path / "fig" / "zone_map.png"
    plot_zone_map(throws, BLOBS, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_run_zones_end_to_end(tmp_path):
    throws = _synthetic_throws()
    pq.write_table(pa.Table.from_pandas(throws, schema=SCHEMAS["throws"], preserve_index=False),
                   tmp_path / "throws.parquet")
    run_zones(tmp_path, n_zones=4)
    assert load_centers(tmp_path / "zones.json").shape == (4, 3)
    assert (tmp_path / "figures" / "zone_map.png").exists()
