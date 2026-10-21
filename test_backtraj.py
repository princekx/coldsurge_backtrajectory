"""
Tests for the NAME-style back-trajectory model using synthetic wind fields.

Run with:  module load scitools/default-current; python -m pytest -q test_backtraj.py
"""

import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from era5_met import MetField  # noqa: E402
from lagrangian import (R_EARTH, BackTrajectoryModel, DispersionOptions, great_circle_move,  # noqa: E402
                        make_release, to_dataframe)
import run_backtraj  # noqa: E402

DETERMINISTIC = DispersionOptions(turbulence=False, mesoscale_motions=False)
T_REL = pd.Timestamp("2006-12-16")


def uniform_met(u=0.0, v=0.0, w=0.0, days=(-12, 2), lat=(-60, 80), lon=None, levels=(1000, 850, 500, 200)):
    times = pd.date_range(T_REL + pd.Timedelta(days=days[0]), T_REL + pd.Timedelta(days=days[1]), freq="6h")
    lats = np.arange(lat[0], lat[1] + 0.1, 1.0)
    lons = np.arange(0, 360, 1.0) if lon is None else np.arange(lon[0], lon[1] + 0.1, 1.0)
    shape = (times.size, len(levels), lats.size, lons.size)
    return MetField(times, levels, lats, lons, np.full(shape, u), np.full(shape, v), np.full(shape, w))


def single_release(lat=4.0, lon=102.0, p=850.0):
    return make_release(T_REL, lat_range=(lat, lat), lon_range=(lon, lon), pressures_hPa=[p], n_lat=1, n_lon=1)


def test_great_circle_matches_small_displacement():
    lat, lon = great_circle_move(np.array([10.0]), np.array([100.0]), np.array([1000.0]), np.array([2000.0]))
    assert lat[0] == pytest.approx(10.0 + np.rad2deg(2000.0 / R_EARTH), abs=1e-6)
    assert lon[0] == pytest.approx(100.0 + np.rad2deg(1000.0 / (R_EARTH * np.cos(np.deg2rad(10)))), abs=1e-5)


def test_great_circle_over_pole():
    lat, lon = great_circle_move(np.array([89.0]), np.array([0.0]), np.array([0.0]), np.array([2 * np.pi * R_EARTH / 180]))
    assert lat[0] == pytest.approx(89.0, abs=1e-6)
    assert lon[0] == pytest.approx(180.0, abs=1e-6)


@pytest.mark.parametrize("scheme", ["euler", "rk2"])
def test_uniform_zonal_wind_backward(scheme):
    met = uniform_met(u=10.0)
    ds = BackTrajectoryModel(met, DETERMINISTIC, scheme=scheme).run(single_release(lat=0.0), days_back=1)
    expected = 102.0 - np.rad2deg(10.0 * 86400.0 / R_EARTH)
    assert ds["lon"].values[0, -1] == pytest.approx(expected, abs=1e-6)
    assert ds["lat"].values[0, -1] == pytest.approx(0.0, abs=1e-9)
    assert ds["pressure"].values[0, -1] == pytest.approx(850.0)
    assert ds["hours_back"].values[-1] == pytest.approx(24.0)


def test_uniform_northerly_comes_from_north():
    met = uniform_met(v=-5.0)
    ds = BackTrajectoryModel(met, DETERMINISTIC).run(single_release(), days_back=2)
    expected = 4.0 + np.rad2deg(5.0 * 2 * 86400.0 / R_EARTH)
    assert ds["lat"].values[0, -1] == pytest.approx(expected, abs=1e-6)


def test_omega_and_reflection():
    # Rising air (omega < 0) at arrival means parcels came from higher pressure (below).
    met = uniform_met(w=-0.2)
    ds = BackTrajectoryModel(met, DETERMINISTIC).run(single_release(), days_back=1)
    p = ds["pressure"].values[0]
    assert p[6] == pytest.approx(850.0 + 0.2 / 100.0 * 6 * 3600.0)
    assert np.nanmax(p) <= 1000.0
    # Parcels reach 1000 hPa after 18 h, then stay within one step (|w| dt = 1.8 hPa) of the reflecting floor.
    assert p[-1] >= 1000.0 - 0.2 / 100.0 * 900.0 - 1e-9


def test_rk2_more_accurate_in_solid_body_rotation():
    # Solid-body rotation around (0N, 100E) in a local tangent plane: exact path is a circle.
    times = pd.date_range(T_REL - pd.Timedelta(days=3), T_REL + pd.Timedelta(days=1), freq="3h")
    lats = np.arange(-30, 30.1, 0.25)
    lons = np.arange(70, 130.1, 0.25)
    omega_rot = 2 * np.pi / (2 * 86400.0)
    Y, X = np.meshgrid(np.deg2rad(lats) * R_EARTH, np.deg2rad(lons - 100.0) * R_EARTH, indexing="ij")
    u2d, v2d = -omega_rot * Y, omega_rot * X
    shape = (times.size, 2, lats.size, lons.size)
    u = np.broadcast_to(u2d, shape)
    v = np.broadcast_to(v2d, shape)
    met = MetField(times, [1000, 500], lats, lons, u, v, np.zeros(shape))
    rel = single_release(lat=5.0, lon=100.0)
    errs = {}
    for scheme in ("euler", "rk2"):
        ds = BackTrajectoryModel(met, DETERMINISTIC, dt_minutes=30, scheme=scheme).run(rel, days_back=2)
        r = np.hypot(ds["lat"].values[0], ds["lon"].values[0] - 100.0)
        errs[scheme] = np.nanmax(np.abs(r - 5.0))
    assert errs["rk2"] < 0.1 * errs["euler"]
    assert errs["rk2"] < 0.05


def test_mesoscale_spread_matches_name_diffusivity():
    met = uniform_met()
    rel = make_release(T_REL, lat_range=(0, 0), lon_range=(100, 100), pressures_hPa=[500], n_lat=1, n_lon=1,
                       members=2000)
    T = 86400.0
    for damping in (False, True):
        opt = DispersionOptions(turbulence=False, mesoscale_motions=True, damping=damping)
        ds = BackTrajectoryModel(met, opt, seed=1).run(rel, days_back=1)
        dy = np.deg2rad(ds["lat"].values[:, -1]) * R_EARTH
        k = 2 * opt.meso_sigu ** 2 * opt.meso_tauu
        tau = opt.meso_tauu
        # NAME damping: variance rate multiplied by (1 - exp(-t/tau)), integrated over travel time
        expected_var = k * (T - tau * (1 - np.exp(-T / tau))) if damping else k * T
        assert np.var(dy) == pytest.approx(expected_var, rel=0.1)


def test_vertical_turbulence_in_pressure():
    met = uniform_met()
    opt = DispersionOptions(turbulence=True, mesoscale_motions=False, damping=False)
    rel = make_release(T_REL, lat_range=(0, 0), lon_range=(100, 100), pressures_hPa=[500], n_lat=1, n_lon=1,
                       members=2000)
    ds = BackTrajectoryModel(met, opt, seed=2).run(rel, days_back=0.5)
    rho_g = 500 * 100.0 / (287.05 * opt.t_ref_K) * 9.80665
    expected_sd = rho_g * np.sqrt(2 * opt.free_trop_sigw ** 2 * opt.free_trop_tauw * 43200.0) / 100.0
    assert np.std(ds["pressure"].values[:, -1]) == pytest.approx(expected_sd, rel=0.1)


def test_particles_leaving_regional_domain_are_terminated():
    met = uniform_met(u=20.0, lon=(90, 110))
    ds = BackTrajectoryModel(met, DETERMINISTIC).run(single_release(), days_back=5)
    lon = ds["lon"].values[0]
    assert np.isnan(lon[-1])
    assert np.nanmin(lon) >= 90.0


def test_dateline_and_greenwich_wrap():
    met = uniform_met(u=-30.0)  # easterlies: parcels came from the east, crossing 180 and 360
    ds = BackTrajectoryModel(met, DETERMINISTIC).run(single_release(lat=0.0, lon=350.0), days_back=2)
    assert np.all(np.isfinite(ds["lon"].values))
    assert ds["lon"].values[0, -1] == pytest.approx((350.0 + np.rad2deg(30 * 2 * 86400 / R_EARTH)) % 360, abs=1e-6)


def test_release_ensemble_and_dataframe():
    rel = make_release(["2006-12-16", "2006-12-17"], n_lat=3, n_lon=4, pressures_hPa=[825, 850, 875], members=2)
    assert len(rel) == 2 * 3 * 4 * 3 * 2
    assert rel["release_pressure"].between(825, 875).all()
    met = uniform_met(u=5.0, days=(-3, 3))
    ds = BackTrajectoryModel(met, DETERMINISTIC).run(rel, days_back=1)
    df = to_dataframe(ds)
    assert {"time", "hours_back", "pressure_hPa", "lat_deg", "lon_deg", "u_ms", "v_ms", "omega_Pas",
            "path_label"} <= set(df.columns)
    last = df[df.hours_back == 24]
    assert (last["time"] == last["release_time"] - pd.Timedelta(hours=24)).all()


# ---------------------------------------------------------------------------
# End-to-end with ERA5-like files on disk
# ---------------------------------------------------------------------------

def _write_era5_like(tmp_path, zipped_temperature=True):
    """Write ERA5 CDS-style monthly files, one file per variable and pressure level."""
    times = pd.date_range("2006-11-01", "2006-12-31", freq="D")
    lats = np.arange(90, -30.1, -1.0)
    lons = np.arange(-180, 180, 1.0)
    for level in (1000, 850, 500, 200):
        for token, name, value in (("u_component_of_wind", "u", -8.0), ("v_component_of_wind", "v", -4.0),
                                   ("vertical_velocity", "w", 0.01), ("temperature", "t", 270.0)):
            for ym, tsel in times.to_series().groupby(times.strftime("%Y%m")):
                data = np.full((tsel.size, 1, lats.size, lons.size), value, dtype=np.float32)
                ds = xr.Dataset({name: (("valid_time", "pressure_level", "latitude", "longitude"), data)},
                                coords=dict(valid_time=tsel.values, pressure_level=[float(level)],
                                            latitude=lats, longitude=lons, number=0))
                fname = tmp_path / f"ERA5_Plevels_{token}_{level}hPa_daily.{ym}.nc"
                if name == "t" and zipped_temperature:
                    inner = tmp_path / f"inner_{level}_{ym}.nc"
                    ds.to_netcdf(inner)
                    with zipfile.ZipFile(fname, "w") as zf:
                        zf.write(inner, arcname="data_stream-oper.nc")
                    inner.unlink()
                else:
                    ds.to_netcdf(fname)


def test_load_era5_files_and_cli(tmp_path):
    _write_era5_like(tmp_path)
    met = MetField.from_era5("2006-12-06", "2006-12-16", data_dir=str(tmp_path), domain=(30, 210, -30, 90))
    assert met.p.tolist() == [200.0, 500.0, 850.0, 1000.0]
    assert met.has_temperature
    assert met.lon.min() >= 30 and met.lon.max() <= 210
    s = met.sample(met.to_seconds("2006-12-10"), np.array([850.0]), np.array([4.0]), np.array([102.0]))
    assert s[0].tolist() == pytest.approx([-8.0, -4.0, 0.01, 270.0])

    out = tmp_path / "out"
    run_backtraj.main(["--start-time", "2006-12-16", "--data-dir", str(tmp_path), "--out-dir", str(out),
                       "--members", "2", "--seed", "0", "--csv"])
    ds = xr.open_dataset(out / "name_backtraj_2006121600.nc")
    assert ds.sizes["particle"] == 5 * 5 * 3 * 2
    assert ds.sizes["step"] == 241
    # North-easterly flow: parcels originate to the north-east of Malaysia.
    assert float(ds["lat"].isel(step=-1).mean()) > 20.0
    assert float(ds["lon"].isel(step=-1).mean()) > 110.0
    assert (out / "name_backtraj_2006121600.csv").exists()


def test_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MetField.from_era5("2006-12-06", "2006-12-16", data_dir=str(tmp_path))
