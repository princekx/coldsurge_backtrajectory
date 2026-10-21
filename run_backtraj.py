#!/usr/bin/env python
"""
Run NAME-style ensemble back trajectories from Peninsular Malaysia using ERA5 3-D winds.

Examples
--------
# Single date, default ensemble (5 x 5 grid over Peninsular Malaysia x 825/850/875 hPa), 10 days back
python run_backtraj.py --start-time 2006-12-16

# Several dates from a file (one date/time per line), 4 stochastic members per release point
python run_backtraj.py --dates-file cs_onsets.txt --members 4 --csv

# Pure (deterministic) trajectories, all levels in one file per variable
python run_backtraj.py --start-time 2006-12-16T12 --no-turbulence --no-meso \
    --data-dir /data/scratch/prince.xavier/SEAPy_climate/obs \
    --file-pattern 'ERA5_Plevels_{var}_daily.{yyyymm}.nc' --var-token u=U --var-token v=V --var-token w=OMEGA
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from era5_met import DEFAULT_DATA_DIR, DEFAULT_FILE_PATTERN, DEFAULT_LEVELS, MetField  # noqa: E402
from lagrangian import BackTrajectoryModel, DispersionOptions, make_release, to_dataframe  # noqa: E402

DEFAULT_OUT_DIR = "/data/users/prince.xavier/SEAPy_climate/coldsurge_comps/ERA5/name_backtraj"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = ap.add_argument_group("release times")
    g.add_argument("--start-time", nargs="+", default=[], help="Release (arrival) time(s), e.g. 2006-12-16T00")
    g.add_argument("--dates-file", help="Text file with one release date/time per line ('#' comments allowed)")
    g.add_argument("--days-back", type=float, default=10.0, help="Length of back trajectories (days)")

    g = ap.add_argument_group("release ensemble (default: Peninsular Malaysia at 850 hPa)")
    g.add_argument("--lat-range", nargs=2, type=float, default=[1.5, 6.5])
    g.add_argument("--lon-range", nargs=2, type=float, default=[100.5, 104.0])
    g.add_argument("--pressures", nargs="+", type=float, default=[825.0, 850.0, 875.0],
                   help="Release pressure levels in hPa (random mode uses their min/max as a range)")
    g.add_argument("--n-lat", type=int, default=5)
    g.add_argument("--n-lon", type=int, default=5)
    g.add_argument("--members", type=int, default=1, help="Stochastic members per release point")
    g.add_argument("--release-mode", choices=["grid", "random"], default="grid")
    g.add_argument("--n-random", type=int, default=100, help="Number of random release points")

    g = ap.add_argument_group("ERA5 met data")
    g.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    g.add_argument("--file-pattern", default=DEFAULT_FILE_PATTERN,
                   help="File pattern with {var}, {yyyymm} and optionally {level} placeholders")
    g.add_argument("--levels", nargs="+", type=float, default=list(DEFAULT_LEVELS),
                   help="Pressure levels (hPa) to read when files are split by {level}")
    g.add_argument("--var-token", action="append", default=[], metavar="KEY=TOKEN",
                   help="Override the {var} file token for u, v, w or t (e.g. w=OMEGA)")
    g.add_argument("--domain", nargs=4, type=float, default=[30.0, 210.0, -30.0, 90.0],
                   metavar=("LON0", "LON1", "LAT0", "LAT1"), help="Met sub-domain to load (degrees)")
    g.add_argument("--global-domain", action="store_true", help="Load the full horizontal domain")
    g.add_argument("--p-min", type=float, default=100.0, help="Top pressure level to load (hPa)")
    g.add_argument("--p-max", type=float, default=1000.0, help="Bottom pressure level to load (hPa)")
    g.add_argument("--no-temperature", action="store_true", help="Do not read T (use --t-ref for hydrostatics)")
    g.add_argument("--time-offset-hours", type=float, default=0.0,
                   help="Shift met time stamps, e.g. 12 to centre daily means at midday")

    g = ap.add_argument_group("NAME dispersion options")
    g.add_argument("--no-turbulence", action="store_true", help="Switch off free-tropospheric turbulence")
    g.add_argument("--no-meso", action="store_true", help="Switch off unresolved mesoscale motions")
    g.add_argument("--no-damping", action="store_true",
                   help="Switch off NAME damping of the random walks at short travel times (on by default)")
    g.add_argument("--free-trop-sigu", type=float, default=0.25)
    g.add_argument("--free-trop-sigw", type=float, default=0.1)
    g.add_argument("--free-trop-tauu", type=float, default=300.0)
    g.add_argument("--free-trop-tauw", type=float, default=100.0)
    g.add_argument("--meso-sigu", type=float, default=0.9)
    g.add_argument("--meso-tauu", type=float, default=9600.0)
    g.add_argument("--t-ref", type=float, default=280.0, help="Reference temperature (K) if T not available")

    g = ap.add_argument_group("numerics and output")
    g.add_argument("--dt-minutes", type=float, default=15.0)
    g.add_argument("--output-minutes", type=float, default=60.0)
    g.add_argument("--scheme", choices=["rk2", "euler"], default="rk2")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    g.add_argument("--csv", action="store_true", help="Also write a long-format CSV like trajectory.py")
    g.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def read_times(args):
    times = list(args.start_time)
    if args.dates_file:
        with open(args.dates_file) as fh:
            times += [ln.split("#")[0].strip() for ln in fh if ln.split("#")[0].strip()]
    if not times:
        raise SystemExit("Provide --start-time and/or --dates-file")
    return [pd.Timestamp(pd.to_datetime(t)) for t in times]


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    var_tokens = dict(kv.split("=", 1) for kv in args.var_token)
    options = DispersionOptions(
        turbulence=not args.no_turbulence, mesoscale_motions=not args.no_meso, damping=not args.no_damping,
        free_trop_sigu=args.free_trop_sigu, free_trop_sigw=args.free_trop_sigw,
        free_trop_tauu=args.free_trop_tauu, free_trop_tauw=args.free_trop_tauw,
        meso_sigu=args.meso_sigu, meso_tauu=args.meso_tauu, t_ref_K=args.t_ref)

    for t_rel in read_times(args):
        stem = os.path.join(args.out_dir, f"name_backtraj_{t_rel:%Y%m%d%H}")
        if os.path.exists(stem + ".nc") and not args.overwrite:
            print(f"[skip] {stem}.nc exists")
            continue

        tic = time.time()
        met = MetField.from_era5(
            t_rel - pd.Timedelta(days=args.days_back), t_rel, data_dir=args.data_dir,
            file_pattern=args.file_pattern, levels=args.levels, var_tokens=var_tokens,
            domain=None if args.global_domain else args.domain, p_min=args.p_min, p_max=args.p_max,
            use_temperature=not args.no_temperature, time_offset_hours=args.time_offset_hours)
        print(f"[met] {t_rel:%Y-%m-%d %H}: {met.times[0]} -> {met.times[-1]}, levels {met.p.tolist()}, "
              f"grid {met.lat.size}x{met.lon.size}, T={'yes' if met.has_temperature else 'no'} "
              f"({time.time() - tic:.1f}s)")

        release = make_release(t_rel, lat_range=args.lat_range, lon_range=args.lon_range,
                               pressures_hPa=args.pressures, n_lat=args.n_lat, n_lon=args.n_lon,
                               members=args.members, mode=args.release_mode, n_random=args.n_random,
                               seed=args.seed)
        model = BackTrajectoryModel(met, options, dt_minutes=args.dt_minutes, output_minutes=args.output_minutes,
                                    scheme=args.scheme, seed=args.seed)
        ds = model.run(release, days_back=args.days_back)
        ds.attrs["met_data_dir"] = args.data_dir
        ds.attrs["met_file_pattern"] = args.file_pattern

        ds.to_netcdf(stem + ".nc")
        if args.csv:
            to_dataframe(ds).to_csv(stem + ".csv", index=False)
        complete = int(ds["lat"].isel(step=-1).notnull().sum())
        print(f"[done] {len(release)} particles ({complete} full length) -> {stem}.nc "
              f"({time.time() - tic:.1f}s)")


if __name__ == "__main__":
    main()
