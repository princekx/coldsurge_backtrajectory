#!/usr/bin/env python
"""
Download ERA5 pressure-level u, v, omega and T from the Copernicus CDS for back trajectories.

One file per variable and month, all levels, named so that run_backtraj.py can read them with
    --data-dir OUT_DIR --file-pattern 'ERA5_Plevels_{var}_6hr.{yyyymm}.nc'

With --combined, all variables go in one request/file (much less CDS queueing):
    ERA5_Plevels_uvwt_6hr.YYYYMM.nc, read with
    --file-pattern 'ERA5_Plevels_{var}_6hr.{yyyymm}.nc'
    --var-token u=uvwt --var-token v=uvwt --var-token w=uvwt --var-token t=uvwt

Needs a CDS key in ~/.cdsapirc.

Example (10-day back trajectories arriving 2025-12-13):
    python download_era5_cds.py --start 2025-12-01 --end 2025-12-14
"""
import argparse
import os

import pandas as pd

VARIABLES = ["u_component_of_wind", "v_component_of_wind", "vertical_velocity", "temperature"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]
DEFAULT_OUT = "/data/scratch/prince.xavier/SEAPy_climate/obs/ERA5_plevels_6hr"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="First day (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="Last day (YYYY-MM-DD)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--hours", nargs="+", default=["00:00", "06:00", "12:00", "18:00"])
    ap.add_argument("--levels", nargs="+", type=int, default=LEVELS)
    ap.add_argument("--variables", nargs="+", default=VARIABLES)
    ap.add_argument("--area", nargs=4, type=float, default=[90, 30, -30, 210], metavar=("N", "W", "S", "E"))
    ap.add_argument("--grid", type=float, default=0.5, help="Output grid spacing (deg)")
    ap.add_argument("--combined", action="store_true", help="One request/file for all variables")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    import cdsapi

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    days = pd.date_range(args.start, args.end, freq="D")
    client = cdsapi.Client()
    for (year, month), grp in days.to_series().groupby([days.year, days.month]):
        groups = [("uvwt", args.variables)] if args.combined else [(v, [v]) for v in args.variables]
        for token, var in groups:
            out = os.path.join(args.out_dir, f"ERA5_Plevels_{token}_6hr.{year:04d}{month:02d}.nc")
            if os.path.exists(out) and not args.overwrite:
                print(f"exists, skipping: {out}")
                continue
            request = {
                "product_type": ["reanalysis"],
                "variable": var,
                "pressure_level": [str(p) for p in args.levels],
                "year": [f"{year:04d}"],
                "month": [f"{month:02d}"],
                "day": [f"{d:02d}" for d in grp.index.day],
                "time": args.hours,
                "area": args.area,
                "grid": [args.grid, args.grid],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            print(f"requesting {token} {year}-{month:02d} -> {out}", flush=True)
            client.retrieve("reanalysis-era5-pressure-levels", request, out)


if __name__ == "__main__":
    main()
