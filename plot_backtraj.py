#!/usr/bin/env python
"""
Plot NAME-style ensemble back trajectories (output of run_backtraj.py).

python plot_backtraj.py /path/name_backtraj_2006121600.nc -o traj.png
"""

import argparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection


def plot_map(ds, ax=None, extent=(60, 180, -15, 60), cmap="viridis_r", vmin=200, vmax=1000, lw=0.6,
             show_mean=True):
    """Map of trajectories coloured by pressure, with the ensemble-mean path."""
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    pc = ccrs.PlateCarree()
    ax.set_extent(extent, crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="0.93")
    ax.coastlines(resolution="50m", linewidth=0.7)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = gl.right_labels = False

    norm = plt.Normalize(vmin, vmax)
    lc = None
    for i in range(ds.sizes["particle"]):
        lon = np.unwrap(np.deg2rad(ds["lon"].values[i]))
        lon = np.rad2deg(lon)
        lat, p = ds["lat"].values[i], ds["pressure"].values[i]
        ok = np.isfinite(lat)
        if ok.sum() < 2:
            continue
        pts = np.column_stack([lon[ok], lat[ok]])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw, transform=pc)
        lc.set_array(p[ok][:-1])
        ax.add_collection(lc)

    if show_mean:
        lon_r = np.deg2rad(ds["lon"])
        mlon = np.rad2deg(np.arctan2(np.sin(lon_r).mean("particle"), np.cos(lon_r).mean("particle")))
        ax.plot(mlon, ds["lat"].mean("particle"), "k-", lw=2, transform=pc, label="ensemble mean")
        ax.legend(loc="lower left")

    ax.plot(ds["release_lon"], ds["release_lat"], "r.", ms=3, transform=pc)
    if lc is not None:
        plt.colorbar(lc, ax=ax, orientation="vertical", shrink=0.7, label="Pressure (hPa)")
    return ax


def plot_pressure(ds, ax=None):
    """Pressure against hours back for all particles with ensemble median and 10-90% range."""
    if ax is None:
        ax = plt.gca()
    h = ds["hours_back"].values / 24.0
    ax.plot(h, ds["pressure"].values.T, color="0.6", lw=0.3)
    q = ds["pressure"].quantile([0.1, 0.5, 0.9], dim="particle", skipna=True)
    ax.fill_between(h, q.sel(quantile=0.1), q.sel(quantile=0.9), color="C0", alpha=0.3, label="10-90%")
    ax.plot(h, q.sel(quantile=0.5), color="C0", lw=2, label="median")
    ax.set_xlabel("Days before arrival")
    ax.set_ylabel("Pressure (hPa)")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.legend()
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--extent", nargs=4, type=float, default=[60, 180, -15, 60])
    args = ap.parse_args()

    ds = xr.open_dataset(args.file)
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree(central_longitude=180))
    plot_map(ds, ax=ax1, extent=args.extent)
    ax2 = fig.add_subplot(1, 2, 2)
    plot_pressure(ds, ax=ax2)
    t0 = np.datetime_as_string(ds["release_time"].values[0], unit="h")
    fig.suptitle(f"NAME-style back trajectories arriving {t0} ({ds.sizes['particle']} particles)")
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
