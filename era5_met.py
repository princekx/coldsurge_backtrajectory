"""
ERA5 pressure-level met reader for the lightweight NAME-style back-trajectory model.

Loads 3-D winds (u, v, omega) and optionally temperature from monthly ERA5 netCDF
files, normalises coordinate names/units, subsets to the time window and domain
needed for a run, and provides fast vectorised 4-D interpolation
(time, pressure, latitude, longitude) for an ensemble of particles.

File naming is controlled by a pattern with the placeholders
    {var}     - file token of the variable (e.g. 'u_component_of_wind')
    {yyyymm}  - year and month (e.g. '200612')
    {level}   - pressure level in hPa (only for one-file-per-level datasets)
Wildcards ('*') are allowed. Examples:
    ERA5_Plevels_{var}_{level}hPa_daily.{yyyymm}.nc   (SEAPy_obs_data layout)
    ERA5_Plevels_{var}_daily.{yyyymm}.nc               (all levels in one file)
    ERA5_plev_{yyyymm}.nc                              (all variables in one file)
CDS downloads that are zip archives with a '.nc' suffix are unpacked automatically.
"""

import glob
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

DEFAULT_DATA_DIR = "/data/users/prince.xavier/SEAPy_climate/SEAPy_obs_data/ERA5"
DEFAULT_FILE_PATTERN = "ERA5_Plevels_{var}_{level}hPa_daily.{yyyymm}.nc"
DEFAULT_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200)

# Token used for {var} in file names (CDS long variable names by default).
DEFAULT_VAR_TOKENS = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "t": "temperature",
}

# Possible netCDF variable names / standard names for each field.
VAR_ALIASES = {
    "u": ("u", "ua", "U", "uwnd", "var131"),
    "v": ("v", "va", "V", "vwnd", "var132"),
    "w": ("w", "wap", "omega", "OMEGA", "W", "var135"),
    "t": ("t", "ta", "T", "air", "var130"),
}
STANDARD_NAMES = {
    "u": ("eastward_wind",),
    "v": ("northward_wind",),
    "w": ("lagrangian_tendency_of_air_pressure",),
    "t": ("air_temperature",),
}

COORD_ALIASES = {
    "time": ("valid_time", "time", "t", "date"),
    "pressure": ("pressure_level", "level", "plev", "isobaricInhPa", "air_pressure", "lev", "pressure"),
    "lat": ("latitude", "lat", "nav_lat"),
    "lon": ("longitude", "lon", "nav_lon"),
}


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def months_between(t_first, t_last):
    """List of 'YYYYMM' strings covering the closed interval [t_first, t_last]."""
    periods = pd.period_range(pd.Timestamp(t_first).to_period("M"), pd.Timestamp(t_last).to_period("M"), freq="M")
    return [p.strftime("%Y%m") for p in periods]


def _expand_zip(path, cache_dir):
    """Return netCDF paths inside a CDS zip archive (extracted once into cache_dir)."""
    target = os.path.join(cache_dir, os.path.basename(path) + ".unzipped")
    if not os.path.isdir(target):
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target)
    return sorted(glob.glob(os.path.join(target, "*.nc")))


def resolve_files(data_dir, pattern, var_token, yyyymm, level=None, cache_dir=None):
    """Find (and unzip if needed) files matching the pattern for one variable/month/level."""
    name = pattern.format(var=var_token, yyyymm=yyyymm, level=level if level is not None else "*")
    paths = sorted(glob.glob(os.path.join(data_dir, name)))
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), f"era5_unzip_{os.getuid()}")
    out = []
    for p in paths:
        out.extend(_expand_zip(p, cache_dir) if zipfile.is_zipfile(p) else [p])
    return out


# ---------------------------------------------------------------------------
# Dataset normalisation
# ---------------------------------------------------------------------------

def _find_name(names, aliases):
    lower = {n.lower(): n for n in names}
    for a in aliases:
        if a in names:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _find_var(ds, key):
    name = _find_name(list(ds.data_vars), VAR_ALIASES[key])
    if name is not None:
        return name
    for n, da in ds.data_vars.items():
        if da.attrs.get("standard_name") in STANDARD_NAMES[key]:
            return n
    return None


def normalise_dataarray(da, level=None):
    """Rename coords to (time, pressure, lat, lon), pressure in hPa, lon in [0, 360)."""
    rename = {}
    for std, aliases in COORD_ALIASES.items():
        name = _find_name(list(da.dims) + list(da.coords), aliases)
        if name is not None and name != std:
            rename[name] = std
    da = da.rename(rename)

    # Drop scalar/auxiliary coords such as 'number' or 'expver'.
    da = da.drop_vars([c for c in da.coords if c not in ("time", "pressure", "lat", "lon")])

    if "pressure" not in da.dims:
        if "pressure" in da.coords:
            da = da.expand_dims("pressure")
        else:
            if level is None:
                raise ValueError(f"No pressure coordinate in field '{da.name}' and no level given.")
            da = da.expand_dims(pressure=[float(level)])

    p = da["pressure"].values.astype(float)
    units = str(da["pressure"].attrs.get("units", "")).lower()
    if units in ("pa", "pascal") or np.nanmax(p) > 2000.0:
        p = p / 100.0
    da = da.assign_coords(pressure=p)

    extra = [d for d in da.dims if d not in ("time", "pressure", "lat", "lon")]
    for d in extra:
        if da.sizes[d] != 1:
            raise ValueError(f"Unexpected non-singleton dimension '{d}' in field '{da.name}'.")
    da = da.squeeze(extra, drop=True) if extra else da

    da = da.assign_coords(lon=np.mod(da["lon"].values.astype(float), 360.0))
    da = da.sortby("lon").sortby("lat").sortby("pressure")
    return da.transpose("time", "pressure", "lat", "lon")


def _subset(da, t0, t1, domain, p_min, p_max):
    da = da.sel(time=slice(t0, t1))
    da = da.sel(pressure=slice(p_min, p_max))
    if domain is not None:
        lon0, lon1, lat0, lat1 = domain
        lon0, lon1 = lon0 % 360.0, lon1 % 360.0
        da = da.sel(lat=slice(lat0, lat1))
        if lon0 <= lon1:
            da = da.sel(lon=slice(lon0, lon1))
        else:  # box crosses the 0/360 meridian
            da = xr.concat([da.sel(lon=slice(lon0, 360.0)), da.sel(lon=slice(0.0, lon1)).assign_coords(
                lon=lambda d: d["lon"] + 360.0)], dim="lon")
    return da


def load_era5_field(key, months, t0, t1, data_dir=DEFAULT_DATA_DIR, file_pattern=DEFAULT_FILE_PATTERN,
                    levels=DEFAULT_LEVELS, var_tokens=None, domain=None, p_min=0.0, p_max=1100.0,
                    time_offset_hours=0.0, required=True):
    """
    Load one ERA5 field ('u', 'v', 'w' or 't') as a (time, pressure, lat, lon) DataArray.

    Returns None if the field is not found and `required` is False.
    """
    var_tokens = {**DEFAULT_VAR_TOKENS, **(var_tokens or {})}
    token = var_tokens[key]
    per_level = "{level}" in file_pattern
    level_list = list(levels) if per_level else [None]

    pieces = []
    for lev in level_list:
        if lev is not None and not (p_min <= float(lev) <= p_max):
            continue
        month_pieces = []
        for ym in months:
            for path in resolve_files(data_dir, file_pattern, token, ym, lev):
                with xr.open_dataset(path) as ds:
                    name = _find_var(ds, key)
                    if name is None:
                        continue
                    da = normalise_dataarray(ds[name], level=lev)
                    tsel = t0 - pd.Timedelta(hours=time_offset_hours)
                    tsel1 = t1 - pd.Timedelta(hours=time_offset_hours)
                    da = _subset(da, tsel, tsel1, domain, p_min, p_max).load()
                if da.sizes["time"] > 0 and da.sizes["pressure"] > 0:
                    month_pieces.append(da)
        if month_pieces:
            da = xr.concat(month_pieces, dim="time")
            pieces.append(da)

    if not pieces:
        if required:
            raise FileNotFoundError(
                f"No ERA5 data found for '{key}' (token '{token}') in {data_dir} with pattern "
                f"'{file_pattern}', months {months}, levels {list(levels)}.")
        return None

    da = xr.concat(pieces, dim="pressure") if len(pieces) > 1 else pieces[0]
    # Remove duplicated time stamps (e.g. overlapping files) and sort.
    _, idx = np.unique(da["time"].values, return_index=True)
    da = da.isel(time=np.sort(idx)).sortby("time").sortby("pressure")
    if time_offset_hours:
        da = da.assign_coords(time=da["time"] + np.timedelta64(int(round(time_offset_hours * 3600)), "s"))
    da.name = key
    return da


# ---------------------------------------------------------------------------
# Interpolation container
# ---------------------------------------------------------------------------

class MetField:
    """
    Gridded 3-D met fields with vectorised linear interpolation in (t, p, lat, lon).

    Parameters
    ----------
    time : array-like of datetime64
    pressure_hPa, lat_deg, lon_deg : 1-D arrays (any order, lon in any convention)
    u, v : arrays (nt, np, nlat, nlon), m s-1
    omega : array (nt, np, nlat, nlon), Pa s-1
    temperature : optional array (nt, np, nlat, nlon), K
    """

    def __init__(self, time, pressure_hPa, lat_deg, lon_deg, u, v, omega, temperature=None):
        time = pd.to_datetime(np.asarray(time))
        self.t_ref = time.min()
        t_sec = np.asarray((time - self.t_ref).total_seconds(), dtype=float)

        p = np.asarray(pressure_hPa, dtype=float)
        lat = np.asarray(lat_deg, dtype=float)
        lon = np.mod(np.asarray(lon_deg, dtype=float), 360.0)

        fields = [np.asarray(u), np.asarray(v), np.asarray(omega)]
        self.has_temperature = temperature is not None
        if self.has_temperature:
            fields.append(np.asarray(temperature))
        data = np.stack(fields, axis=-1).astype(np.float64)

        for axis, coord in ((0, t_sec), (1, p), (2, lat), (3, lon)):
            idx = np.argsort(coord, kind="stable")
            data = np.take(data, idx, axis=axis)
            if axis == 0:
                t_sec = t_sec[idx]
            elif axis == 1:
                p = p[idx]
            elif axis == 2:
                lat = lat[idx]
            else:
                lon = lon[idx]

        # Periodic longitude: add a wrap-around column for global grids.
        dlon = np.median(np.diff(lon)) if lon.size > 1 else 360.0
        self.periodic = lon.size > 1 and abs((lon[-1] - lon[0]) + dlon - 360.0) < 0.5 * dlon
        if self.periodic:
            lon = np.append(lon, lon[0] + 360.0)
            data = np.concatenate([data, data[:, :, :, :1]], axis=3)

        if t_sec.size < 2:
            raise ValueError("At least two time levels are required for interpolation.")

        self.t_sec, self.p, self.lat, self.lon = t_sec, p, lat, lon
        self._interp = RegularGridInterpolator((t_sec, p, lat, lon), data, method="linear",
                                               bounds_error=False, fill_value=np.nan)

    # -- construction from ERA5 files ---------------------------------------
    @classmethod
    def from_era5(cls, t_start, t_end, data_dir=DEFAULT_DATA_DIR, file_pattern=DEFAULT_FILE_PATTERN,
                  levels=DEFAULT_LEVELS, var_tokens=None, domain=None, p_min=0.0, p_max=1100.0,
                  use_temperature=True, time_offset_hours=0.0, pad_days=1.0):
        """
        Load ERA5 u, v, omega (and optionally T) covering [t_start, t_end] (+/- pad_days).
        """
        t0 = pd.Timestamp(t_start) - pd.Timedelta(days=pad_days)
        t1 = pd.Timestamp(t_end) + pd.Timedelta(days=pad_days)
        months = months_between(t0 - pd.Timedelta(hours=abs(time_offset_hours)),
                                t1 + pd.Timedelta(hours=abs(time_offset_hours)))
        kw = dict(data_dir=data_dir, file_pattern=file_pattern, levels=levels, var_tokens=var_tokens,
                  domain=domain, p_min=p_min, p_max=p_max, time_offset_hours=time_offset_hours)
        u = load_era5_field("u", months, t0, t1, **kw)
        v = load_era5_field("v", months, t0, t1, **kw)
        w = load_era5_field("w", months, t0, t1, **kw)
        t = load_era5_field("t", months, t0, t1, required=False, **kw) if use_temperature else None

        u, v, w = xr.align(u, v, w, join="inner")
        if t is not None:
            try:
                t = t.sel(time=u["time"], pressure=u["pressure"], lat=u["lat"], lon=u["lon"])
            except KeyError:
                t = None  # temperature grid does not match the winds; use reference temperature
        if u.sizes["pressure"] < 2:
            raise ValueError("At least two pressure levels are needed for 3-D trajectories.")

        met = cls(u["time"].values, u["pressure"].values, u["lat"].values, u["lon"].values,
                  u.values, v.values, w.values, None if t is None else t.values)
        met.source_info = dict(data_dir=data_dir, file_pattern=file_pattern, months=months)
        return met

    # -- properties ---------------------------------------------------------
    @property
    def times(self):
        return self.t_ref + pd.to_timedelta(self.t_sec, unit="s")

    @property
    def p_bottom(self):
        return float(self.p.max())

    @property
    def p_top(self):
        return float(self.p.min())

    def to_seconds(self, when):
        return (pd.Timestamp(when) - self.t_ref).total_seconds()

    def wrap_lon(self, lon):
        lon = np.mod(lon, 360.0)
        if not self.periodic and self.lon[-1] >= 360.0:
            # Regional grid crossing 0/360: map small longitudes to the extended range.
            lon = np.where(lon < self.lon[0], lon + 360.0, lon)
        return lon

    def sample(self, t_sec, p_hpa, lat_deg, lon_deg):
        """
        Interpolate fields at particle positions.

        Returns array (n, nvar) with columns u, v, omega[, T]; NaN outside the data.
        """
        t_sec, p_hpa, lat_deg, lon_deg = np.broadcast_arrays(
            np.asarray(t_sec, float), np.asarray(p_hpa, float), np.asarray(lat_deg, float),
            np.asarray(lon_deg, float))
        pts = np.column_stack([t_sec.ravel(), p_hpa.ravel(), lat_deg.ravel(), self.wrap_lon(lon_deg.ravel())])
        return self._interp(pts)
