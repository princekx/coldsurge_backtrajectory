"""
Lightweight NAME-style Lagrangian back-trajectory model driven by ERA5 3-D winds.

The particle update follows the NAME III (vn8.8) code, simplified for pressure-level
reanalysis data:

* Backward runs   - mean flow is reversed (NAME `ReverseFlow`), travel time is positive.
* Mean advection  - explicit Euler as in NAME `EvolveParticle`, or a 2nd-order
                    predictor-corrector (Heun) scheme (default).
* Horizontal move - great-circle update of lat/lon from metric displacements
                    (NAME `TrajectoryCalc`), valid near the poles.
* Turbulence      - NAME homogeneous diffusive ("long-range") random walk with the
                    default free-tropospheric values sigma_u = 0.25 m/s, tau_u = 300 s,
                    sigma_w = 0.1 m/s, tau_w = 100 s (NAME `XHomogSdeTerms`).
                    Vertical displacements are converted to pressure hydrostatically.
* Mesoscale       - NAME unresolved-mesoscale-motions random walk
                    dX = sqrt(2 sigma_m^2 tau_m dt) N(0,1) with the NAME ERA5 met
                    definition values sigma_m = 0.9 m/s, tau_m = 9600 s.
* Damping         - NAME damping of the diffusive random walks at short travel times
                    (on by default, as in NAME).
* Boundaries      - particles reflect at the bottom/top pressure levels of the data
                    (NAME reflects at the ground); particles leaving the horizontal
                    domain or the met time range are terminated (set to NaN).

All particles are advanced simultaneously with vectorised numpy operations.
"""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import xarray as xr

R_EARTH = 6_371_229.0  # m, NAME EarthRadius
G = 9.80665            # m s-2
RD = 287.05            # J kg-1 K-1


@dataclass
class DispersionOptions:
    """NAME-like dispersion options (defaults follow NAME vn8.8)."""
    turbulence: bool = True
    mesoscale_motions: bool = True
    damping: bool = True           # NAME default 'Damping? = Yes'
    free_trop_sigu: float = 0.25   # m s-1
    free_trop_sigw: float = 0.1    # m s-1
    free_trop_tauu: float = 300.0  # s
    free_trop_tauw: float = 100.0  # s
    meso_sigu: float = 0.9         # m s-1  (NAME 'ERA5 3hr' met definition)
    meso_tauu: float = 9600.0      # s
    t_ref_K: float = 280.0         # temperature used for hydrostatics if T is unavailable


# ---------------------------------------------------------------------------
# Release (source) definition
# ---------------------------------------------------------------------------

def make_release(start_times, lat_range=(1.5, 6.5), lon_range=(100.5, 104.0), pressures_hPa=(825.0, 850.0, 875.0),
                 n_lat=5, n_lon=5, members=1, mode="grid", n_random=100, seed=None):
    """
    Build an ensemble of release points in latitude, longitude and pressure.

    Parameters
    ----------
    start_times : datetime-like or sequence
        Release (arrival) times; each gets the full set of release points.
    lat_range, lon_range : (min, max)
        Release box (default: Peninsular Malaysia).
    pressures_hPa : sequence
        Release pressure levels (grid mode) or (min, max) range used in random mode.
    n_lat, n_lon : int
        Number of grid points in the box (grid mode).
    members : int
        Number of stochastic members released from each point.
    mode : 'grid' | 'random'
        Regular grid of points, or n_random points uniformly distributed in the box
        (random pressure uniform between min and max of pressures_hPa).

    Returns
    -------
    pandas.DataFrame with columns release_time, release_lat, release_lon,
    release_pressure, point_id, member.
    """
    rng = np.random.default_rng(seed)
    times = pd.to_datetime(np.atleast_1d(start_times))

    if mode == "grid":
        lats = np.linspace(lat_range[0], lat_range[1], n_lat) if n_lat > 1 else np.array([np.mean(lat_range)])
        lons = np.linspace(lon_range[0], lon_range[1], n_lon) if n_lon > 1 else np.array([np.mean(lon_range)])
        P, Y, X = np.meshgrid(np.asarray(pressures_hPa, float), lats, lons, indexing="ij")
        pts = np.column_stack([Y.ravel(), X.ravel(), P.ravel()])
    elif mode == "random":
        p_lo, p_hi = float(np.min(pressures_hPa)), float(np.max(pressures_hPa))
        pts = np.column_stack([rng.uniform(*lat_range, n_random), rng.uniform(*lon_range, n_random),
                               rng.uniform(p_lo, p_hi, n_random)])
    else:
        raise ValueError("mode must be 'grid' or 'random'")

    rows = []
    for t in times:
        for ip, (la, lo, p) in enumerate(pts):
            for m in range(members):
                rows.append((t, la, lo, p, ip, m))
    return pd.DataFrame(rows, columns=["release_time", "release_lat", "release_lon", "release_pressure",
                                       "point_id", "member"])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def great_circle_move(lat_deg, lon_deg, dx_east, dy_north):
    """
    Move points by metric displacements along a great circle (NAME TrajectoryCalc).
    """
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)
    d = np.hypot(dx_east, dy_north) / R_EARTH
    theta = np.arctan2(dx_east, dy_north)

    x = -np.sin(d) * np.cos(theta)
    y = np.sin(d) * np.sin(theta)
    z = np.cos(d)

    x2 = x * np.sin(phi) + z * np.cos(phi)
    y2 = y
    z2 = -x * np.cos(phi) + z * np.sin(phi)

    phi_new = np.arctan2(z2, np.hypot(x2, y2))
    lam_new = lam + np.arctan2(y2, x2)
    return np.rad2deg(phi_new), np.mod(np.rad2deg(lam_new), 360.0)


def _reflect(p, p_top, p_bottom):
    """Reflect pressures at the top/bottom of the model domain."""
    p = np.where(p > p_bottom, 2.0 * p_bottom - p, p)
    p = np.where(p < p_top, 2.0 * p_top - p, p)
    return np.clip(p, p_top, p_bottom)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BackTrajectoryModel:
    """
    NAME-style ensemble back-trajectory model.

    Parameters
    ----------
    met : era5_met.MetField
        Met fields covering the release times and `days_back` before them.
    options : DispersionOptions, optional
    dt_minutes : float
        Particle time step (NAME default sync/time step for ERA5-type runs is ~15 min).
    output_minutes : float
        Output interval along the trajectories (multiple of dt_minutes).
    scheme : 'rk2' | 'euler'
        Mean-flow integration scheme ('euler' reproduces NAME's explicit update).
    seed : int, optional
        Random seed for reproducible stochastic members.
    """

    def __init__(self, met, options=None, dt_minutes=15.0, output_minutes=60.0, scheme="rk2", seed=None):
        if scheme not in ("rk2", "euler"):
            raise ValueError("scheme must be 'rk2' or 'euler'")
        self.met = met
        self.opt = options or DispersionOptions()
        self.dt = float(dt_minutes) * 60.0
        self.out_every = max(1, int(round(output_minutes / dt_minutes)))
        self.scheme = scheme
        self.rng = np.random.default_rng(seed)

    # -- physics ----------------------------------------------------------
    def _velocity(self, t_sec, p, lat, lon):
        s = self.met.sample(t_sec, p, lat, lon)
        temp = s[:, 3] if self.met.has_temperature else np.full(p.shape, self.opt.t_ref_K)
        return s[:, 0], s[:, 1], s[:, 2], temp

    def _stochastic(self, n, p_hpa, temp, travel_time):
        """Random-walk displacements (east m, north m, pressure hPa) for one step."""
        o, dt = self.opt, self.dt
        dx = np.zeros(n)
        dy = np.zeros(n)
        dp = np.zeros(n)

        if o.turbulence:
            k_h = o.free_trop_sigu ** 2 * o.free_trop_tauu
            k_z = o.free_trop_sigw ** 2 * o.free_trop_tauw
            fac_h = fac_z = 1.0
            if o.damping:
                tt = travel_time + dt / 2.0
                fac_h = tt / np.sqrt(tt ** 2 + o.free_trop_tauu ** 2)
                fac_z = tt / np.sqrt(tt ** 2 + o.free_trop_tauw ** 2)
            b_h = np.sqrt(2.0 * k_h * dt * fac_h)
            b_z = np.sqrt(2.0 * k_z * dt * fac_z)
            dx += b_h * self.rng.standard_normal(n)
            dy += b_h * self.rng.standard_normal(n)
            dz = b_z * self.rng.standard_normal(n)
            rho = p_hpa * 100.0 / (RD * temp)
            dp += -rho * G * dz / 100.0

        if o.mesoscale_motions:
            if o.damping:
                k = np.sqrt(2.0 * o.meso_sigu ** 2 * o.meso_tauu * dt *
                            (1.0 - np.exp(-(travel_time + dt / 2.0) / o.meso_tauu)))
            else:
                k = np.sqrt(2.0 * o.meso_sigu ** 2 * o.meso_tauu * dt)
            dx += k * self.rng.standard_normal(n)
            dy += k * self.rng.standard_normal(n)

        return dx, dy, dp

    def _step(self, t_sec, p, lat, lon, travel_time):
        """Advance particles one step backwards in time. Returns new state and velocities at start."""
        dt = self.dt
        u, v, w, temp = self._velocity(t_sec, p, lat, lon)

        # Backward run: reverse the mean flow (NAME ReverseFlow).
        dx, dy, dp = -u * dt, -v * dt, -w / 100.0 * dt

        if self.scheme == "rk2":
            lat_p, lon_p = great_circle_move(lat, lon, dx, dy)
            p_p = _reflect(p + dp, self.met.p_top, self.met.p_bottom)
            u2, v2, w2, _ = self._velocity(t_sec - dt, p_p, lat_p, lon_p)
            ok = np.isfinite(u2) & np.isfinite(v2) & np.isfinite(w2)
            dx = np.where(ok, -0.5 * (u + u2) * dt, dx)
            dy = np.where(ok, -0.5 * (v + v2) * dt, dy)
            dp = np.where(ok, -0.5 * (w + w2) / 100.0 * dt, dp)

        sx, sy, sp = self._stochastic(p.size, p, temp, travel_time)
        lat_n, lon_n = great_circle_move(lat, lon, dx + sx, dy + sy)
        p_n = _reflect(p + dp + sp, self.met.p_top, self.met.p_bottom)
        return p_n, lat_n, lon_n, (u, v, w)

    # -- driver -------------------------------------------------------------
    def run(self, release, days_back=10.0):
        """
        Run back trajectories for all particles in `release` (see make_release).

        Returns
        -------
        xarray.Dataset with dims (particle, step); variables lat, lon, pressure,
        u, v, omega and time, plus per-particle release information.
        """
        release = release.reset_index(drop=True)
        n_steps = int(np.ceil(days_back * 86400.0 / self.dt))
        n_out = n_steps // self.out_every + 1
        hours_back = np.arange(n_out) * self.out_every * self.dt / 3600.0

        npart = len(release)
        out = {k: np.full((npart, n_out), np.nan) for k in ("lat", "lon", "pressure", "u", "v", "omega")}
        tmin, tmax = self.met.t_sec[0], self.met.t_sec[-1]

        for rt, grp in release.groupby("release_time", sort=True):
            idx = grp.index.to_numpy()
            t0 = self.met.to_seconds(rt)
            if not (tmin <= t0 <= tmax):
                raise ValueError(f"Release time {rt} is outside the met data range "
                                 f"{self.met.times[0]} - {self.met.times[-1]}.")
            lat = grp["release_lat"].to_numpy(float)
            lon = np.mod(grp["release_lon"].to_numpy(float), 360.0)
            p = grp["release_pressure"].to_numpy(float)
            alive = np.ones(idx.size, bool)

            for k in range(n_steps + 1):
                t_sec = t0 - k * self.dt
                if t_sec < tmin:
                    break
                travel = k * self.dt
                p_n, lat_n, lon_n, (u, v, w) = self._step(t_sec, p, lat, lon, travel)
                alive &= np.isfinite(u) & np.isfinite(v) & np.isfinite(w)

                if k % self.out_every == 0:
                    j = k // self.out_every
                    rows = idx[alive]
                    for name, val in (("lat", lat), ("lon", lon), ("pressure", p), ("u", u), ("v", v), ("omega", w)):
                        out[name][rows, j] = val[alive]

                if not alive.any() or k == n_steps:
                    break
                lat = np.where(alive, lat_n, lat)
                lon = np.where(alive, lon_n, lon)
                p = np.where(alive, p_n, p)

        rel_time = pd.to_datetime(release["release_time"]).to_numpy()
        times = rel_time[:, None] - (hours_back[None, :] * 3600.0).astype("timedelta64[s]")
        ds = xr.Dataset(
            data_vars=dict(
                lat=(("particle", "step"), out["lat"], {"units": "degrees_north"}),
                lon=(("particle", "step"), out["lon"], {"units": "degrees_east"}),
                pressure=(("particle", "step"), out["pressure"], {"units": "hPa"}),
                u=(("particle", "step"), out["u"], {"units": "m s-1"}),
                v=(("particle", "step"), out["v"], {"units": "m s-1"}),
                omega=(("particle", "step"), out["omega"], {"units": "Pa s-1"}),
                time=(("particle", "step"), times),
                release_time=("particle", rel_time),
                release_lat=("particle", release["release_lat"].to_numpy(float)),
                release_lon=("particle", release["release_lon"].to_numpy(float)),
                release_pressure=("particle", release["release_pressure"].to_numpy(float)),
                point_id=("particle", release["point_id"].to_numpy(int)),
                member=("particle", release["member"].to_numpy(int)),
            ),
            coords=dict(particle=np.arange(npart), hours_back=("step", hours_back)),
        )
        ds.attrs.update({f"opt_{k}": (int(v) if isinstance(v, bool) else v) for k, v in asdict(self.opt).items()})
        ds.attrs.update(dict(dt_minutes=self.dt / 60.0, scheme=self.scheme, days_back=float(days_back),
                             met_temperature_used=int(self.met.has_temperature),
                             description="NAME-style ensemble back trajectories from ERA5 3-D winds"))
        return ds


def to_dataframe(ds):
    """
    Long-format DataFrame using the column names of the existing trajectory.py output
    (time, hours_back, pressure_hPa, lat_deg, lon_deg, u_ms, v_ms, omega_Pas, path_label).
    """
    df = pd.DataFrame({
        "time": ds["time"].values.ravel(),
        "hours_back": np.broadcast_to(ds["hours_back"].values, ds["lat"].shape).ravel(),
        "pressure_hPa": ds["pressure"].values.ravel(),
        "lat_deg": ds["lat"].values.ravel(),
        "lon_deg": ds["lon"].values.ravel(),
        "u_ms": ds["u"].values.ravel(),
        "v_ms": ds["v"].values.ravel(),
        "omega_Pas": ds["omega"].values.ravel(),
        "path_label": np.repeat(ds["particle"].values, ds.sizes["step"]),
    })
    for name in ("release_time", "release_lat", "release_lon", "release_pressure", "point_id", "member"):
        df[name] = np.repeat(ds[name].values, ds.sizes["step"])
    return df.dropna(subset=["lat_deg"]).reset_index(drop=True)
