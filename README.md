# NAME_backtraj – lightweight NAME-style back trajectories from ERA5

A small Python re-implementation of the Lagrangian particle numerics of the Met Office
NAME III model (vn8.8, `Code_NameIII/Particle.F90`, `Case.F90`), driven directly by
ERA5 pressure-level 3-D winds (u, v, omega). It launches ensembles of back trajectories
in latitude, longitude and pressure (default: Peninsular Malaysia, 825/850/875 hPa) and
traces them back 10 days.

| File | Purpose |
|------|---------|
| `era5_met.py` | ERA5 reader (monthly files, per-level or multi-level, zipped CDS files), sub-setting, 4-D interpolation (`MetField`) |
| `lagrangian.py` | Particle model (`BackTrajectoryModel`), release ensembles (`make_release`), NAME options (`DispersionOptions`), `to_dataframe` |
| `run_backtraj.py` | Command-line driver (one or many arrival dates) |
| `run_backtraj.sh` | SLURM wrapper |
| `download_era5_cds.py` | Download ERA5 pressure-level u, v, omega, T from the Copernicus CDS (`~/.cdsapirc` needed) |
| `run_cs_case.sh` | One cold-surge date end to end: CDS download → 10-day back trajectories → plot |
| `plot_backtraj.py` | Map (coloured by pressure) and pressure–time plots |
| `test_backtraj.py` | Tests with synthetic winds and ERA5-like files |

## NAME physics carried over

| NAME (vn8.8) | Here |
|--------------|------|
| `ReverseFlow` for backwards runs | mean wind reversed, travel time positive |
| `EvolveParticle` explicit mean-flow update | `--scheme euler`; default `rk2` (Heun predictor–corrector, more accurate for long steps/curved flow) |
| `TrajectoryCalc` great-circle lat/long update | `great_circle_move` (valid across the poles and dateline) |
| Homogeneous diffusive free-troposphere turbulence (σu = 0.25 m/s, τu = 300 s, σw = 0.1 m/s, τw = 100 s) | same defaults; vertical displacement converted to pressure hydrostatically (uses ERA5 T if available) |
| Unresolved mesoscale motions, `ERA5 3hr` met definition (σ = 0.9 m/s, τ = 9600 s) | same defaults (`--meso-sigu`, `--meso-tauu`) |
| Damping of diffusive random walks at short travel times (NAME default `Damping? = Yes`) | on by default; `--no-damping` |
| Ground reflection | reflection at the bottom/top data levels |

### Differences from NAME

- NAME uses ERA5 model levels (L72, 3-hourly) in a terrain-following height/eta coordinate
  with reflection at the real ground. This code uses pressure levels and reflects at
  1000 hPa, so over high terrain (Tibet, Mongolia) parcels can sit in below-ground
  (extrapolated) levels.
- No boundary-layer turbulence (NAME uses BLH/heat flux/u* profiles), so low-level vertical
  mixing is underestimated.
- No convection scheme, Langevin (velocity-memory) turbulence, deposition or plume rise.
- Linear interpolation in pressure rather than height.
- Particles leaving the loaded sub-domain are terminated (NAME is global).

Use `--no-turbulence --no-meso` for classical deterministic
trajectories, or `--members N` for a stochastic NAME-like ensemble per release point.
Note: the NAME mesoscale values are tuned for 3-hourly 0.25° ERA5; with daily-mean
winds more of the variability is unresolved, so consider larger `--meso-sigu`.

## Met data

The model needs ERA5 **pressure-level** u, v and omega (`vertical_velocity`, Pa s-1) on
at least two levels spanning the trajectories, e.g. 1000–200 hPa; temperature is optional.
Data must cover `days_back` + 1 day before each arrival time.
Download them with `download_era5_cds.py` (CDS dataset `reanalysis-era5-pressure-levels`,
6-hourly, 0.5°). `--combined` puts all variables in one request per month, which avoids
the CDS queue limit on parallel requests (queueing can still take hours):

```bash
python download_era5_cds.py --start 2025-12-01 --end 2025-12-31 --combined \
    --out-dir /data/users/prince.xavier/SEAPy_climate/SEAPy_obs_data/ERA5
# -> ERA5_Plevels_uvwt_6hr.202512.nc, read with
#    --file-pattern 'ERA5_Plevels_{var}_6hr.{yyyymm}.nc' --var-token u=uvwt --var-token v=uvwt \
#    --var-token w=uvwt --var-token t=uvwt
```

File names are given with `--file-pattern` using `{var}`, `{yyyymm}` and optional `{level}`:

```bash
# default: one file per variable and level (SEAPy_obs_data naming)
--file-pattern 'ERA5_Plevels_{var}_{level}hPa_daily.{yyyymm}.nc'   # var = u_component_of_wind, v_component_of_wind, vertical_velocity, temperature
# all levels in one file per variable, custom variable tokens
--file-pattern 'ERA5_Plevels_{var}_daily.{yyyymm}.nc' --var-token u=U --var-token v=V --var-token w=OMEGA
```

Daily-mean ERA5 files are stamped at 00 UTC; use `--time-offset-hours 12` to centre them at midday.
Only the sub-domain `--domain 30 210 -30 90` (lon0 lon1 lat0 lat1) and levels `--p-min/--p-max`
are loaded to save memory (~1 GB per 10 days at 0.25°, 10 levels).

## Usage

```bash
module load scitools/default-current

# 75 release points (5 lat x 5 lon x 3 levels) over Peninsular Malaysia, 10 days back
python run_backtraj.py --start-time 2006-12-16 --csv

# cold-surge onset dates, 4 stochastic members per point, as a batch job
sbatch run_backtraj.sh --dates-file cs_onsets.txt --members 4 --seed 1

# a cold-surge onset date end to end (download, run with 4 members, plot);
# writes name_backtraj_2025121300.{nc,csv,png} to /data/users/prince.xavier/SEAPy_climate/back_trajs
./run_cs_case.sh 2025-12-13

python plot_backtraj.py /data/users/prince.xavier/SEAPy_climate/coldsurge_comps/ERA5/name_backtraj/name_backtraj_2006121600.nc -o traj.png
```

Output (`name_backtraj_YYYYMMDDHH.nc`): variables `lat, lon (0–360), pressure, u, v, omega, time`
with dims `(particle, step)`, coordinate `hours_back`, and per-particle release information
(`release_lat/lon/pressure/time, point_id, member`). Particles leaving the loaded domain are
set to NaN from then on. `--csv` writes the same long-format columns as `../trajectory.py`
(`time, hours_back, pressure_hPa, lat_deg, lon_deg, u_ms, v_ms, omega_Pas, path_label`).

From Python / notebooks:

```python
from era5_met import MetField
from lagrangian import BackTrajectoryModel, DispersionOptions, make_release, to_dataframe

met = MetField.from_era5("2006-12-06", "2006-12-16", data_dir=..., file_pattern=...)
rel = make_release("2006-12-16", pressures_hPa=[850], members=10, seed=0)
ds = BackTrajectoryModel(met, DispersionOptions(), dt_minutes=15).run(rel, days_back=10)
df = to_dataframe(ds)
```

## Tests

```bash
module load scitools/default-current
python -m pytest -q test_backtraj.py
```
