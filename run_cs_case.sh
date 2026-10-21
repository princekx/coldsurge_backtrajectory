#!/bin/bash
# Download ERA5 (CDS) for one cold-surge arrival date, run 10-day back trajectories and plot.
# Usage: ./run_cs_case.sh 2025-12-13 [extra run_backtraj.py args]
set -euo pipefail
DATE=${1:?arrival date YYYY-MM-DD}; shift || true
ERA5_DIR=/data/users/prince.xavier/SEAPy_climate/SEAPy_obs_data/ERA5
OUT_DIR=/data/users/prince.xavier/SEAPy_climate/back_trajs
cd "$(dirname "$0")"
module load scitools/default-current

START=$(date -d "$DATE - 11 days" +%F)
END=$(date -d "$DATE + 1 day" +%F)
# Whole months so the files can be reused for other dates.
python download_era5_cds.py --combined --out-dir "$ERA5_DIR" \
    --start "$(date -d "$START" +%Y-%m-01)" --end "$(date -d "$(date -d "$END" +%Y-%m-01) + 1 month - 1 day" +%F)"

python run_backtraj.py --start-time "${DATE}T00" --days-back 10 --members 4 --seed 1 \
    --data-dir "$ERA5_DIR" --file-pattern 'ERA5_Plevels_{var}_6hr.{yyyymm}.nc' \
    --var-token u=uvwt --var-token v=uvwt --var-token w=uvwt --var-token t=uvwt \
    --out-dir "$OUT_DIR" --csv --overwrite "$@"

STEM="$OUT_DIR/name_backtraj_$(date -d "$DATE" +%Y%m%d)00"
python plot_backtraj.py "$STEM.nc" -o "$STEM.png"
echo "Output: $STEM.nc $STEM.csv $STEM.png"
