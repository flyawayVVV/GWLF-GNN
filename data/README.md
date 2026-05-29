# Data Description

This directory contains the four processed input workbooks required to reproduce the neural-network experiments:

- `runoff-Volume.xlsx`: daily GWLF surface runoff volume for 16 subbasins.
- `groundwater-Volume.xlsx`: daily GWLF groundwater/baseflow volume for 16 subbasins.
- `Flow.xlsx`: observed daily outlet streamflow at Cuntan hydrological station.
- `WatershedInfo_new.xlsx`: subbasin attributes and river-network topology.

These files are the direct model inputs used by the released code. Raw hydrological yearbook records, raw meteorological station observations, land-use rasters, DEM data, and GIS preprocessing products are not redistributed in this repository. Please refer to the paper for the original data sources and preprocessing workflow.

SHA256 checksums are provided in `CHECKSUMS_SHA256.txt`.

Run the full experiment matrix from the repository root with:

```bash
python run_experiments.py --data-dir data --result-root results_final
```
