# EpiNOAA-Python

Turns raw USDA county-level maize and soy yield data and NOAA weather data
into a merged county-year panel, ready for weather-sensitivity analysis.
It does not do the econometric modelling itself, the LLDVE and RCS models
live in a seperate repository.

The paper says maize, the code says corn. Same crop, I've left the code
alone rather than renaming every file and variable.

## paper

Uneven buffering of U.S. maize yields against extreme heat - a time-varying
coefficient panel approach

> U.S. maize production supplies one-third of the global harvest and faces
> escalating risks from extreme weather. Despite decades of technological
> advances intended to enhance climate resilience, how large-scale yield
> sensitivity to temperature and rainfall has evolved remains poorly
> understood.

Link and full citation once available.

## why

Before any modelling can happen, the raw yield and weather data need
cleaning, a county-to-climate-division crosswalk, and merging into one
panel. I kept that work in its own repo, separate from the modelling code.

This started as a fork of NOAA's
[EpiNOAA-Python](https://github.com/NOAA-Big-Data-Program/EpiNOAA-Python),
which is where the nclimgrid access code came from. Almost none of the
original remains, see `03b_weather_nclimgrid_importer/README.md` for what
happened to it.

## setup

Python 3.9 or newer, and Poetry.

```powershell
pip install poetry
python -m poetry install
```

`make install` does the same thing, if you'd rather.

On Windows, if that fails with "Microsoft Visual C++ 14.0 or greater is
required", one of the scientific packages is being built from source and
needs the [Microsoft C++ Build
Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), with the
"Desktop development with C++" workload selected.

Notebooks (stages 01, 02, 04) run interactively in VS Code. Stage 03b's
scripts are plain `.py` files and need Poetry's environment invoked a
specific way, which is where most of the friction is. That's all written
down in `03b_weather_nclimgrid_importer/README.md`.

## layout

The numeric prefix is the pipeline stage, in run order. An `a`/`b` suffix
means two parallel siblings at the same stage, either the same step for a
different crop or a different implementation of it, not a separate stage.

Inside each stage folder:

- `extracted_<source>_<data>_data/` is raw input, unmodified.
- `created_dfs_stepN/` is that stage's output, `N` matching the stage prefix.
- `created_dfs_step_final/` is the pipeline's actual final output, stage 03
  onwards.

At the root:

- `config.py` holds the shared thresholds: the growing season (1 April to 30
  September), GDD base 10 degrees C capped at 29, KDD above 29, concurrent
  hot-dry days above 30 degrees C with under 1 mm of rain.
- `OLD/` is superseded files and earlier versions, kept rather than deleted.

Every stage folder has its own README with the detail for that step. This
one is the the map.

## data

Sources and links: `docs/Sources data.txt`. Things I've considered but not
used are in `docs/New Potential Sources.txt`.

- USDA NASS Quick Stats: county-level corn and soy yield.
- NOAA climdiv: monthly temperature and precipitation, plus the
  county-to-division crosswalk.
- NOAA nclimgrid-daily (S3): daily weather, 1951 onwards.
- Census Gazetteer: county centroid coordinates, used for the
  100th-meridian filter.

Raw extracts live in each stage's own `extracted_*` folder. The large
nclimgrid parquet archive that stage 03a reads is not in the repo, and that
stage isn't currently maintained.

## the pipeline (5-state paper dataset)

Illinois, Indiana, Iowa, Minnesota and Nebraska.

1. `01a_yield_data_corn/format_corn_yield_data_paper.ipynb` and
   `01b_yield_data_soy/format_soy_yield_data_paper.ipynb`: clean the raw
   USDA export, one crop each. Output:
   `created_dfs_step1/df_<crop>_yield_2026.csv`.
2. `02_weather_climdiv_crosswalk/merge_yield_monthly_weather.ipynb`: joins
   in climdiv monthly weather. Run once per crop. Output:
   `created_dfs_step2/df_yield_climdiv_<crop>_paper.csv`.
3. `03b_weather_nclimgrid_importer/get_data_importer.py`: joins in
   nclimgrid daily weather, aggregated to growing-season metrics (GDD, KDD,
   TMAX_AVG, PREC, CHD). Run once per crop. Output:
   `created_dfs_step_final/df_final_importer_<crop>_paper.csv`. This is the
   paper's dataset.
4. `04_compare_validate/compare_GS_climdiv_nclimgrid.ipynb`: sanity-checks
   climdiv against nclimgrid for the same counties.

## the 100th meridian branch

A second, parallel pipeline covering all US counties east of the 100th
meridian, a common convention in the literature, as a larger dataset to test
the same method against. It reuses stages 01 to 03b unmodified, just pointed
at different filenames (`_east100m` rather than `_paper`) so it can't
overwrite the paper's data. See `00_filter_east_100th_meridian/README.md`.

## where things stand

- 5-state dataset: stages 01, 02 and 03b are run and current for both crops,
  as of August 2026.
- 100th meridian: corn is done through 03b, 31 states and 1,859 counties.
  Soy hasn't started, the all-states raw export still needs downloading.
- Stage 04: the growing-season comparison has been rerun for corn
  (September 2026). Still to do: the same comparison for the 100th meridian
  corn data (`df_yield_climdiv_corn_east100m.csv` against
  `df_final_importer_corn_east100m.csv`, the notebook's filenames need
  changing for that). Soy doesn't need its own run, its county-years and
  weather are identical to corn's. The other two comparisons are still
  waiting on 03a.
- Stage 03a isn't maintained.

## notes

- Nebraska has no county-level soybean yield data before 1960 in USDA's
  records. Not a bug, I checked the raw export directly.  The panel is
  deliberately unbalanced rather than trimmed to a common start year.
- Both `03b` scripts also compute an SGF ("Silking to Grain-Fill") weather
  block alongside the growing-season one: GDD_SGF, KDD_SGF, TMAX_AVG_SGF,
  PREC_SGF, CHD_SGF. Not used in the paper, only the growing-season metrics
  are. It's left in and still configurable in `config.py`
  (`SGF_START_MONTH`/`SGF_END_MONTH`, currently 1 July to 15 August) for
  anyone who wants to set a different window and look at that instead.
- The month-level comparison notebook in stage 04 is blocked on 03a. Details,
  including where the parquet files come from and what it would take to
  bring that stage back: `03a_weather_nclimgrid_local/README.md`.
- July 2023 precipitation in nClimGrid-Daily looks wrong. Summed over the
  month, the daily county values come out at roughly half of NOAA's monthly
  climdiv values across the whole country (median 49 mm against 105 mm),
  while the other months I checked match to within rounding. Station data
  from GHCN-Daily backs climdiv, for example Chicago O'Hare recorded 193 mm
  that month against 58 mm in the daily data for Cook County. The problem is
  in NOAA's own files, not in this code, and a fresh download is identical.
  Until it's fixed, `PREC_GS` for 2023 is too low for most counties, and
  `CHD_GS` and the precipitation extremes for that year shouldn't be
  trusted. How to handle it in the paper is still open.
- Environment quirks (Poetry, PATH) are documented in
  `03b_weather_nclimgrid_importer/README.md`, not repeated here.

## citation

BibTeX, once available.
