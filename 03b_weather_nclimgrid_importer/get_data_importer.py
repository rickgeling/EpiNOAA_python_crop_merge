#process_ag_weather_batched.py

#--- Standard Library Imports ---
import os
import time 
from typing import List, Dict, Optional, Any, Tuple 
from collections import defaultdict #for grouping FIPS by state

#--- Third-party Library Imports ---
import polars as pl
from polars.exceptions import ColumnNotFoundError
import numpy as np #for NaN

#--- Custom Module Imports ---
from nclimgrid_importer import load_nclimgrid_data #from python_script_s3_debug
import sys
#resolve the repo root from this file, not the cwd, so the script runs from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#now import config
import config


#which crop to run this script for
crop = "corn"  # "corn" or "soy"

#both filenames are derived from `crop`, so they can't drift apart
input_csv_filename = 'df_yield_climdiv_corn_east100m.csv' #f"df_yield_climdiv_{crop}_paper.csv"
output_csv_filename = 'df_final_importer_corn_east100m.csv' #f"df_final_importer_{crop}_paper.csv"


#--- Chunks 1, 2, 3, 4 (Helper functions - unchanged from previous version) ---
def load_and_prepare_yield_data(csv_path: str) -> Optional[Tuple[pl.DataFrame, List[str], int, int]]:
    """
    Loads the yield data CSV, prepares FIPS codes, and extracts unique FIPS and year range.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file not found at '{csv_path}'")
        return None
    try:
        yield_column_dtypes = {
            "state": pl.String, "county": pl.String, "year": pl.Int64
        }
        df_yield = pl.read_csv(csv_path, schema_overrides=yield_column_dtypes, null_values=["NA", "N/A", "", " "])
        print(f"\nSuccessfully loaded input CSV: '{csv_path}', Shape: {df_yield.shape}")
        required_cols = ["state", "county", "year"]
        for col_name_req in required_cols:
            if col_name_req not in df_yield.columns:
                print(f"Error: CSV must contain a '{col_name_req}' column.")
                return None
        df_yield = df_yield.drop_nulls(subset=["state", "county", "year"])
        if df_yield.height == 0:
            print("Error: No valid rows in CSV after dropping nulls in state, county, or year.")
            return None
        #county needs 3 digits otherwise the fips dont match what nclimgrid uses
        df_yield = df_yield.with_columns(
            pl.col("county").str.zfill(3).alias("county_padded"),
            (pl.col("state") + pl.col("county").str.zfill(3)).alias("fips_full")
        )
        unique_fips_to_process = df_yield.get_column("fips_full").unique().sort().to_list()
        print(f"\nFound {len(unique_fips_to_process)} unique FIPS codes to process from yield data.")
        if not unique_fips_to_process:
            print("Error: No unique FIPS codes found.")
            return None
        min_year_csv = df_yield.select(pl.col("year").min()).item()
        max_year_csv = df_yield.select(pl.col("year").max()).item()
        print(f"Year range in CSV: {min_year_csv} - {max_year_csv}")
        return df_yield, unique_fips_to_process, min_year_csv, max_year_csv
    except Exception as e:
        print(f"Error loading or processing input CSV: {type(e).__name__} - {e}")
        return None

def fetch_and_clean_daily_weather_batched( 
    fips_codes_batch: List[str], 
    start_year: int, 
    end_year: int
) -> Tuple[Optional[pl.DataFrame], List[Dict[str, str]]]:
    """
    Fetches daily weather data for a batch of FIPS codes and year range,
    then performs initial cleaning.
    (Unchanged from previous version)
    """
    print(f"\nFetching daily weather for FIPS batch (count: {len(fips_codes_batch)}), Years: {start_year}-{end_year}")
    
    start_date_str = f"{start_year}-01-01"
    end_date_str = f"{end_year}-12-31"

    #pulls the whole state batch in one go, this is the slow part
    daily_arrow_table, file_load_errors = load_nclimgrid_data(
        start_date=start_date_str, end_date=end_date_str, spatial_scale='cty', 
        scaled=True, counties=fips_codes_batch, 
        variables=config.TARGET_DAILY_WEATHER_VARIABLES 
    )

    if daily_arrow_table is None or daily_arrow_table.num_rows == 0:
        print(f"  No daily weather data returned or data was empty for FIPS batch for years {start_year}-{end_year}.")
        return None, file_load_errors 
    
    print(f"  Successfully fetched {daily_arrow_table.num_rows} daily records from S3 for FIPS batch.")
    df_daily_weather = pl.from_arrow(daily_arrow_table)

    if "date" not in df_daily_weather.columns:
        print(f"  Error: 'date' column missing for FIPS batch. Returning None & errors.")
        return None, file_load_errors
    if df_daily_weather["date"].dtype != pl.Datetime:
        try:
            df_daily_weather = df_daily_weather.with_columns(pl.col("date").cast(pl.Datetime))
        except Exception as e:
            print(f"  Error casting 'date' column for FIPS batch: {e}. Returning raw table & errors.")
            return df_daily_weather, file_load_errors 
    
    #the raw files carry -999.99 where a value is missing, turn those into real nulls
    #before anything downstream tries to average or sum them
    cast_and_clean_expressions = []
    for col_name in config.TARGET_DAILY_WEATHER_VARIABLES:
        if col_name in df_daily_weather.columns:
            if df_daily_weather[col_name].dtype == pl.String:
                cast_and_clean_expressions.append(
                    pl.when(pl.col(col_name).str.strip_chars().cast(pl.Float64, strict=False) == config.RAW_DATA_MISSING_VALUE_PLACEHOLDER)
                    .then(None).otherwise(pl.col(col_name).str.strip_chars().cast(pl.Float64, strict=False)).alias(col_name)
                )
            elif df_daily_weather[col_name].dtype == pl.Float64:
                cast_and_clean_expressions.append(
                    pl.when(pl.col(col_name) == config.RAW_DATA_MISSING_VALUE_PLACEHOLDER)
                    .then(None).otherwise(pl.col(col_name)).alias(col_name)
                )
            elif df_daily_weather[col_name].dtype.is_numeric():
                cast_and_clean_expressions.append(
                    pl.when(pl.col(col_name).cast(pl.Float64, strict=False) == config.RAW_DATA_MISSING_VALUE_PLACEHOLDER)
                    .then(None).otherwise(pl.col(col_name).cast(pl.Float64, strict=False)).alias(col_name)
                )
    if cast_and_clean_expressions:
        try:
            df_daily_weather = df_daily_weather.with_columns(cast_and_clean_expressions)
        except Exception as e:
            print(f"  Error applying casting/cleaning for FIPS batch: {e}. Returning raw table & errors.")
            return df_daily_weather, file_load_errors 
            
    final_columns_to_keep = ["date", "fips"] 
    if "state_name" in df_daily_weather.columns:
        final_columns_to_keep.append("state_name")
    for var_name in config.TARGET_DAILY_WEATHER_VARIABLES:
        if var_name in df_daily_weather.columns and df_daily_weather[var_name].dtype == pl.Float64:
            if var_name not in final_columns_to_keep: final_columns_to_keep.append(var_name)
        elif var_name in df_daily_weather.columns: 
             if var_name not in final_columns_to_keep : final_columns_to_keep.append(var_name)
    final_columns_to_keep = list(dict.fromkeys(final_columns_to_keep))
    
    missing_final_cols = [col for col in final_columns_to_keep if col not in df_daily_weather.columns]
    if missing_final_cols:
        print(f"  Error: Columns {missing_final_cols} missing before final select for FIPS batch. Available: {df_daily_weather.columns}")
        return None, file_load_errors
    df_daily_weather = df_daily_weather.select(final_columns_to_keep)
    print(f"  Cleaned daily weather data for FIPS batch has shape: {df_daily_weather.shape}")
    return df_daily_weather, file_load_errors

#tavg gets capped at GDD_MAX_TEMP_C first, so days hotter than that stop adding to GDD
def calculate_daily_gdd(tavg_series: pl.Series) -> pl.Series:
    tavg_float = tavg_series.cast(pl.Float64, strict=False) 
    capped_tavg = pl.min_horizontal(tavg_float, pl.lit(config.GDD_MAX_TEMP_C, dtype=pl.Float64))
    gdd_potential = capped_tavg - config.GDD_BASE_TEMP_C
    daily_gdd = pl.max_horizontal(pl.lit(0.0, dtype=pl.Float64), gdd_potential)
    return daily_gdd 

def calculate_daily_kdd(tmax_series: pl.Series) -> pl.Series:
    tmax_float = tmax_series.cast(pl.Float64, strict=False)
    kdd_potential = tmax_float - config.KDD_TEMP_THRESHOLD_C
    daily_kdd = pl.max_horizontal(pl.lit(0.0, dtype=pl.Float64), kdd_potential)
    return daily_kdd

#hot and dry on the same day counts as 1, a null just counts as no
def calculate_daily_chd(tmax_series: pl.Series, prcp_series: pl.Series) -> pl.Series:
    tmax_float = tmax_series.cast(pl.Float64, strict=False)
    prcp_float = prcp_series.cast(pl.Float64, strict=False)
    is_hot = (tmax_float > config.CHD_TEMP_THRESHOLD_C).fill_null(False)
    is_dry = (prcp_float < config.CHD_PRECIP_THRESHOLD_MM).fill_null(False)
    daily_chd = (is_hot & is_dry).cast(pl.Int8) 
    return daily_chd

def _calculate_seasonal_aggregates(df_season: pl.DataFrame, season_prefix: str) -> Dict[str, Any]:
    #strict on purpose: one missing day anywhere in the window voids the whole
    #seasonal figure rather than quietly returning a partial sum
    aggs: Dict[str, Any] = {}
    df_season_with_daily_calcs = df_season.clone() 
    if "tavg" in df_season_with_daily_calcs.columns and not df_season_with_daily_calcs.get_column("tavg").is_null().all():
        if df_season_with_daily_calcs.get_column("tavg").is_null().any(): aggs[f"GDD_{season_prefix}"] = np.nan
        else:
            df_season_with_daily_calcs = df_season_with_daily_calcs.with_columns(calculate_daily_gdd(df_season_with_daily_calcs.get_column("tavg")).alias("_daily_gdd"))
            aggs[f"GDD_{season_prefix}"] = df_season_with_daily_calcs.get_column("_daily_gdd").sum()
    else: aggs[f"GDD_{season_prefix}"] = np.nan
    if "tmax" in df_season_with_daily_calcs.columns and not df_season_with_daily_calcs.get_column("tmax").is_null().all():
        if df_season_with_daily_calcs.get_column("tmax").is_null().any():
            aggs[f"KDD_{season_prefix}"] = np.nan
            aggs[f"TMAX_AVG_{season_prefix}"] = np.nan
        else:
            df_season_with_daily_calcs = df_season_with_daily_calcs.with_columns(calculate_daily_kdd(df_season_with_daily_calcs.get_column("tmax")).alias("_daily_kdd"))
            aggs[f"KDD_{season_prefix}"] = df_season_with_daily_calcs.get_column("_daily_kdd").sum()
            aggs[f"TMAX_AVG_{season_prefix}"] = df_season_with_daily_calcs.get_column("tmax").mean()
    else:
        aggs[f"KDD_{season_prefix}"] = np.nan
        aggs[f"TMAX_AVG_{season_prefix}"] = np.nan
    if "prcp" in df_season_with_daily_calcs.columns and not df_season_with_daily_calcs.get_column("prcp").is_null().all():
        if df_season_with_daily_calcs.get_column("prcp").is_null().any(): aggs[f"PREC_{season_prefix}"] = np.nan
        else: aggs[f"PREC_{season_prefix}"] = df_season_with_daily_calcs.get_column("prcp").sum()
    else: aggs[f"PREC_{season_prefix}"] = np.nan
    if ("tmax" in df_season_with_daily_calcs.columns and "prcp" in df_season_with_daily_calcs.columns and
        not df_season_with_daily_calcs.get_column("tmax").is_null().all() and 
        not df_season_with_daily_calcs.get_column("prcp").is_null().all()):
        if df_season_with_daily_calcs.get_column("tmax").is_null().any() or df_season_with_daily_calcs.get_column("prcp").is_null().any():
            aggs[f"CHD_{season_prefix}"] = np.nan
        else:
            df_season_with_daily_calcs = df_season_with_daily_calcs.with_columns(calculate_daily_chd(df_season_with_daily_calcs.get_column("tmax"), df_season_with_daily_calcs.get_column("prcp")).alias("_daily_chd"))
            aggs[f"CHD_{season_prefix}"] = df_season_with_daily_calcs.get_column("_daily_chd").sum()
    else: aggs[f"CHD_{season_prefix}"] = np.nan
    return aggs

def aggregate_weather_to_yearly(daily_df_for_year: pl.DataFrame, year: int) -> Dict[str, Any]:
    all_aggs: Dict[str, Any] = {"year": year} 
    #growing season window first, 1 apr to 30 sep (see config)
    gs_start_date = pl.datetime(year, config.GS_START_MONTH, config.GS_START_DAY)
    gs_end_date = pl.datetime(year, config.GS_END_MONTH, config.GS_END_DAY)
    df_gs = daily_df_for_year.filter((pl.col("date") >= gs_start_date) & (pl.col("date") <= gs_end_date))
    gs_essential_cols = [col for col in config.TARGET_DAILY_WEATHER_VARIABLES if col != 'date']
    gs_essential_cols_present = all(col in df_gs.columns for col in gs_essential_cols)
    #no days in the window or a variable missing, then the whole GS block is just nan
    if df_gs.height == 0 or not gs_essential_cols_present:
        gs_aggs = {f"GDD_GS": np.nan, f"KDD_GS": np.nan, f"TMAX_AVG_GS": np.nan, f"PREC_GS": np.nan, f"CHD_GS": np.nan}
    else: gs_aggs = _calculate_seasonal_aggregates(df_gs, "GS")
    all_aggs.update(gs_aggs)
    #same again for the SGF window (jul 1 to aug 15), not used for the paper atm
    sgf_start_date = pl.datetime(year, config.SGF_START_MONTH, config.SGF_START_DAY)
    sgf_end_date = pl.datetime(year, config.SGF_END_MONTH, config.SGF_END_DAY)
    df_sgf = daily_df_for_year.filter((pl.col("date") >= sgf_start_date) & (pl.col("date") <= sgf_end_date))
    sgf_essential_cols_present = all(col in df_sgf.columns for col in gs_essential_cols)
    if df_sgf.height == 0 or not sgf_essential_cols_present:
        sgf_aggs = {f"GDD_SGF": np.nan, f"KDD_SGF": np.nan, f"TMAX_AVG_SGF": np.nan, f"PREC_SGF": np.nan, f"CHD_SGF": np.nan}
    else: sgf_aggs = _calculate_seasonal_aggregates(df_sgf, "SGF")
    all_aggs.update(sgf_aggs)
    return all_aggs

def main():
    overall_start_time = time.time()
    print("--- Starting Agricultural Weather Data Processing (State-Level FIPS Batching) ---")
    print(f"Target daily weather variables to fetch: {config.TARGET_DAILY_WEATHER_VARIABLES}")
    print(f"Weather data processing will start from year: {config.DATA_START_YEAR}")

    ##----- GET THE DATA ------
    #crop / input_csv_filename / output_csv_filename come from the config block at the top

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_df = os.path.join(script_dir, "..", "02_weather_climdiv_crosswalk", "created_dfs_step2")
    input_csv_path = os.path.join(path_df, input_csv_filename)

    output_csv_path = os.path.join(script_dir, "created_dfs_step_final", output_csv_filename)

    print(f"Crop: {crop}")
    print(f"  input : {input_csv_path}")
    print(f"  output: {output_csv_path}")

    prepared_data = load_and_prepare_yield_data(input_csv_path)
    if prepared_data is None:
        print("Failed to load and prepare yield data. Exiting.")
        return

    df_yield, unique_fips_to_process, min_year_csv, max_year_csv = prepared_data
    
    overall_fetch_start_year = config.DATA_START_YEAR
    overall_fetch_end_year = max_year_csv 
    print(f"Overall weather data will be fetched for years: {overall_fetch_start_year} - {overall_fetch_end_year}.")

    all_yearly_weather_aggregates: List[Dict[str, Any]] = []
    all_s3_load_errors_summary: Dict[str, List[Dict[str,str]]] = {} 

    #group the counties by state so each S3 fetch covers a whole state at once,
    #one request per county was far too slow
    fips_by_state = defaultdict(list)
    for fips in unique_fips_to_process:
        state_fips_prefix = fips[:2]
        fips_by_state[state_fips_prefix].append(fips)
    
    num_states = len(fips_by_state)
    print(f"\n--- Starting Main Processing Loop for {num_states} states ---")

    for state_idx, (state_fips_prefix, fips_in_state_list) in enumerate(fips_by_state.items()):
        state_start_time = time.time()
        print(f"\nProcessing State Group {state_idx+1}/{num_states} (State FIPS Prefix: {state_fips_prefix}, {len(fips_in_state_list)} counties)")

        max_year_for_this_state_batch = df_yield.filter(
            pl.col("fips_full").is_in(fips_in_state_list)
        ).select(pl.col("year").max()).item()
        
        #if a state's yield data ends before 1951 this deliberately lands below the start
        #year, which the check just underneath reads as "nothing to fetch"
        current_state_fetch_start_year = overall_fetch_start_year
        current_state_fetch_end_year = max(overall_fetch_start_year -1, min(overall_fetch_end_year, max_year_for_this_state_batch))

        if current_state_fetch_end_year < current_state_fetch_start_year:
            print(f"  Max relevant year for FIPS in state {state_fips_prefix} ({max_year_for_this_state_batch}) is before effective data start year ({current_state_fetch_start_year}). Skipping.")
            for fips_code in fips_in_state_list:
                df_fips_yield_years = df_yield.filter(pl.col("fips_full") == fips_code)
                min_year_for_this_fips = df_fips_yield_years.select(pl.col("year").min()).item()
                max_year_for_this_fips_indiv = df_fips_yield_years.select(pl.col("year").max()).item()
                for year_to_fill in range(min_year_for_this_fips, max_year_for_this_fips_indiv + 1):
                    nan_aggs = {"fips_full": fips_code, "year": year_to_fill}
                    for prefix in ["GS", "SGF"]:
                        nan_aggs.update({
                            f"GDD_{prefix}": np.nan, f"KDD_{prefix}": np.nan, f"TMAX_AVG_{prefix}": np.nan,
                            f"PREC_{prefix}": np.nan, f"CHD_{prefix}": np.nan 
                        })
                    all_yearly_weather_aggregates.append(nan_aggs)
            continue 
            
        cleaned_df_for_state_batch, errors_for_this_state_load = fetch_and_clean_daily_weather_batched(
            fips_in_state_list, current_state_fetch_start_year, current_state_fetch_end_year
        )

        if errors_for_this_state_load:
            print(f"  S3 Loading Errors for state batch (FIPS prefix {state_fips_prefix}, {len(errors_for_this_state_load)} files/attempts):")
            for fips_in_batch_for_error_log in fips_in_state_list: 
                 if fips_in_batch_for_error_log not in all_s3_load_errors_summary:
                    all_s3_load_errors_summary[fips_in_batch_for_error_log] = []
                 all_s3_load_errors_summary[fips_in_batch_for_error_log].extend(errors_for_this_state_load)
            for error_info in errors_for_this_state_load[:3]: 
                print(f"    - File for {error_info['yyyymm']}: {error_info['error_type']} - {error_info['error_message'][:100]}...")
            if len(errors_for_this_state_load) > 3:
                print(f"    ... and {len(errors_for_this_state_load) - 3} more S3 loading errors for this state batch.")
        
        if cleaned_df_for_state_batch is None or cleaned_df_for_state_batch.height == 0:
            print(f"  No cleaned daily weather data available for FIPS in state {state_fips_prefix}. Aggregates will be NaN.")
            for fips_code in fips_in_state_list:
                df_fips_yield_years = df_yield.filter(pl.col("fips_full") == fips_code)
                min_year_for_this_fips = df_fips_yield_years.select(pl.col("year").min()).item()
                max_year_for_this_fips_indiv = df_fips_yield_years.select(pl.col("year").max()).item()
                for year_to_fill in range(max(overall_fetch_start_year, min_year_for_this_fips), max_year_for_this_fips_indiv + 1):
                    nan_aggs = {"fips_full": fips_code, "year": year_to_fill}
                    for prefix in ["GS", "SGF"]:
                        nan_aggs.update({
                            f"GDD_{prefix}": np.nan, f"KDD_{prefix}": np.nan, f"TMAX_AVG_{prefix}": np.nan,
                            f"PREC_{prefix}": np.nan, f"CHD_{prefix}": np.nan 
                        })
                    all_yearly_weather_aggregates.append(nan_aggs)
        else:
            cleaned_df_for_state_batch = cleaned_df_for_state_batch.with_columns(
                pl.col("date").dt.year().alias("year_daily") 
            )
            for fips_code in fips_in_state_list:
                # print(f"    Aggregating for FIPS: {fips_code}") # Less verbose
                df_single_fips_daily_data = cleaned_df_for_state_batch.filter(pl.col("fips") == fips_code)
                df_fips_yield_years = df_yield.filter(pl.col("fips_full") == fips_code)
                min_year_for_this_fips = df_fips_yield_years.select(pl.col("year").min()).item()
                max_year_for_this_fips_indiv = df_fips_yield_years.select(pl.col("year").max()).item()
                relevant_yield_years_for_fips = sorted(
                    df_fips_yield_years.filter(pl.col("year") >= current_state_fetch_start_year) 
                                        .select("year").unique().to_series().to_list()
                )
                if df_single_fips_daily_data.height == 0 and relevant_yield_years_for_fips:
                    # print(f"      No daily data for FIPS {fips_code} in the fetched batch. Setting aggregates to NaN.") # Less verbose
                    for year_to_aggregate in relevant_yield_years_for_fips:
                        nan_aggs = {"fips_full": fips_code, "year": year_to_aggregate}
                        for prefix in ["GS", "SGF"]:
                            nan_aggs.update({
                                f"GDD_{prefix}": np.nan, f"KDD_{prefix}": np.nan, f"TMAX_AVG_{prefix}": np.nan,
                                f"PREC_{prefix}": np.nan, f"CHD_{prefix}": np.nan
                            })
                        all_yearly_weather_aggregates.append(nan_aggs)
                elif relevant_yield_years_for_fips:
                    for year_to_aggregate in relevant_yield_years_for_fips:
                        daily_data_for_this_year_fips = df_single_fips_daily_data.filter(pl.col("year_daily") == year_to_aggregate)
                        if daily_data_for_this_year_fips.height == 0:
                            yearly_aggs = {"fips_full": fips_code, "year": year_to_aggregate}
                            for prefix in ["GS", "SGF"]:
                                yearly_aggs.update({
                                    f"GDD_{prefix}": np.nan, f"KDD_{prefix}": np.nan, f"TMAX_AVG_{prefix}": np.nan,
                                    f"PREC_{prefix}": np.nan, f"CHD_{prefix}": np.nan
                                })
                        else:
                            yearly_aggs = aggregate_weather_to_yearly(daily_data_for_this_year_fips, year_to_aggregate)
                            yearly_aggs["fips_full"] = fips_code 
                        all_yearly_weather_aggregates.append(yearly_aggs)
            for fips_code in fips_in_state_list:
                df_fips_yield_years = df_yield.filter(pl.col("fips_full") == fips_code)
                #nclimgrid-daily starts in 1951, so yield years before that get NaN weather
                pre_data_yield_years = sorted(
                    df_fips_yield_years.filter(pl.col("year") < overall_fetch_start_year) 
                                        .select("year").unique().to_series().to_list()
                )
                for year_to_fill_nan in pre_data_yield_years:
                    nan_aggs = {"fips_full": fips_code, "year": year_to_fill_nan}
                    for prefix in ["GS", "SGF"]:
                         nan_aggs.update({
                            f"GDD_{prefix}": np.nan, f"KDD_{prefix}": np.nan, f"TMAX_AVG_{prefix}": np.nan,
                            f"PREC_{prefix}": np.nan, f"CHD_{prefix}": np.nan
                        })
                    all_yearly_weather_aggregates.append(nan_aggs)
        state_end_time = time.time()
        print(f"  Finished processing state batch (FIPS prefix {state_fips_prefix}) in {state_end_time - state_start_time:.2f} seconds.")

    #--- Chunk 6: Merge results and save ---
    if not all_yearly_weather_aggregates:
        print("\nNo yearly weather aggregates were generated. Cannot merge or save.")
    else:
        #define the schema for the aggregated weather DataFrame
        #all agronomic variables should be Float64 to accomodate NaNs
        aggs_schema = {
            "fips_full": pl.String, "year": pl.Int64,
            "GDD_GS": pl.Float64, "KDD_GS": pl.Float64, "TMAX_AVG_GS": pl.Float64,
            "PREC_GS": pl.Float64, "CHD_GS": pl.Float64, 
            "GDD_SGF": pl.Float64, "KDD_SGF": pl.Float64, "TMAX_AVG_SGF": pl.Float64,
            "PREC_SGF": pl.Float64, "CHD_SGF": pl.Float64
        }
        df_final_weather_aggs = pl.DataFrame(all_yearly_weather_aggregates, schema=aggs_schema)
        
        print(f"\n--- Processing Complete: Total yearly aggregate records: {df_final_weather_aggs.height} ---")
        # print("Sample of aggregated weather data (first 5 rows):")
        # print(df_final_weather_aggs.head(5))
        
        print("\n--- Merging weather aggregates with yield data ---")
        if "fips_full" not in df_yield.columns or df_yield["fips_full"].dtype != pl.String:
             df_yield = df_yield.with_columns( #recreate if missing or wrong type
                (pl.col("state").cast(pl.String) + pl.col("county").cast(pl.String).str.zfill(3)).alias("fips_full")
            )
        if df_yield["year"].dtype != pl.Int64:
             df_yield = df_yield.with_columns(pl.col("year").cast(pl.Int64))
        
        #left join, a county-year with no weather keeps its yield row and just gets NaNs
        df_merged = df_yield.join(
            df_final_weather_aggs,
            on=["fips_full", "year"],
            how="left"
        )
        print(f"\nShape of merged DataFrame: {df_merged.shape}")
        # print("Sample of merged DataFrame (first 5 rows with new columns):")
        # cols_to_show = ["year", "fips_full"] + [col for col in df_final_weather_aggs.columns if col not in ["fips_full", "year"]]
        # cols_to_show = [col for col in cols_to_show if col in df_merged.columns] 
        # if cols_to_show: print(df_merged.head(5).select(cols_to_show))

        print(f"\nSaving merged data to: {output_csv_path}")
        try:
            df_merged.write_csv(output_csv_path)
            print(f"Successfully saved data to {output_csv_path}")
        except Exception as e:
            print(f"Error saving merged data to CSV: {e}")

    if all_s3_load_errors_summary:
        print("\n--- Summary of S3 File Loading Errors Encountered ---")
        for fips_key, errors_list in all_s3_load_errors_summary.items():
            print(f"  FIPS {fips_key}: {len(errors_list)} file(s)/attempts had errors.")
            unique_file_errors = {} 
            for err_detail in errors_list:
                error_signature = (err_detail['yyyymm'], err_detail['error_type']) 
                if err_detail['file_path'] not in unique_file_errors or unique_file_errors[err_detail['file_path']] > 0 : 
                    path_display = err_detail['file_path'][-40:] if err_detail['file_path'] != 'N/A - Batch Attempt' else 'Batch Attempt'
                    print(f"    - File/Attempt {err_detail['yyyymm']} ({path_display}): {err_detail['error_type']}")
                    if err_detail['file_path'] not in unique_file_errors:
                         unique_file_errors[err_detail['file_path']] = 2 
                    else:
                         unique_file_errors[err_detail['file_path']] -=1
    else:
        print("\nNo S3 file loading errors were reported across all FIPS processing.")

    overall_end_time = time.time()
    print(f"\n--- Agricultural Weather Data Processing Finished in {overall_end_time - overall_start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()