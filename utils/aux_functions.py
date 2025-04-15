import os
import pandas as pd
from data.get_soil_data import get_soil_data
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")

#base_dir = Path(__file__).resolve().parent.parent
#source_data_directory = f"{base_dir} / {source_data_directory}"

def scan_stations_in_corn_belt_states(
        stations: pd.DataFrame | str,
        fips_map: dict[str, str],
        state_col: str = "stateCode"
) -> pd.DataFrame:
    """
    Return the subset of SCAN stations that are located in the states
    contained in `fips_map`.

    Parameters
    ----------
    stations : DataFrame | str
        • A pandas DataFrame that already holds the station data, **or**  
        • A path/URL to a CSV file that can be read with `pd.read_csv`.
    fips_map : dict
        Mapping of 2‑digit state FIPS codes to 2‑letter abbreviations.
    state_col : str, default "stateCode"
        Name of the column that contains state abbreviations.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only stations whose state is in `fips_map`.
    """
    
    # Normalise the column to string and filter
    stations[state_col] = stations[state_col].astype(str).str.upper()
    target_states: set[str] = set(fips_map.values())

    return stations[stations[state_col].isin(target_states)].reset_index(drop=True)



def create_historical_monthly_climate_data_by_scan_station(stations_data_list): 
    dfs_result = []
    for station_data in stations_data_list:
        stationTriplet = station_data['stationTriplet']
        latitude = station_data['latitude']
        longitude = station_data['longitude']
        dfs = []
        for element_info in station_data['data']:
            element_code = element_info['stationElement']['elementCode']
            depth = element_info['stationElement'].get('heightDepth', '')
            # Nombre único para la columna
            if depth:
                unique_column_name = f"{element_code}_{depth}".replace(":", "_")
            else:
                unique_column_name = element_code.replace(":", "_")
            df_temp = pd.DataFrame(element_info['values'])
            # Validamos si 'value', 'year' y 'month' existen
            if not {'value', 'year', 'month'}.issubset(df_temp.columns):
                print(f"Saltando {unique_column_name} por falta de columnas necesarias")
                continue
            # Renombrar la columna 'value' por su nombre único
            df_temp = df_temp.rename(columns={'value': unique_column_name})
            dfs.append(df_temp)
            #print(f"✓ Añadido: {unique_column_name}")
        if not dfs:
            print(f"⚠️ No hay datos válidos para la estación {stationTriplet}, se omite.")
            continue

        # Merge de todos los elementos
        climate_df = dfs[0]
        for df_temp in dfs[1:]:
            common_cols = climate_df.columns.intersection(df_temp.columns).difference(['year', 'month'])
            if not common_cols.empty:
                print(f"⚠️ Conflicto de columnas: {common_cols.tolist()}")
            climate_df = climate_df.merge(df_temp, on=['year', 'month'], how='outer')
        # Añadir metadatos de estación
        climate_df['stationTriplet'] = stationTriplet
        climate_df['latitude'] = latitude
        climate_df['longitude'] = longitude
        dfs_result.append(climate_df)
    monthly_climate_data_by_scan_stations = pd.concat(dfs_result, ignore_index=True)
    return monthly_climate_data_by_scan_stations

def save_historical_monthly_climate_data_by_scan_station(monthly_climate_data_by_scan_station): 
    os.makedirs('source_data', exist_ok=True)
    monthly_climate_data_by_scan_station.to_csv(f'{source_data_directory}/historical_monthly_climate_data_by_scan_station.csv', index=False)

def impute_soil_moisture_depth_8(df):
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values(["stationTriplet", "date"])

    # --- 2. Interpolación lineal dentro de cada estación -------------------------
    df["SMS_-8_interp"] = (
        df.groupby("stationTriplet", group_keys=False)["SMS_-8"]
        .apply(lambda s: s.interpolate(method="linear", limit_direction="both"))
    )

    # --- 3. Fallback climatológico (stationId‑mes) -------------------------------
    df["SMS_-8"] = df["SMS_-8_interp"].fillna(
        df.groupby(["stationTriplet", "month"])["SMS_-8_interp"].transform("mean")
    )

    df = df.drop(columns=["SMS_-8_interp"])
    return df


