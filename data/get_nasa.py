import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

nasa_url = os.environ.get('NASA_API_URL')
source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
nasa_url_parameters = "parameters=T2M_MAX,T2M_MIN,T2M,PRECTOTCORR&community=ag&"
months = [f"{m:02d}" for m in range(4, 11)]

api_to_df_cols = {
    'T2M_MAX'      : 'TMAX',
    'T2M_MIN'      : 'TMIN',
    'T2M'          : 'TAVG',
    'PRECTOTCORR'  : 'PRCP',
    'WS10M'        : 'WS10M',
    'RH2M'         : 'RH2M'          # ← faltaba
}
def get_nasa_data(lat: float, lon: float, year: int, months_needed: list[int]) -> list[dict]:
    url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point?"
        f"parameters={','.join(api_to_df_cols.keys())}&"
        "community=ag&"
        f"longitude={lon}&latitude={lat}&"
        f"start={year}&end={year}&format=JSON"
    )
    print(f"Consultando URL: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()["properties"]["parameter"]
    except Exception as e:
        print(f"Error consultando la NASA: {e}")
        return []

    result = []
    for month in months_needed:
        key = f"{year}{month:02d}"
        row = {
            "latitude":  lat,
            "longitude": lon,
            "year":      year,
            "month":     month
        }
        # rellenamos las variables climáticas de forma genérica
        for api_name, col_name in api_to_df_cols.items():
            row[col_name] = data.get(api_name, {}).get(key)
        result.append(row)

    return result
# Función principal para rellenar los valores faltantes
def get_climate_missing_values(df):
    climate_cols = list(api_to_df_cols.values())  # ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'WS10M', 'RH2M']

    # Asegurar que todas las columnas estén presentes en el DataFrame original
    for col in climate_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Filas con valores faltantes
    missing_mask = df[climate_cols].isnull().any(axis=1)
    locations = df[missing_mask][['latitude', 'longitude', 'year', 'month']].drop_duplicates()
    locations.to_csv(f'{source_data_directory}/locations.csv')
    print(f"Total de ubicaciones únicas con faltantes: {len(locations)}")

    climate_data = []

    grouped = locations.groupby(['latitude', 'longitude', 'year'])
    for (lat, lon, year), group in grouped:
        months_needed = group['month'].tolist()
        print(f"→ Consultando ({lat}, {lon}) en {year} para meses: {months_needed}")
        result = get_nasa_data(lat, lon, year, months_needed)

        if result:
            climate_data.extend(result)

    if not climate_data:
        print("⚠️ No se recuperaron datos climáticos.")
        return df

    df_nasa = pd.DataFrame(climate_data)
    df_merged = df.merge(df_nasa, on=['latitude', 'longitude', 'year', 'month'], how='left', suffixes=('', '_nasa'))

    for col in climate_cols:
        df_merged[col] = df_merged[col].fillna(df_merged[f"{col}_nasa"])
        df_merged.drop(columns=[f"{col}_nasa"], inplace=True)

    print("✅ Valores climáticos completados exitosamente.")
    return df_merged

def save_climate_missing_values(climate_missing_values, file_name = 'historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa.csv'):
    climate_missing_values.to_csv(f'{source_data_directory}/{file_name}', index=False)
    print("✅ Datos climáticos guardados en historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa.csv'")