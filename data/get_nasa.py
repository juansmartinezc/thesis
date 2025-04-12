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
    'T2M_MAX': 'TMAX',
    'T2M_MIN': 'TMIN',
    'PRECTOTCORR': 'PRCP'
}
def get_nasa_data(lat, lon, year, months_needed):
    url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point?"
        "parameters=T2M_MAX,T2M_MIN,T2M,PRECTOTCORR&"
        "community=ag&"
        f"longitude={lon}&latitude={lat}&"
        f"start={year}&end={year}&"
        "format=JSON"
    )

    print(f"Consultando URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("properties", {}).get("parameter", {})
    except Exception as e:
        print(f"Error consultando la NASA: {e}")
        return []

    result = []
    for month in months_needed:
        key = f"{year}{str(month).zfill(2)}"
        result.append({
        "lat_centroid": lat,
        "lon_centroid": lon,
        "year": int(year),
        "month": int(month),
        api_to_df_cols['T2M_MAX']: data.get("T2M_MAX", {}).get(key),
        api_to_df_cols['T2M_MIN']: data.get("T2M_MIN", {}).get(key),
        api_to_df_cols['PRECTOTCORR']: data.get("PRECTOTCORR", {}).get(key)
    })

    return result

# Función principal para rellenar los valores faltantes
def get_climate_missing_values(df):
    
    climate_cols = list(api_to_df_cols.values())  # ['TMAX', 'TMIN', 'PRCP']

    # Filas con valores faltantes
    missing_mask = df[climate_cols].isnull().any(axis=1)
    locations = df[missing_mask][['lat_centroid', 'lon_centroid', 'year', 'month']].drop_duplicates()
    locations.to_csv(f'{source_data_directory}/locations.csv')
    print(f"Total de ubicaciones únicas con faltantes: {len(locations)}")

    climate_data = []

    # Agrupar por coordenadas y año para consultar solo una vez por grupo
    grouped = locations.groupby(['lat_centroid', 'lon_centroid', 'year'])
    
    for (lat, lon, year), group in grouped:
        months_needed = group['month'].tolist()
        print(f"→ Consultando ({lat}, {lon}) en {year} para meses: {months_needed}")
        result = get_nasa_data(lat, lon, year, months_needed)

        if result:
            climate_data.extend(result)

        #time.sleep(1)  # Evitar saturar la API

    # Si no hay datos devueltos, salir
    if not climate_data:
        print("⚠️ No se recuperaron datos climáticos.")
        return df

    # Convertimos y hacemos merge
    df_nasa = pd.DataFrame(climate_data)
    df_merged = df.merge(df_nasa, on=['lat_centroid', 'lon_centroid', 'year', 'month'], how='left', suffixes=('', '_nasa'))

    # Reemplazamos NaNs solamente donde haya datos nuevos
    for col in climate_cols:
        df_merged[col] = df_merged[col].fillna(df_merged[f"{col}_nasa"])
        df_merged.drop(columns=[f"{col}_nasa"], inplace=True)

    print("✅ Valores climáticos completados exitosamente.")
    return df_merged


def save_climate_missing_values(climate_missing_values):
    climate_missing_values.to_csv(f'{source_data_directory}/missing_values_climate_apr_oct.csv', index=False)
    print("✅ Datos climáticos guardados en 'nasa_climate_apr_oct.csv'")