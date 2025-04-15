import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
source_directory = os.environ.get('SOURCE_DATA_DIRECTORY')

monthly_historical_climate_soil_data_by_station = pd.read_csv(f'{source_directory}/monthly_historical_climate_soil_data_by_station.csv')

# Filtrar solo los meses de abril (4) a octubre (10)
months_of_interest = list(range(4, 11))
filtered_df = monthly_historical_climate_soil_data_by_station[monthly_historical_climate_soil_data_by_station['month'].isin(months_of_interest)].reset_index(drop=True)

# Guardar el resultado en un nuevo archivo
filtered_path = "monthly_historical_apr_oct_climate_soil_data_by_station.csv"
filtered_df.to_csv(f'{source_directory}/{filtered_path}', index=False)


# Obtener combinaciones únicas de lat/lon/año
locations = filtered_df[['lat_centroid', 'lon_centroid', 'year']].drop_duplicates().head(3)
locations.to_csv('locations.csv')

# Definir los meses de interés (abril a octubre)
months = [f"{m:02d}" for m in range(4, 11)]

nasa_url = os.environ.get('NASA_API_URL')
parameters = "parameters=T2M_MAX,T2M_MIN,T2M,PRECTOTCORR&community=ag&"

# Función para obtener datos mensuales de NASA POWER por año
def get_nasa_data(lat, lon, year):
    url = (
        f"{nasa_url}"
        f"{parameters}"
        f"longitude={lon}&latitude={lat}&start={year}&end={year}&format=JSON"
    )
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json().get("properties", {}).get("parameter", {})
    result = []
    for month in months:
        key = f"{year}{month}"
        result.append({
            "lat_centroid": lat,
            "lon_centroid": lon,
            "year": year,
            "month": int(month),
            "T2M_MAX": data.get("T2M_MAX", {}).get(key),
            "T2M_MIN": data.get("T2M_MIN", {}).get(key),
            "T2M": data.get("T2M", {}).get(key),
            "PRECTOTCORR": data.get("PRECTOTCORR", {}).get(key)
        })
    return result

# Descargar todos los datos
climate_data = []
for _, row in locations.iterrows():
    print(f"Consultando {row['lat_centroid']}, {row['lon_centroid']}, {int(row['year'])}")
    result = get_nasa_data(row['lat_centroid'], row['lon_centroid'], int(row['year']))
    if result:
        climate_data.extend(result)
    time.sleep(1)  # Para no saturar la API

# Convertir a DataFrame y guardar
climate_df = pd.DataFrame(climate_data)
climate_df.to_csv(f'{source_directory}/nasa_climate_apr_oct.csv', index=False)
print("✅ Datos climáticos guardados en 'nasa_climate_apr_oct.csv'")