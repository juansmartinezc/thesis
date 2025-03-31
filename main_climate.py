from data.get_climate_data import get_stations_data, get_station_data
from data.get_scan_data import get_usda_stations, filter_scan_data, get_usda_weather_data
from data.get_usda_data import get_usda_quick_stats
from eda.years_histogram import plot_years_histogram, plot_crops_states, filter_top_states
from graphics.plot_states import plot_states_with_filtered_stations, plot_selected_states, plot_states_with_filtered_stations_voronoi
import pandas as pd
import os
from dotenv import load_dotenv
from data.utils import calculate_statisticals_months

load_dotenv()

source_data_directory = 'source_data'
stations_df = get_stations_data()
elements = "TMAX, TMIN, TAVG, PRCP, SMS:-2:1, SMS:-4:1, SMS:-8:1, SMS:-20:1, SMS:-40:1"
duration = "MONTHLY"
stations_data_list = get_station_data(stations_df, duration, elements)

result = pd.DataFrame()
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

        print(f"✓ Añadido: {unique_column_name}")

    if not dfs:
        print(f"⚠️ No hay datos válidos para la estación {stationTriplet}, se omite.")
        continue

    # Merge de todos los elementos
    weather_df = dfs[0]
    for df_temp in dfs[1:]:
        common_cols = weather_df.columns.intersection(df_temp.columns).difference(['year', 'month'])
        if not common_cols.empty:
            print(f"⚠️ Conflicto de columnas: {common_cols.tolist()}")
        weather_df = weather_df.merge(df_temp, on=['year', 'month'], how='outer')

    # Añadir metadatos de estación
    weather_df['stationTriplet'] = stationTriplet
    weather_df['latitude'] = latitude
    weather_df['longitude'] = longitude
    dfs_result.append(weather_df)

# Concatenar todos los resultados
result = pd.concat(dfs_result, ignore_index=True)
result.to_csv('result.csv', index=False)
