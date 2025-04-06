from data.get_scan_data import get_usda_stations, filter_scan_data, get_usda_weather_data
from data.get_usda_data import get_usda_quick_stats
from eda.years_histogram import plot_years_histogram, plot_crops_states, filter_top_states
from graphics.plot_states import plot_states_with_filtered_stations, plot_selected_states, plot_states_with_filtered_stations_voronoi
from data.get_climate_data import get_stations_data

import pandas as pd
import os
from dotenv import load_dotenv
from utils.states_codes import states_dict

'''
Estados con datos disponibles: ['AL', 'AR', 'AZ', 'CA', 'CO', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MD', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NJ', 'NM', 'NY', 'OH', 'OK', 'OR', 'PA', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI', 'WV', 'WY']

Estados sin datos disponibles: ['AS', 'CT', 'GU', 'HI', 'MA', 'ME', 'MP', 'NH', 'NV', 'PR', 'RI', 'VI', 'VT']

'''

load_dotenv()

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
stations_data = get_stations_data(networks="SNTL")
stations_df = filter_scan_data(stations_data)



# Lista de códigos de estado correctos

# Obtener API Key desde variables de entorno
usda_api_key = os.environ.get("USDA_API_KEY")

# Listas para almacenar los estados con y sin datos
states_with_data = []
states_without_data = []
dfs = []

# Iterar sobre los estados y obtener los datos
for state_code, state_name in states_dict.items():
    print(f"Llamando API para: {state_name} ({state_code})")
    response = get_usda_quick_stats(api_key=usda_api_key, state_alpha=state_code)
    if response and "data" in response and response["data"]:  # Verifica que "data" no esté vacío
        df = pd.DataFrame(response["data"])  # Convertir en DataFrame
        dfs.append(df)
        states_with_data.append(state_code)  # Agregar a la lista de estados con datos
    else:
        states_without_data.append(state_code)  # Agregar a la lista de estados sin datos

# Concatenar todos los DataFrames en uno solo
if dfs:
    crop_yield_df = pd.concat(dfs, ignore_index=True)
    print("Datos recopilados exitosamente.")
    print(crop_yield_df.head())  # Muestra las primeras filas
    os.makedirs('source_data', exist_ok=True)
    crop_yield_df.to_csv(f'{source_data_directory}/crop_yield.csv')
else:
    print("No se obtuvieron datos de ningún estado.")

# Mostrar los estados que sí tienen datos y los que no
print("\nEstados con datos disponibles:", states_with_data)
print("\nEstados sin datos disponibles:", states_without_data)

'''
file_path = f'{source_data_directory}/crop_yield.csv'

# Cargar la hoja de datos
#xls = pd.ExcelFile(file_path)
df = pd.read_csv(file_path)

plot_years_histogram(df)
plot_crops_states(df)
df_filtered = filter_top_states(df)
df_stations = pd.read_excel('stations.xlsx')
plot_selected_states(df_filtered)
# Llamar a la función con los datos filtrados
plot_states_with_filtered_stations(df_filtered, df_stations)
plot_states_with_filtered_stations_voronoi(df_filtered, df_stations)
#plot_states_with_filtered_stations_voronoi(df_filtered, df_stations)
#print("Estados con más de 2000 registros:", df_filtered)
'''