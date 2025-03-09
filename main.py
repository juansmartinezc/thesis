from data.get_scan_data import get_usda_stations, filter_scan_data, get_usda_weather_data
from data.get_usda_data import get_usda_quick_stats
from eda.years_histogram import plot_years_histogram, plot_crops_states
import pandas as pd
import os

from dotenv import load_dotenv

load_dotenv()

'''
stations_data = get_usda_stations(networks="SNTL", bbox="-120,35,-110,40")
stations_df = filter_scan_data(stations_data)
station_1 = stations_df.loc[0,:]
print(station_1)
stationTriplet = station_1['stationTriplet']
elements="TMAX,TMIN,PRCP"
beginDate = station_1['beginDate']
endDate = station_1['endDate']
weather_df = get_usda_weather_data(stationTriplet, elements, beginDate, endDate)
print(weather_df)
'''
import os
import pandas as pd

'''
Estados con datos disponibles: ['AL', 'AR', 'AZ', 'CA', 'CO', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MD', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NJ', 'NM', 'NY', 'OH', 'OK', 'OR', 'PA', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI', 'WV', 'WY']

Estados sin datos disponibles: ['AS', 'CT', 'GU', 'HI', 'MA', 'ME', 'MP', 'NH', 'NV', 'PR', 'RI', 'VI', 'VT']

'''
'''
# Lista de códigos de estado correctos
state_dict = {
    "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California",
    "CO": "Colorado", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "IA": "Iowa", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "MD": "Maryland",
    "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi",
    "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota",
    "NE": "Nebraska", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VA": "Virginia",
    "WA": "Washington", "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming"
}

# Obtener API Key desde variables de entorno
usda_api_key = os.environ.get("USDA_API_KEY")

# Listas para almacenar los estados con y sin datos
states_with_data = []
states_without_data = []
dfs = []

# Iterar sobre los estados y obtener los datos
for state_code, state_name in state_dict.items():
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
else:
    print("No se obtuvieron datos de ningún estado.")

# Mostrar los estados que sí tienen datos y los que no
print("\nEstados con datos disponibles:", states_with_data)
print("\nEstados sin datos disponibles:", states_without_data)

'''
file_path = "crop_yield.xlsx"

# Cargar la hoja de datos
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name="Sheet1")

##plot_years_histogram(df)
plot_crops_states(df)