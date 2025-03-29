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
elements="TMAX, TMIN, TAVG, PRCP"
duration = "MONTHLY"
weather_list = get_station_data(stations_df, duration,  elements)
station_data = weather_list[0]
stationTriplet = station_data['stationTriplet']
latitude = station_data['latitude']
longitude = station_data['longitude']

# “station_data['data']” es una lista donde cada elemento corresponde a un elemento (TAVG, TMAX, TMIN).
# Vamos a crear un DataFrame para cada elemento y luego hacer un merge (o un pivot).
dfs = []
for element_info in station_data['data']:
    element_code = element_info['stationElement']['elementCode']  # TAVG, TMAX o TMIN
    # Creamos un DataFrame temporal con las columnas month, year y value
    df_temp = pd.DataFrame(element_info['values'])
    # Renombramos “value” como el nombre del elemento (TAVG, TMAX, etc.)
    df_temp = df_temp.rename(columns={'value': element_code})
    dfs.append(df_temp)

# Hacemos merge sucesivos para unificar los tres DataFrames en uno solo
weather_df = dfs[0]
for df_temp in dfs[1:]:
    weather_df = weather_df.merge(df_temp, on=['year', 'month'], how='outer')

# Añadimos la información de la estación (triplet, lat, lon)
weather_df['stationTriplet'] = stationTriplet
weather_df['latitude'] = latitude
weather_df['longitude'] = longitude

print(weather_df)
weather_df.to_csv('weather.csv')