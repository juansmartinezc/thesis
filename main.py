import os
import pandas as pd
from dotenv import load_dotenv
from data.get_crop_yield_data import get_crop_yield
from data.get_soil_data import get_soil_scan_stations_dataframe, save_soil_scan_stations_dataframe
from utils.aux_functions import create_monthly_climate_data_by_scan_station, save_monthly_climate_data_by_scan_station
from data.get_climate_data import get_scan_stations_data, get_station_data, save_scan_stations_data
from data.get_centroids import get_counties_centroids, save_counties_centroids, get_counties_centroids_cornbelt, save_counties_centroids_cornbelt, assign_scan_station_to_cb_yield_counties
from data.merge_data import merge_monthly_scan_stations_with_soil, save_monthly_climate_soil_data_by_station, merge_counties_crop_yield_with_scan_stations, merge_counties_crop_yield_with_historical_scan_stations, save_crop_yield_scan_stations, save_counties_crop_yield_with_historical_scan_stations

load_dotenv()

#################################################
#######Declaracion Variables de entorno##########
#################################################
source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
os.makedirs('source_data', exist_ok=True)

#################################################
##############Variables##########################
#################################################

#Elementos que se quieren obtener
elements = "TMAX, TMIN, TAVG, PRCP, SMS:-2:1, SMS:-4:1, SMS:-8:1, SMS:-20:1, SMS:-40:1"
soil_elements = ["phh2o", "ocd", "cec", "sand", "silt", "clay"]
duration = "MONTHLY"

#Obtener el listado de todas las estaciones
scan_stations_df = get_scan_stations_data()
#Se guardan los datos de las estaciones
save_scan_stations_data(scan_stations_df)
#Obtener un lista con datos de cada una de las estaciones
stations_data_list = get_station_data(scan_stations_df, duration, elements)

#Función para obtener los datos de cada estación
monthly_climate_data_by_scan_stations = create_monthly_climate_data_by_scan_station(stations_data_list)
save_monthly_climate_data_by_scan_station(monthly_climate_data_by_scan_stations)
# Concatenar todos los resultados
station_coords = monthly_climate_data_by_scan_stations.groupby('stationTriplet')[['latitude', 'longitude']].first().reset_index()
print(station_coords)

## Funcion para obtener el soil data de cada estacion
soil_data_by_scan_stations = get_soil_scan_stations_dataframe(station_coords, soil_elements)
save_soil_scan_stations_dataframe(soil_data_by_scan_stations)

#Obtener los datos de rendimiento
crop_yield_df = get_crop_yield()

#Obtener los centroides por cada condado
counties_centroids, counties_cornbelt_wgs84 = get_counties_centroids(crop_yield_df)
save_counties_centroids(counties_centroids)

#Filtrar solo los condados que pertenecen al cornbelt
cornbelt_yield_county_centroids = get_counties_centroids_cornbelt(counties_centroids)
save_counties_centroids_cornbelt(cornbelt_yield_county_centroids)

counties_nearest_usda_station = assign_scan_station_to_cb_yield_counties(counties_cornbelt_wgs84, scan_stations_df)

#################################################
#############Merge data##########################
#################################################

## Funcion para hacer merge entre datos de suelo por estacion y los datos mensuales de cada estacion.
monthly_climate_soil_data_by_station = merge_monthly_scan_stations_with_soil(soil_data_by_scan_stations, monthly_climate_data_by_scan_stations)
save_monthly_climate_soil_data_by_station(monthly_climate_soil_data_by_station)

## Funcion para hacer cruce de datos de rendimiento de cada estado con los datos de clima y suelo.
crop_yield_by_scan_stations = merge_counties_crop_yield_with_scan_stations(cornbelt_yield_county_centroids, counties_nearest_usda_station)
save_crop_yield_scan_stations(crop_yield_by_scan_stations)

monthly_historical_climate_soil_data_by_scan_stations = merge_counties_crop_yield_with_historical_scan_stations(crop_yield_by_scan_stations, monthly_climate_soil_data_by_station)
save_counties_crop_yield_with_historical_scan_stations(monthly_historical_climate_soil_data_by_scan_stations)

print("Proceso terminado exitosamente")