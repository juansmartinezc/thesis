import os
import pandas as pd
from dotenv import load_dotenv
from data.get_crop_yield_data import get_crop_yield
from utils.aux_functions import create_climate_dateframe
from data.get_soil_data import get_soil_dataframe, save_soil_dataframe
from data.get_climate_data import get_scan_stations_data, get_station_data, save_stations_data
from data.merge_data import merge_monthly_scan_stations_with_soil, save_monthly_climate_soil_data_by_station
from data.get_centroids import get_counties_centroids, save_counties_centroids, get_counties_centroids_cornbelt, save_counties_centroids_cornbelt, assign_scan_station_to_cb_yield_counties

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
stations_df = get_scan_stations_data()
#Se guardan los datos de las estaciones
save_stations_data(stations_df)
#Obtener un lista con datos de cada una de las estaciones
stations_data_list = get_station_data(stations_df, duration, elements)

#Función para obtener los datos de cada estación
monthly_climate_data_by_station = create_climate_dateframe(stations_data_list)

# Concatenar todos los resultados
station_coords = monthly_climate_data_by_station.groupby('stationTriplet')[['latitude', 'longitude']].first().reset_index()

## Funcion para obtener el soil data de cada estacion
soil_data_by_station = get_soil_dataframe(monthly_climate_data_by_station, station_coords, soil_elements)
save_soil_dataframe(soil_data_by_station)

#Obtener los datos de rendimiento
crop_yield_df = get_crop_yield()

#Obtener los centroides por cada condado
counties_centroids, counties_cornbelt_wgs84 = get_counties_centroids(crop_yield_df)
save_counties_centroids(crop_yield_df)

#Filtrar solo los condados que pertenecen al cornbelt
counties_centroids_cornbelt = get_counties_centroids_cornbelt(counties_centroids)
save_counties_centroids_cornbelt

counties_nearest_usda_station = assign_scan_station_to_cb_yield_counties(counties_cornbelt_wgs84, stations_df)

#################################################
#############Merge data##########################
#################################################
## Funcion para hacer merge entre datos de suelo por estacion y los datos mensuales de cada estacion.
monthly_climate_soil_data_by_station = merge_monthly_scan_stations_with_soil(soil_data_by_station, monthly_climate_data_by_station)
save_monthly_climate_soil_data_by_station(monthly_climate_soil_data_by_station)

merge_counties_crop_yield_with_scan_stations(centroids_cornbelt_counties_crop_yield, stations_voronoi)


