import os
import pandas as pd
from utils.states_codes import state_fips_to_abbr
from dotenv import load_dotenv
from data.get_nasa import get_climate_missing_values, save_climate_missing_values
from data.get_crop_yield_data import get_crop_yield, save_crop_yield_data
from data.get_soil_data import get_soil_scan_stations_dataframe, save_soil_scan_stations_dataframe
from utils.aux_functions import create_historical_monthly_climate_data_by_scan_station, save_historical_monthly_climate_data_by_scan_station, impute_soil_moisture_depth_8, scan_stations_in_corn_belt_states
from data.get_climate_data import get_scan_stations_data, get_station_data, save_scan_stations_data
from data.get_centroids import get_counties_centroids, save_counties_centroids, get_counties_centroids_cornbelt, save_counties_centroids_cornbelt, assign_scan_station_to_cb_yield_counties
from data.merge_data import merge_monthly_scan_stations_with_soil, save_monthly_climate_soil_data_by_scan_station, merge_counties_crop_yield_with_scan_stations, merge_counties_crop_yield_with_historical_scan_stations, save_crop_yield_scan_stations, save_counties_crop_yield_with_historical_scan_stations, save_historical_monthly_climate_imputed_data_by_scan_stations

load_dotenv()

#################################################
#######Declaracion Variables de entorno##########
#################################################

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
elements = "TMAX, TMIN, TAVG, PRCP, SMS:-8:1"
soil_elements = ["phh2o", "ocd", "cec", "sand", "silt", "clay"]
duration = "MONTHLY"

'''
os.makedirs('source_data', exist_ok=True)

#################################################
##############Variables##########################
#################################################

#################################################
###Obtener los datos de clima###
#################################################

#Elementos que se quieren obtener

#Obtener el listado de todas las estaciones
scan_stations_df = get_scan_stations_data()

#Obtener las estaciones unicamente que estan en los estados de Corn belt
#scan_stations_df = scan_stations_in_corn_belt_states(scan_stations_df, state_fips_to_abbr, "stateCode") 

#Se guardan los datos de las estaciones
save_scan_stations_data(scan_stations_df)

#Obtener un lista con datos de cada una de las estaciones
stations_data_list = get_station_data(scan_stations_df, duration, elements)

#Función para obtener los datos de cada estación
historical_monthly_climate_data_by_scan_stations = create_historical_monthly_climate_data_by_scan_station(stations_data_list)
save_historical_monthly_climate_data_by_scan_station(historical_monthly_climate_data_by_scan_stations)

#################################################
#######Imput climate data#######################
#################################################

#########Impute SMS-8###########################
historical_monthly_climate_imputed_data_by_scan_stations = impute_soil_moisture_depth_8(historical_monthly_climate_data_by_scan_stations)
save_historical_monthly_climate_imputed_data_by_scan_stations(historical_monthly_climate_imputed_data_by_scan_stations)



######################################################
#Obtener los missing values de temperatura de la NASA#
######################################################

######Filtrar los meses de interes##############
###### abril (4) a octubre (10) ################
months_of_interest = list(range(4, 11))
historical_monthly_climate_data_apr_sept_by_scan_stations = historical_monthly_climate_imputed_data_by_scan_stations[historical_monthly_climate_imputed_data_by_scan_stations['month'].isin(months_of_interest)].reset_index(drop=True)

# Guardar el resultado en un nuevo archivo
historical_monthly_climate_data_apr_sept_by_scan_stations.to_csv(f'{source_data_directory}/historical_monthly_climate_data_apr_sept_by_scan_stations.csv', index=False)

historical_monthly_climate_data_apr_sept_by_scan_stations = get_climate_missing_values(historical_monthly_climate_data_apr_sept_by_scan_stations)
save_climate_missing_values(historical_monthly_climate_data_apr_sept_by_scan_stations)


######################################################
########Obtener los datos de suelo####################
######################################################

historical_monthly_climate_data_apr_sept_by_scan_stations = pd.read_csv(f'{source_data_directory}/historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa.csv')

##Obtener los datos de cada estacion climatica para realizar la consulta de suelo sobre esa ubicación geografica.
station_coords = historical_monthly_climate_data_apr_sept_by_scan_stations.groupby('stationTriplet')[['latitude', 'longitude']].first().reset_index()
print(station_coords)

## Funcion para obtener el soil data de cada estacion
soil_data_by_scan_stations = get_soil_scan_stations_dataframe(station_coords, soil_elements)
save_soil_scan_stations_dataframe(soil_data_by_scan_stations)



#################################################
###Obtener los datos de rendimiento de cultivo###
#################################################

historical_monthly_climate_data_apr_sept_by_scan_stations = pd.read_csv(f"{source_data_directory}/historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa.csv")
#Obtener los datos de rendimiento
crop_yield_df = get_crop_yield()
save_crop_yield_data(crop_yield_df)

'''
scan_stations_df = pd.read_csv(f"{source_data_directory}/scan_stations.csv")
crop_yield_df = pd.read_csv(f'{source_data_directory}/crop_yield.csv')
historical_monthly_climate_soil_data_apr_sept_by_scan_stations = pd.read_csv(f'{source_data_directory}/historical_monthly_climate_soil_data_apr_sept_by_scan_station.csv')
#Obtener los centroides por cada condado
counties_centroids, counties_cornbelt_wgs84 = get_counties_centroids(crop_yield_df, state_fips_to_abbr)
save_counties_centroids(counties_centroids)

#Filtrar solo los condados que pertenecen al cornbelt
centroids_cornbelt_counties_crop_yield = get_counties_centroids_cornbelt(counties_centroids)
save_counties_centroids_cornbelt(centroids_cornbelt_counties_crop_yield)

counties_nearest_usda_station = assign_scan_station_to_cb_yield_counties(counties_cornbelt_wgs84, scan_stations_df)

#################################################
#############Merge data##########################
#################################################

## Funcion para hacer merge entre datos de suelo por estacion y los datos mensuales de cada estacion.
'''
historical_monthly_climate_soil_data_apr_sept_by_scan_stations = merge_monthly_scan_stations_with_soil(soil_data_by_scan_stations, historical_monthly_climate_data_apr_sept_by_scan_stations)
save_monthly_climate_soil_data_by_scan_station(historical_monthly_climate_soil_data_apr_sept_by_scan_stations)
'''

## Funcion para hacer cruce de datos de rendimiento de cada estado con los datos de clima y suelo.
crop_yield_by_scan_stations = merge_counties_crop_yield_with_scan_stations(centroids_cornbelt_counties_crop_yield, counties_nearest_usda_station)
save_crop_yield_scan_stations(crop_yield_by_scan_stations)


######## Check point##########
monthly_historical_climate_soil_crop_yield_data_by_scan_stations = merge_counties_crop_yield_with_historical_scan_stations(crop_yield_by_scan_stations, historical_monthly_climate_soil_data_apr_sept_by_scan_stations)
save_counties_crop_yield_with_historical_scan_stations(monthly_historical_climate_soil_crop_yield_data_by_scan_stations)

#######Proceso de imputacion de soil moisture -8#####


print("Proceso terminado exitosamente")