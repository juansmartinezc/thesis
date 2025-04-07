import os
import pandas as pd
from dotenv import load_dotenv
from utils.aux_functions import create_climate_dateframe, get_soil_dataframe
from data.get_climate_data import get_scan_stations_data, get_station_data

load_dotenv()

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

#Obtener un lista con datos de cada una de las estaciones
stations_data_list = get_station_data(stations_df, duration, elements)

#Función para obtener los datos de cada estación
monthly_climate_data_by_station = create_climate_dateframe(stations_data_list)

# Concatenar todos los resultados
station_coords = monthly_climate_data_by_station.groupby('stationTriplet')[['latitude', 'longitude']].first().reset_index()

## Funcion para obtener el soil data
soil_data_by_station = get_soil_dataframe(monthly_climate_data_by_station, station_coords, soil_elements)

## Hacemos merge entre los dataframes de clima y suelo.
monthly_climate_soil_data_by_station = soil_data_by_station.merge(monthly_climate_data_by_station, how = 'inner', on = ['latitude', 'longitude'])
monthly_climate_soil_data_by_station.to_csv(f'{source_data_directory}/monthly_climate_soil_data_by_station.csv', index=False)