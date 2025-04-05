import os
import pandas as pd
from dotenv import load_dotenv
from data.get_climate_data import get_stations_data, get_station_data
from data.get_soil_grid import get_soil_data

source_data_directory = 'source_data'

def create_climate_dateframe(stations_data_list): 
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
            #print(f"✓ Añadido: {unique_column_name}")
        if not dfs:
            print(f"⚠️ No hay datos válidos para la estación {stationTriplet}, se omite.")
            continue

        # Merge de todos los elementos
        climate_df = dfs[0]
        for df_temp in dfs[1:]:
            common_cols = climate_df.columns.intersection(df_temp.columns).difference(['year', 'month'])
            if not common_cols.empty:
                print(f"⚠️ Conflicto de columnas: {common_cols.tolist()}")
            climate_df = climate_df.merge(df_temp, on=['year', 'month'], how='outer')
        # Añadir metadatos de estación
        climate_df['stationTriplet'] = stationTriplet
        climate_df['latitude'] = latitude
        climate_df['longitude'] = longitude
        dfs_result.append(climate_df)
    monthly_climate_data_by_station = pd.concat(dfs_result, ignore_index=True)
    os.makedirs('source_data', exist_ok=True)
    monthly_climate_data_by_station.to_csv(f'{source_data_directory}/monthly_climate_data_by_station.csv', index=False)
    return monthly_climate_data_by_station

def get_soil_dataframe(sowing_month, station_coords, elements = ["phh2o", "ocd", "cec", "sand", "silt", "clay"]):
    soils_list = []
    for idx, sowing_month in station_coords.iterrows():
        latitude = sowing_month['latitude']
        longitude = sowing_month['longitude']
        soil_df = get_soil_data(latitude, longitude, elements, depth_range = (15,30), max_retries=5) 
        if soil_df is not None and not soil_df.empty:
            soils_list.append(soil_df)
    # Concatenar todos los resultados en un único DataFrame
    soil_data_by_station = pd.concat(soils_list, ignore_index=True)
    os.makedirs('source_data', exist_ok=True)
    soil_data_by_station.to_csv(f'{source_data_directory}/soil_data_by_station.csv', index=False)
    return soil_data_by_station


load_dotenv()

#################################################
##############Variables##########################
#################################################
#Elementos que se quieren obtener
elements = "TMAX, TMIN, TAVG, PRCP, SMS:-2:1, SMS:-4:1, SMS:-8:1, SMS:-20:1, SMS:-40:1"
soil_elements = ["phh2o", "ocd", "cec", "sand", "silt", "clay"]
duration = "MONTHLY"

#Obtener el listado de todas las estaciones
stations_df = get_stations_data()

#Obtener un lista con datos de cada una de las estaciones
stations_data_list = get_station_data(stations_df, duration, elements)

#Función para obtener los datos de cada estación
monthly_climate_data_by_station = create_climate_dateframe(stations_data_list)

# Concatenar todos los resultados
station_coords = monthly_climate_data_by_station.groupby('stationTriplet')[['latitude', 'longitude']].first().reset_index()

## Funcion para obtener el soil data
soil_data_by_station = get_soil_dataframe(monthly_climate_data_by_station, station_coords, soil_elements)

os.makedirs('source_data', exist_ok=True)

## Hacemos merge entre los dataframes de clima y suelo.
monthly_climate_soil_data_by_station = soil_data_by_station.merge(monthly_climate_data_by_station, how = 'inner', on = ['latitude', 'longitude'])
monthly_climate_soil_data_by_station.to_csv(f'{source_data_directory}/monthly_climate_soil_data_by_station.csv', index=False)
