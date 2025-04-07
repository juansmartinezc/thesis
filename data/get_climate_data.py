from data.get_scan_data import get_usda_stations, filter_scan_data, get_usda_weather_data
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")

def get_scan_stations_data():
    stations_data = get_usda_stations(networks="SNTL")
    stations_df = filter_scan_data(stations_data)
    return stations_df

def save_stations_data(stations_df):
    stations_df.to_csv(f"{source_data_directory}/stations.csv")

def read_stations_data():
    stations_df = pd.read_csv(f"{source_data_directory}/stations.csv")
    return stations_df

def get_station_data(stations_df, duration, elements = "TMAX,TMIN,PREC"):
    results = []
    counter = 1
    for station in stations_df.itertuples(index=True):
        print(f"consultando la estacion: {station}")
        station_triplet = station.stationTriplet
        station_begin_date = station.beginDate
        #station_begin_date = "2024-02-02"
        weather_list = get_usda_weather_data(station_triplet, elements, station_begin_date, duration)
        station_data = {
            "stationTriplet": station_triplet,
            "latitude": station.latitude,
            "longitude": station.longitude,
            "data": weather_list[0]['data']  # Esto debería ser la lista de elementos como TMAX, TMIN, PRCP
        }
        results.append(station_data)
        counter = counter + 1
        if counter == 2:
            break
        #print(f"los resultados son: {results}")
    return results
    