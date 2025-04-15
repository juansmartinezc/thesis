from data.get_scan_data import get_usda_stations, filter_scan_data, get_usda_weather_data
from dotenv import load_dotenv
import os
import pandas as pd
import logging

load_dotenv()

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")

def get_scan_stations_data():
    stations_data = get_usda_stations(networks="SNTL")
    scan_stations_df = filter_scan_data(stations_data)
    return scan_stations_df

def save_scan_stations_data(scan_stations_df):
    scan_stations_df.to_csv(f"{source_data_directory}/scan_stations.csv")

def read_stations_data():
    stations_df = pd.read_csv(f"{source_data_directory}/stations.csv")
    return stations_df

def get_station_data(stations_df, duration, elements="TMAX,TMIN,PREC"):
    results = []

    for station in stations_df.itertuples(index=False):
        logging.info("Consultando estación %s", station.stationTriplet)
        weather = get_usda_weather_data(
            station.stationTriplet, elements, station.beginDate, duration
        )

        if not weather:
            logging.warning("Sin datos para %s (respuesta vacía o error).", station.stationTriplet)
            continue

        # Asegúrate de qué estructura recibes
        if isinstance(weather, list):
            first = weather[0] if weather else {}
        else:
            first = weather

        if "data" not in first or not first["data"]:
            logging.warning("Respuesta sin 'data' para %s: %s", station.stationTriplet, first)
            continue

        results.append({
            "stationTriplet": station.stationTriplet,
            "latitude": station.latitude,
            "longitude": station.longitude,
            "data": first["data"],
        })

    return results