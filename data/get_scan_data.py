import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

def get_usda_stations(networks=None, station_ids=None):
    """
    Llama a la API del USDA para obtener estaciones de monitoreo.

    Parámetros:
        networks (str, opcional): Código(s) de la red de estaciones separados por comas.
        station_ids (str, opcional): ID(s) de las estaciones separados por comas.
        bbox (str, opcional): Cuadro delimitador en formato "minLon,minLat,maxLon,maxLat".

    Retorna:
        dict: Respuesta en formato JSON con los datos de las estaciones.
    """
    url = f"{os.environ.get('USDA_API_URL')}/stations"

    params = {}

    if networks:
        params["networkCds"] = networks
    if station_ids:
        params["stationIds"] = station_ids

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Lanza un error si el request falla
        stations_data = response.json()
        stations_df = pd.DataFrame(stations_data)
        return stations_df
    except requests.exceptions.RequestException as e:
        print(f"Error al llamar a la API: {e}")
        return None

def filter_scan_data(stations_df):
    stations_df = stations_df[stations_df['networkCode'] == 'SCAN'] 
    return stations_df

def get_usda_weather_data(station_triplets, elements, begin_date, duration):
    """
    Llama a la API del USDA para obtener datos meteorológicos.

    Parámetros:
        station_triplets (str): Identificador de la estación en formato 'ID:STATE:NETWORK'.
        elements (str): Elementos climáticos a obtener, separados por comas (Ej: 'TMAX,TMIN,PRCP,RHUM').
        begin_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.

    Retorna:
        dict: Respuesta en formato JSON con los datos meteorológicos.
    """
    url = f"{os.environ.get('USDA_API_URL')}/data"
    params = {
        "stationTriplets": station_triplets,
        "elements": elements,
        "beginDate": begin_date,
        "duration": duration
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Lanza un error si el request falla
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error al llamar a la API: {e}")
        return None
