import os
import time
import requests
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, List
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")

def get_soil_data(
    lat: float,
    lon: float,
    elements: Iterable[str] | None = None,
    depth_range: Tuple[int, int] = (15, 30),
    max_retries: int = 5
) -> pd.DataFrame:
    """
    Obtiene datos de suelo para la profundidad especificada (por defecto 15‑30 cm)
    desde la API de SoilGrids y devuelve el percentil Q0.5 de los elementos pedidos.

    Parameters
    ----------
    lat, lon : float
        Coordenadas del punto de interés.
    elements : Iterable[str] | None
        Lista de propiedades a solicitar.  Si es None se usa ["silt"].
    depth_range : (int, int)
        Profundidad (cm) superior e inferior.  Ej.: (15, 30).
    max_retries : int
        Número máximo de reintentos ante errores 429 u otros fallos de conexión.

    Returns
    -------
    pd.DataFrame
        Una fila con lat, lon y las columnas <element>_<top>_<bottom>cm_Q0.5.
    """

    if elements is None:
        elements = ["silt"]

    url = os.environ.get("SOILGRID_API_URL")

    # Construimos el parámetro depth en el formato que espera la API, p. ej. "15-30cm"
    depth_param = f"{depth_range[0]}-{depth_range[1]}cm"

    params = {
        "lat": lat,
        "lon": lon,
        "property": elements,
        "depth": depth_param
    }

    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                layers = data.get("properties", {}).get("layers", [])

                result = {"latitude": lat, "longitude": lon}

                for element in elements:
                    layer = next((layer for layer in layers if layer["name"] == element), None)
                    if layer:
                        depths = layer.get("depths", [])
                        value_q05 = None

                        for d in depths:
                            top = d["range"]["top_depth"]
                            bottom = d["range"]["bottom_depth"]
                            if (top, bottom) == depth_range:
                                value_q05 = d["values"].get("Q0.5")
                                break

                        col_name = f"{element}"
                        result[col_name] = value_q05
                    else:
                        result[f"{element}"] = None

                return pd.DataFrame([result])

            elif response.status_code == 429:
                print(f"429 Too Many Requests en intento {attempt}/{max_retries} para {lat}, {lon}")
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = int(retry_after)
                    except ValueError:
                        pass
                if attempt < max_retries:
                    print(f"Esperando {delay} s antes de reintentar…")
                    time.sleep(delay)
                    delay *= 2
                else:
                    break  # se sale al final y devuelve NaN

            else:
                print(f"Error {response.status_code} - {response.text}")
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    break

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión en intento {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                break

    # Si llega aquí es que hubo error: devolvemos NaNs
    return pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        **{f"{e}": None for e in elements}
    }])

def get_soil_scan_stations_dataframe(
    station_coords: pd.DataFrame, 
    elements: List[str] = ["phh2o", "ocd", "cec", "sand", "silt", "clay"],
    sleep_time: int = 12,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Obtiene los datos de SoilGrids para cada estación en `station_coords`,
    respetando el límite de 6 llamadas por minuto (1 cada 10 segundos).
    
    Parameters
    ----------
    station_coords : pd.DataFrame
        DataFrame con columnas 'latitude', 'longitude' y 'stationTriplet'.
    elements : List[str]
        Lista de propiedades edáficas a consultar.
    sleep_time : int
        Tiempo de espera entre llamadas (por defecto 10 segundos).
    verbose : bool
        Muestra progreso si True.
    
    Returns
    -------
    pd.DataFrame
        Datos de suelo por estación.
    """
    soils_list = []

    iterator = tqdm(station_coords.iterrows(), total=len(station_coords)) if verbose else station_coords.iterrows()

    for idx, station_coord in iterator:
        latitude = station_coord['latitude']
        longitude = station_coord['longitude']
        triplet = station_coord['stationTriplet']

        try:
            soil_df = get_soil_data(latitude, longitude, elements, depth_range=(15, 30), max_retries=5)
            soil_df['stationTriplet'] = triplet

            if soil_df is not None and not soil_df.empty:
                soils_list.append(soil_df)

        except Exception as e:
            print(f"Error en estación {triplet} ({latitude}, {longitude}): {e}")
            continue

        time.sleep(sleep_time)

    if soils_list:
        return pd.concat(soils_list, ignore_index=True)
    else:
        return pd.DataFrame()  # Vacío si todas fallan

def save_soil_scan_stations_dataframe(soil_data_by_scan_stations):
    os.makedirs('source_data', exist_ok=True)
    soil_data_by_scan_stations.to_csv(f'{source_data_directory}/historical_soil_data_by_scan_stations.csv', index=False)