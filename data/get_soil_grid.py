import os
import time
import requests
import numpy as np
import pandas as pd

# 1. Definición de los límites de Iowa
min_lat, max_lat = 40.3, 43.5
min_lon, max_lon = -96.7, -90.1

# Número de puntos en la malla total
num_points = 20

# 2. Generar la malla de puntos con meshgrid
lat_vals = np.linspace(min_lat, max_lat, int(np.sqrt(num_points)))
lon_vals = np.linspace(min_lon, max_lon, int(np.sqrt(num_points)))
lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals)

# Conversion a listas 1D de lat y lon
coordinates = list(zip(lat_grid.ravel(), lon_grid.ravel()))

def get_soil_data(lat, lon, max_retries=5, elements=None):
    """
    Obtiene datos de suelo (0-30 cm) desde la API de SoilGrids para los elementos solicitados.
    """

    if elements is None:
        elements = ["silt"]  # valor por defecto

    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"

    params = {
        "lat": lat,
        "lon": lon,
        "property": elements
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
                        mean_values = [
                            d["values"].get("mean", None)
                            for d in depths
                            if d["range"]["top_depth"] < 30 and d["range"]["bottom_depth"] > 0
                        ]
                        mean_values = [v for v in mean_values if v is not None]

                        if mean_values:
                            avg_val = round(sum(mean_values) / len(mean_values), 2)
                        else:
                            avg_val = None
                    else:
                        avg_val = None
                    
                    result[f"{element}_0_30cm"] = avg_val

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
                    print(f"Esperando {delay} segundos antes de reintentar...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print("Demasiados intentos fallidos.")
                    return pd.DataFrame([{"latitude": lat, "longitude": lon, **{f"{e}_0_30cm": None for e in elements}}])

            else:
                print(f"Error {response.status_code} - {response.text}")
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    return pd.DataFrame([{"latitude": lat, "longitude": lon, **{f"{e}_0_30cm": None for e in elements}}])

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión en intento {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                return pd.DataFrame([{"latitude": lat, "longitude": lon, **{f"{e}_0_30cm": None for e in elements}}])

    return pd.DataFrame([{"latitude": lat, "longitude": lon, **{f"{e}_0_30cm": None for e in elements}}])