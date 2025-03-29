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

def get_soil_data(lat, lon, max_retries=5):
    """
    Obtiene datos de 'silt' (0-30 cm) de la API de SoilGrids.
    Implementa reintentos con backoff si encuentra un error 429 (Too Many Requests).
    """
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=silt"
    
    delay = 2  # segundos de espera inicial si recibimos 429
    for attempt in range(1, max_retries+1):
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Petición exitosa
                data = response.json()
                layers = data.get("properties", {}).get("layers", [])
                
                # Hallar la capa 'silt'
                silt_layer = next((layer for layer in layers if layer["name"] == "silt"), None)
                if silt_layer:
                    depths = silt_layer.get("depths", [])
                    mean_values = []
                    
                    # Recorrer subcapas y filtrar las que caen dentro de 0-30 cm
                    for d in depths:
                        top = d["range"]["top_depth"]
                        bottom = d["range"]["bottom_depth"]
                        if bottom > 0 and top < 30:  # rango 0-30 cm
                            val = d["values"].get("mean", None)
                            if val is not None:
                                mean_values.append(val)
                    
                    if mean_values:
                        avg_silt = round(sum(mean_values)/len(mean_values), 2)
                        return {"lat": lat, "lon": lon, "silt_0_30cm": avg_silt}
                
                # Si no hay capa silt o está vacía, devolver None
                return {"lat": lat, "lon": lon, "silt_0_30cm": None}
            
            elif response.status_code == 429:
                # Límite de peticiones (rate limit)
                print(f"Recibido 429 (Too Many Requests) en intento {attempt}/{max_retries} para {lat}, {lon}")
                
                # Chequear cabecera Retry-After (si la hay)
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = int(retry_after)
                    except ValueError:
                        pass  # Si no es un entero, usamos el delay actual
                
                if attempt < max_retries:
                    print(f"Esperando {delay} seg antes de reintentar...")
                    time.sleep(delay)
                    # Exponential backoff: duplicamos el tiempo de espera
                    delay *= 2
                else:
                    print("Se alcanzó el número máximo de reintentos con 429.")
                    return {"lat": lat, "lon": lon, "silt_0_30cm": None}
            
            else:
                print(f"Respuesta inesperada: {response.status_code} (Intento {attempt}/{max_retries}).")
                # Podemos imprimir el contenido para debug:
                print(response.text)
                
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    return {"lat": lat, "lon": lon, "silt_0_30cm": None}
        
        except requests.exceptions.RequestException as e:
            # Cualquier otro error de conexión
            print(f"Excepción (Intento {attempt}/{max_retries}) para {lat}, {lon}: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                return {"lat": lat, "lon": lon, "silt_0_30cm": None}
    
    # Si no se retornó nada durante los reintentos
    return {"lat": lat, "lon": lon, "silt_0_30cm": None}

def main():
    # Nombre del archivo de caché
    cached_file = "soil_data_iowa.csv"
    
    # 3. Comprobar si ya tenemos un CSV con resultados anteriores
    if os.path.exists(cached_file):
        df_cache = pd.read_csv(cached_file)
    else:
        df_cache = pd.DataFrame(columns=["lat", "lon", "silt_0_30cm"])
    
    # Lista final donde iremos acumulando todos los datos
    all_data = []
    
    # Convertimos df_cache en una lista/dict para buscar rápido
    # (Podríamos usar un set de tuplas (lat,lon), pero con float hay que tener cuidado con equivalencias)
    existing_coords = set(zip(df_cache["lat"].values, df_cache["lon"].values))
    
    for (lat, lon) in coordinates:
        # Ver si lat-lon ya están en caché
        if (lat, lon) in existing_coords:
            # Filtrar la fila específica en df_cache
            row = df_cache[(df_cache["lat"] == lat) & (df_cache["lon"] == lon)].iloc[0]
            silt_val = row["silt_0_30cm"]
            data_dict = {"lat": lat, "lon": lon, "silt_0_30cm": silt_val}
        else:
            # Llamar a la API
            data_dict = get_soil_data(lat, lon, max_retries=5)
            
            # Actualizar df_cache con la nueva fila
            new_row = pd.DataFrame([data_dict])
            df_cache = pd.concat([df_cache, new_row], ignore_index=True)
            # Guardar en CSV cada vez que obtenemos un dato nuevo
            df_cache.to_csv(cached_file, index=False)
            
            # Espera fija tras cada request exitoso o fallido para no saturar
            time.sleep(3)
        
        all_data.append(data_dict)
    
    # Convertir la data final a DataFrame
    final_df = pd.DataFrame(all_data)
    
    # Mostrar el resultado en pantalla
    print("\n--- RESULTADOS FINALES ---")
    print(final_df)

if __name__ == "__main__":
    main()
