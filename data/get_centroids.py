import os
import numpy as np
import pandas as pd
import pandas as pd
import geopandas as gpd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from utils.states_codes import state_fips_to_abbr, state_alpha_to_fips
from dotenv import load_dotenv

load_dotenv()

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
shape_files = "shape_files"

# 1) Leer el shapefile de condados
counties_gdf = gpd.read_file("shape_files/country/cb_2018_us_county_500k.shp")  # Ejemplo

def get_counties_centroids(crop_yield_df, state_fips_to_abbr):
    corn_belt_fips = list(state_fips_to_abbr.keys())
    
    # 2) Filtramos
    counties_cornbelt = counties_gdf[counties_gdf["STATEFP"].isin(corn_belt_fips)]

    # Reproyectamos a un CRS apropiado (p.ej. Albers)
    counties_cornbelt_proj = counties_cornbelt.to_crs(epsg=5070)

    # Ahora calculamos centroides en coordenadas proyectadas (metros)
    counties_cornbelt_proj["centroid"] = counties_cornbelt_proj.geometry.centroid

    # 3) Reproyectar el GeoDataFrame principal (que todavía contiene polígonos, no centroides)
    counties_cornbelt_wgs84 = counties_cornbelt_proj.to_crs(epsg=4326)

    # 4) Calcular centroides con la nueva geometría en EPSG:4326
    counties_cornbelt_wgs84["centroid_wgs84"] = counties_cornbelt_wgs84.geometry.centroid

    # Extraer lat/lon
    # 5) Extraer lat/lon
    counties_cornbelt_wgs84["lat_centroid"] = counties_cornbelt_wgs84["centroid_wgs84"].y
    counties_cornbelt_wgs84["lon_centroid"] = counties_cornbelt_wgs84["centroid_wgs84"].x

    # 6 Crear la columna STATEFP a partir de state_alpha
    crop_yield_df["STATEFP"] = crop_yield_df["state_alpha"].map(state_alpha_to_fips)
    crop_yield_df["COUNTYFP"] = crop_yield_df["county_code"].astype(str).str.zfill(3)
    counties_cornbelt_wgs84["COUNTYFP"] = counties_cornbelt_wgs84["COUNTYFP"].astype(str).str.zfill(3)

    counties_centroids = crop_yield_df.merge(
        counties_cornbelt_wgs84[["STATEFP", "COUNTYFP", "lat_centroid", "lon_centroid"]],
        on=["STATEFP", "COUNTYFP"],
        how="left"
    )
    return counties_centroids, counties_cornbelt_wgs84

def save_counties_centroids(counties_centroids):
    counties_centroids.to_csv(f'{source_data_directory}/counties_centroids.csv')

def get_counties_centroids_cornbelt(counties_centroids):
    centroids_cornbelt_counties_crop_yield = counties_centroids.dropna(subset=["lat_centroid", "lon_centroid"])
    return centroids_cornbelt_counties_crop_yield

def save_counties_centroids_cornbelt(centroids_cornbelt_counties_crop_yield):
    centroids_cornbelt_counties_crop_yield.to_csv(f"{source_data_directory}/centroids_cornbelt_counties_crop_yield.csv")

def assign_scan_station_to_cb_yield_counties(counties_cornbelt_wgs84, stations_df):
    #########################################
    ###########Cruce con Voronoi#############
    #########################################

    # 1) Reproyectar condados a CRS proyectado
    counties_cornbelt_proj = counties_cornbelt_wgs84.to_crs(epsg=5070)
    print("CRS antes de reproyectar:", counties_cornbelt_wgs84.crs)
    # Asegura que esté en EPSG:4326
    if counties_cornbelt_wgs84.crs is None:
        counties_cornbelt_wgs84 = counties_cornbelt_wgs84.set_crs("EPSG:4326")
    elif counties_cornbelt_wgs84.crs.to_string() != "EPSG:4326":
        counties_cornbelt_wgs84 = counties_cornbelt_wgs84.to_crs(epsg=4326)

    # ------------------------------------------------------------------------
    # 2. Reproyectar condados a un sistema planar (por ej. EPSG:5070 Albers)
    # ------------------------------------------------------------------------
    counties_cornbelt_proj = counties_cornbelt_wgs84.to_crs(epsg=5070)

    # ------------------------------------------------------------------------
    # 3. Calcular centroides en la proyección (para sjoin con Voronoi luego)
    # ------------------------------------------------------------------------
    counties_cornbelt_proj["centroid"] = counties_cornbelt_proj.geometry.centroid
    centroids_gdf = counties_cornbelt_proj.copy()
    centroids_gdf["geometry"] = centroids_gdf["centroid"]

    # ------------------------------------------------------------------------
    # 4. Cargar y reproyectar las estaciones (de stations.xlsx)
    # ------------------------------------------------------------------------
    # Asumimos que las columnas son "latitude" y "longitude".
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(
            stations_df["longitude"], 
            stations_df["latitude"]
        ),
        crs="EPSG:4326"  # Suponiendo coords en grados WGS84
    )

    stations_proj = stations_gdf.to_crs(epsg=5070)
    stations_proj["station_idx"] = range(len(stations_proj))  # Identificador interno

    # ------------------------------------------------------------------------
    # 5. Generar el Voronoi con scipy.spatial
    # ------------------------------------------------------------------------
    points_array = np.column_stack((
        stations_proj.geometry.x,
        stations_proj.geometry.y
    ))
    vor = Voronoi(points_array)

    # ------------------------------------------------------------------------
    # 6. Convertir las regiones Voronoi en polígonos shapely
    #    Nota: Las regiones infinitas se marcan con -1 y requieren "clipping".
    # ------------------------------------------------------------------------
    polygons = []
    station_indices = []

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        # Si es vacío o infinito, lo ignoramos aquí.
        if not region or -1 in region:
            continue
        # Crear polígono
        poly_coords = [vor.vertices[j] for j in region]
        poly = Polygon(poly_coords)
        polygons.append(poly)
        station_indices.append(i)  # i es el índice de la estación en stations_proj

    voronoi_gdf = gpd.GeoDataFrame(
        {"station_idx": station_indices},
        geometry=polygons,
        crs=stations_proj.crs  # EPSG:5070
    )

    # ------------------------------------------------------------------------
    # 7. "Clipping" para recortar Voronoi al área del Corn Belt
    #    (evita polígonos que se extienden al infinito)
    # ------------------------------------------------------------------------
    # Disolver todos los condados del Corn Belt en un solo polígono
    cornbelt_union = counties_cornbelt_proj.dissolve()  # produce 1 fila con la unión

    # Recortar
    voronoi_clipped = gpd.clip(voronoi_gdf, cornbelt_union)

    # ------------------------------------------------------------------------
    # 8. Hacer un sjoin de los centroides de condado con los polígonos Voronoi
    #    => Asignar estación a cada condado
    # ------------------------------------------------------------------------
    joined = gpd.sjoin(
        centroids_gdf,        # Puntos (centroides)
        voronoi_clipped,      # Polígonos Voronoi
        how="left",
        predicate="within"    # O "op='within'" en versiones < 0.10
    )

    # En 'joined' aparece 'station_idx' indicando la estación más cercana.

    # ------------------------------------------------------------------------
    # 9. Vincular con datos de la estación (nombre, ID, etc.)
    # ------------------------------------------------------------------------
    joined = joined.merge(
        stations_proj[["station_idx", "stationTriplet", "stationId", "latitude", "longitude", "name"]], 
        on="station_idx",
        how="left"
    )

    # ------------------------------------------------------------------------
    # 10. Unir con tu DataFrame de rendimiento, si aún no lo has hecho
    # ------------------------------------------------------------------------
    # Si tu DF de rendimiento ya se "mergeó" con condados, puedes relacionar
    # la key (STATEFP, COUNTYFP, etc.). Si 'joined' también la conserva, 
    # harías algo como:
    # df_rend_merged = df_rend_merged.merge(joined[["STATEFP","COUNTYFP","stationId","name"]], on=["STATEFP","COUNTYFP"], how="left")

    # En este ejemplo, 'joined' ya contiene las columnas del condado + la estación.

    # ------------------------------------------------------------------------
    # 11. Exportar o continuar con tu análisis
    # ------------------------------------------------------------------------

    # Quitar la columna "centroid" para que solo quede la geometría principal
    print(joined.columns)

    joined_single_geom = joined.drop(columns=["centroid","centroid_wgs84"])  

    # Ahora sí, esto guardará sin problema
    os.makedirs(f'{shape_files}/counties_usda', exist_ok=True)
    joined_single_geom.to_file(f"{shape_files}/counties_usda/cornbelt_counties_nearest_usda_station.shp")

    # Para CSV, debes quitar la geometría (por ejemplo joined.drop(columns="geometry"))
    joined.drop(columns="geometry").to_csv(f"{source_data_directory}/cornbelt_counties_nearest_usda_station.csv", index=False)
    return joined