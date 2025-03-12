import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

import plotly.graph_objects as go
import numpy as np

from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
def plot_selected_states(dict_states):
    """
    Genera un mapa coroplético de EE.UU. destacando los estados con datos en dict_states
    y muestra la cantidad de registros junto con el nombre del estado.
    """

    # 1. Cargar el shapefile de los estados de EE.UU.
    gdf_usa = gpd.read_file("shape_files/state/cb_2018_us_state_20m.shp")  # Asegúrate de la ruta correcta

    # 2. Normalizar nombres de los estados en el shapefile
    gdf_usa["NAME"] = gdf_usa["NAME"].str.upper()

    # 3. Filtrar solo los 48 estados contiguos de EE.UU. (excluir Alaska, Hawái y Puerto Rico)
    exclude_states = {"ALASKA", "PUERTO RICO", "HAWAII"}
    gdf_usa = gdf_usa[~gdf_usa["NAME"].isin(exclude_states)]

    # 4. Unir los datos con el GeoDataFrame del mapa de EE.UU.
    gdf_usa["value"] = gdf_usa["NAME"].map(dict_states)

    # 5. Rellenar con 0 en estados sin datos
    gdf_usa["value"] = gdf_usa["value"].fillna(0)

    # 6. Calcular coordenadas del centroide de cada estado (para colocar los textos)
    gdf_usa["centroid"] = gdf_usa.geometry.centroid
    gdf_usa["lat"] = gdf_usa["centroid"].y
    gdf_usa["lon"] = gdf_usa["centroid"].x

    # 7. Crear el mapa interactivo con Plotly
    fig = px.choropleth(
        gdf_usa,
        geojson=gdf_usa.geometry,
        locations=gdf_usa.index,
        color="value",
        hover_name="NAME",
        color_continuous_scale="Blues",
        labels={"value": "Cantidad de registros"},
        title="Cantidad de registros por estado en EE.UU.",
        width=1000,  
        height=700
    )

    # 8. Agregar etiquetas con el nombre del estado y la cantidad de registros
    for _, row in gdf_usa.iterrows():
        fig.add_scattergeo(
            lat=[row["lat"]],
            lon=[row["lon"]],
            text=[f"{row['NAME']}<br>{int(row['value'])} registros"],  # Nombre + número de registros
            mode="text",
            showlegend=False,
            textfont=dict(size=7, color="black", family="Arial Bold")
        )

    # 9. Ajustar el enfoque del mapa y mejorar visualización
    fig.update_geos(
        fitbounds="locations",  
        visible=False,
        projection_type="albers usa"
    )

    # 10. Ajustar layout para mejor presentación
    fig.update_layout(
        margin={"r":0, "t":50, "l":0, "b":0},
        title_font_size=20
    )

    fig.show()

def plot_states_with_filtered_stations(dict_states, df_stations):
    """
    Genera un mapa coroplético de EE.UU. destacando los estados con datos en dict_states
    y superpone solo las estaciones de monitoreo ubicadas en los estados de interés.
    """

    # 1. Cargar el shapefile de los estados de EE.UU. desde GeoJSON en línea
    gdf_usa = gpd.read_file("https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json")

    # 2. Normalizar nombres de los estados en el shapefile
    gdf_usa["NAME"] = gdf_usa["NAME"].str.upper()

    # 3. Filtrar solo los 48 estados contiguos de EE.UU.
    exclude_states = {"ALASKA", "PUERTO RICO", "HAWAII"}
    gdf_usa = gdf_usa[~gdf_usa["NAME"].isin(exclude_states)]

    # 4. Unir los datos con el GeoDataFrame del mapa de EE.UU.
    gdf_usa["value"] = gdf_usa["NAME"].map(dict_states).fillna(0)

    # 5. Crear el mapa coroplético de los estados
    fig = px.choropleth(
        gdf_usa,
        geojson=gdf_usa.geometry,
        locations=gdf_usa.index,
        color="value",
        hover_name="NAME",
        color_continuous_scale="Blues",
        labels={"value": "Cantidad de registros"},
        title="Cantidad de registros por estado en EE.UU.",
        width=1000,
        height=700
    )

    # 6. Filtrar estaciones que estén en estados válidos
    df_stations_filtered = df_stations[~df_stations["stateCode"].isin({"AK", "HI", "PR"})]

    # 7. Agregar puntos de las estaciones filtradas al mapa
    fig.add_trace(go.Scattergeo(
        lon=df_stations_filtered["longitude"],
        lat=df_stations_filtered["latitude"],
        text=df_stations_filtered["name"],
        mode="markers",
        marker=dict(size=5, color="red", symbol="circle"),
        name="Estaciones"
    ))

    # 8. Ajustar el enfoque del mapa y mejorar visualización
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="albers usa"
    )

    # 9. Ajustar layout para mejor presentación
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title_font_size=20
    )

    fig.show()

def plot_states_with_filtered_stations_voronoi(dict_states, df_stations):
    """
    Genera un mapa coroplético de EE.UU. destacando los estados con datos en dict_states,
    superpone las estaciones de monitoreo y dibuja los polígonos de Voronoi en torno
    a cada estación (limitados a los 48 estados contiguos).
    
    Parámetros:
    -----------
    dict_states: dict
        Diccionario con nombre de estado (en mayúsculas) como clave y algún valor
        para colorear (por ejemplo, número de registros).
    df_stations: pd.DataFrame
        DataFrame con al menos las columnas:
         - 'longitude' (float): Longitud de la estación
         - 'latitude'  (float): Latitud de la estación
         - 'name'      (str)  : Nombre de la estación (opcional para hover)
         - 'stateCode' (str)  : Código del estado (por ej. 'CA', 'NY', etc.)
    """

    # -------------------------------------------------------------------
    # 1. Cargar shapefile de EE.UU. (GeoJSON en línea) y normalizar nombres
    # -------------------------------------------------------------------
    gdf_usa = gpd.read_file(
        "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json"
    )
    gdf_usa["NAME"] = gdf_usa["NAME"].str.upper()

    # -------------------------------------------------------------------
    # 2. Filtrar solo los 48 estados contiguos (excluir Alaska, Hawái, Puerto Rico)
    # -------------------------------------------------------------------
    exclude_states = {"ALASKA", "HAWAII", "PUERTO RICO"}
    gdf_usa = gdf_usa[~gdf_usa["NAME"].isin(exclude_states)]

    # -------------------------------------------------------------------
    # 3. Agregar valores desde dict_states a gdf_usa (para el coroplético)
    # -------------------------------------------------------------------
    gdf_usa["value"] = gdf_usa["NAME"].map(dict_states).fillna(0)

    # -------------------------------------------------------------------
    # 4. Crear mapa coroplético base (usa gdf_usa como shapefile)
    # -------------------------------------------------------------------
    fig = px.choropleth(
        gdf_usa,
        geojson=gdf_usa.geometry,
        locations=gdf_usa.index,
        color="value",
        hover_name="NAME",
        color_continuous_scale="Blues",
        labels={"value": "Cantidad de registros"},
        title="Cantidad de registros por estado en EE.UU.",
        width=1000,
        height=700
    )

    # -------------------------------------------------------------------
    # 5. Crear GeoDataFrame de estaciones y filtrar (sin AK, HI, PR)
    # -------------------------------------------------------------------
    df_stations_filtered = df_stations[
        ~df_stations["stateCode"].isin({"AK", "HI", "PR"})
    ].copy()
    
    # GeoDataFrame en EPSG:4326
    gdf_stations = gpd.GeoDataFrame(
        df_stations_filtered,
        geometry=gpd.points_from_xy(
            df_stations_filtered["longitude"], 
            df_stations_filtered["latitude"]
        ),
        crs="EPSG:4326"
    )

    # -------------------------------------------------------------------
    # 6. Proyectar estaciones a Albers USA (EPSG:5070) para Voronoi
    # -------------------------------------------------------------------
    gdf_stations_albers = gdf_stations.to_crs("EPSG:5070")

    # -------------------------------------------------------------------
    # 7. Calcular Voronoi en la proyección Albers
    # -------------------------------------------------------------------
    points_albers = np.column_stack((gdf_stations_albers.geometry.x,
                                     gdf_stations_albers.geometry.y))
    vor = Voronoi(points_albers)

    polygons = []
    for region_index in vor.regions:
        if not region_index:
            continue
        if -1 in region_index:
            # regiones infinitas, se ignoran
            continue
        polygon_coords = [vor.vertices[i] for i in region_index]
        poly = Polygon(polygon_coords)
        polygons.append(poly)

    # Crear GeoDataFrame de polígonos Voronoi (en Albers)
    gdf_voronoi_albers = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:5070")

    # -------------------------------------------------------------------
    # 8. Clip al contorno de EE.UU. para no mostrar Voronoi infinito
    # -------------------------------------------------------------------
    # Reproyectar gdf_usa a Albers
    gdf_usa_albers = gdf_usa.to_crs("EPSG:5070")
    # Unir (disolver) todos los estados en un solo polígono
    gdf_usa_albers_dissolved = gdf_usa_albers.dissolve()
    # Recortar Voronoi con contorno
    gdf_voronoi_clipped_albers = gpd.clip(gdf_voronoi_albers, gdf_usa_albers_dissolved)

    # -------------------------------------------------------------------
    # 9. Regresar Voronoi a EPSG:4326 para Plotly
    # -------------------------------------------------------------------
    gdf_voronoi_clipped = gdf_voronoi_clipped_albers.to_crs("EPSG:4326")

    # -------------------------------------------------------------------
    # 10. Dibujar polígonos Voronoi en el mapa (capa adicional)
    # -------------------------------------------------------------------
    # Usamos un choropleth “temporal” para extraer las trazas
    fig_voronoi = px.choropleth(
        gdf_voronoi_clipped,
        geojson=gdf_voronoi_clipped.geometry,
        locations=gdf_voronoi_clipped.index,
        color_discrete_sequence=["rgba(255,127,14,0.3)"],  # RGBA con 0.3 de transparencia
        labels={"geometry": "Voronoi"},
    )

    # Añadir las trazas del Voronoi a la figura principal
    for trace in fig_voronoi.data:
        fig.add_trace(trace)

    # -------------------------------------------------------------------
    # 11. Añadir los puntos de las estaciones en scatter
    # -------------------------------------------------------------------
    fig.add_trace(go.Scattergeo(
        lon=df_stations_filtered["longitude"],
        lat=df_stations_filtered["latitude"],
        text=df_stations_filtered.get("name", None),  # usa "name" si existe
        mode="markers",
        marker=dict(size=5, color="red", symbol="circle"),
        name="Estaciones"
    ))

    # -------------------------------------------------------------------
    # 12. Ajustar la vista y layout
    # -------------------------------------------------------------------
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="albers usa"
    )
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title_font_size=20
    )

    # Mostrar figura final
    fig.show()