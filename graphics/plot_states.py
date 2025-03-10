import pandas as pd
import geopandas as gpd
import plotly.express as px

def plot_selected_states(dict_states):
    """
    Genera un mapa coroplético de EE.UU. destacando los estados con datos en dict_states.
    """

    # 1. Cargar el shapefile de los estados de EE.UU.
    gdf_usa = gpd.read_file("shape_files/state/cb_2018_us_state_20m.shp")  # Asegura la ruta correcta

    # 2. Normalizar nombres de los estados en el shapefile
    gdf_usa["NAME"] = gdf_usa["NAME"].str.upper()

    # 3. Filtrar solo los 48 estados contiguos de EE.UU. (excluir Alaska y Puerto Rico)
    exclude_states = {"ALASKA", "PUERTO RICO", "HAWAII"}
    gdf_usa = gdf_usa[~gdf_usa["NAME"].isin(exclude_states)]

    # 4. Unir los datos con el GeoDataFrame del mapa de EE.UU.
    gdf_usa["value"] = gdf_usa["NAME"].map(dict_states)

    # 5. Rellenar con 0 en estados sin datos
    gdf_usa["value"] = gdf_usa["value"].fillna(0)

    # 6. Crear el mapa interactivo con Plotly
    fig = px.choropleth(
        gdf_usa,
        geojson=gdf_usa.geometry,
        locations=gdf_usa.index,
        color="value",
        hover_name="NAME",
        color_continuous_scale="Blues",
        labels={"value": "Cantidad de registros"},
        title="Cantidad de registros por estado en EE.UU.",
        width=1000,  # Ajustar tamaño
        height=700  
    )

    # 7. Ajustar el enfoque del mapa
    fig.update_geos(
        fitbounds="locations",  # Enfocar solo en los estados contiguos
        visible=False,
        projection_type="albers usa"  # Mejor proyección para EE.UU.
    )

    # 8. Ajustar layout para que el título y la visualización sean óptimos
    fig.update_layout(
        margin={"r":0, "t":50, "l":0, "b":0},
        title_font_size=20
    )

    fig.show()
