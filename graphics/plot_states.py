import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
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

