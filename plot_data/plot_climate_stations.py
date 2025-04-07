
from data.get_climate_data import read_stations_data
from data.get_crop_yield_data import read_crop_yied_data
from eda.years_histogram import plot_years_histogram, plot_crops_states, filter_top_states
from graphics.plot_states import plot_states_with_filtered_stations, plot_selected_states, plot_states_with_filtered_stations_voronoi
# Cargar la hoja de datos

df_stations = read_stations_data()
df_crop_yield = read_crop_yied_data()
plot_years_histogram(df_crop_yield)
plot_crops_states(df_crop_yield)
df_top_status = filter_top_states(df_crop_yield)
plot_selected_states(df_top_status)
plot_states_with_filtered_stations(df_top_status, df_stations)
plot_states_with_filtered_stations_voronoi(df_top_status, df_stations)
