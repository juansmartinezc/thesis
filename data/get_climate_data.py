from data.get_scan_data import get_usda_stations, filter_scan_data, get_usda_weather_data

stations_directory = 'results'
def get_stations_data():
    stations_data = get_usda_stations(networks="SNTL", bbox="-120,35,-110,40")
    stations_df = filter_scan_data(stations_data)
    stations_df.to_csv(f"{stations_directory}/stations.csv")
    return stations_df
    '''
    station_1 = stations_df.loc[0,:]
    print(station_1)
    stationTriplet = station_1['stationTriplet']
    elements="TMAX,TMIN,PRCP"
    beginDate = station_1['beginDate']
    endDate = station_1['endDate']
    weather_df = get_usda_weather_data(stationTriplet, elements, beginDate, endDate)
    print(weather_df)
    # Ejemplo de uso:
    '''