import pandas as pd

def calculate_statisticals_months(df):
    # Asegurarse de que la columna de fecha sea datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Crear columnas de año y mes
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Agrupar por estación, año y mes, luego calcular estadísticas
    resumen = df.groupby(['stationTriplet', 'year', 'month'])['value'].agg(
        max_temp='max',
        min_temp='min',
        avg_temp='mean'
    ).reset_index()
    
    return resumen
