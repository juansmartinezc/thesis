# Volver a cargar el archivo
import pandas as pd
import matplotlib.pyplot as plt


def plot_years_histogram(df):
    # Verificar la estructura de la columna "year"
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Contar la cantidad de registros por año
    df_year_counts = df["year"].value_counts().sort_index()

    # Generar un histograma de la distribución de datos por año
    plt.figure(figsize=(8, 5))
    plt.bar(df_year_counts.index, df_year_counts.values, color='blue', alpha=0.7)
    plt.xlabel("Año")
    plt.ylabel("Cantidad de registros")
    plt.title("Distribución de datos por año")
    plt.xticks(df_year_counts.index, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Guardar la imagen y mostrarla
    plt.savefig("distribucion_datos_por_anio.png")
    plt.show()

def plot_crops_states(df):
    # Contar la cantidad de registros por estado
    df_state_counts = df["state_name"].value_counts().sort_values(ascending=False)

    # Generar una gráfica de barras con la cantidad de datos por estado
    plt.figure(figsize=(12, 6))
    plt.bar(df_state_counts.index, df_state_counts.values, color='green', alpha=0.7)
    plt.xlabel("Estado")
    plt.ylabel("Cantidad de registros")
    plt.title("Distribución de datos por estado")
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Guardar la imagen y mostrarla
    plt.savefig("distribucion_datos_por_estado.png")
    plt.show()