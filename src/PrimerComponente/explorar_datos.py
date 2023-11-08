# explorar_datos.py

from load_data import load_all_datasets  # Asegúrate de que esta importación funcione correctamente
from pathlib import Path

def explorar_dataframe(df):
    """Explora un DataFrame de pandas y muestra información relevante."""
    # Mostrar las primeras filas del DataFrame
    print("Primeras filas del DataFrame:")
    print(df.head())

    # Obtener el número de filas y columnas
    print(f"\nEl DataFrame tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

    # Verificar la presencia de valores nulos
    null_data = df.isnull().sum()
    null_data = null_data[null_data > 0]
    if not null_data.empty:
        print("\nNúmero de valores nulos por columna:")
        print(null_data)
    else:
        print("\nNo hay valores nulos en el DataFrame.")

    # Obtener estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(df.describe())

def explorar_todos_los_datasets(datasets):
    """Explora todos los DataFrames en el diccionario proporcionado."""
    for name, df in datasets.items():
        print(f"Explorando el dataset: {name}")
        explorar_dataframe(df)
        print("\n" + "-"*50 + "\n")  # Separador entre datasets

if __name__ == "__main__":
    # Obtener la ruta al directorio actual donde se encuentra este script
    current_dir = Path(__file__).parent
    # Construir la ruta al directorio de datos, subiendo un nivel y luego bajando a 'data/DS-DefectPrediction'
    data_dir = current_dir.parent.parent / 'data' / 'DS-DefectPrediction'

    # Cargar todos los datasets y almacenarlos en un diccionario
    all_datasets = load_all_datasets(data_dir)  # Asegúrate de que la función load_all_datasets esté definida y funcione correctamente

    # Explorar cada DataFrame en el diccionario
    explorar_todos_los_datasets(all_datasets)
