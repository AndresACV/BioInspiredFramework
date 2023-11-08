# load_data.py

from pathlib import Path
import pandas as pd
from scipy.io import arff
import traceback
from tabulate import tabulate

def load_arff(file_path):
    """Carga un archivo .arff y lo convierte en un DataFrame de pandas."""
    if file_path.stat().st_size == 0:
        print(f"El archivo {file_path} está vacío. Se omite.")
        return None
    try:
        data = arff.loadarff(file_path)
        return pd.DataFrame(data[0])
    except Exception as e:
        print(f"Error al cargar el archivo {file_path}: {e}")
        traceback.print_exc()
        return None

def load_datasets(root_path):
    """Carga todos los archivos .arff en una carpeta y sus subcarpetas."""
    datasets = {}
    for file_path in Path(root_path).rglob('*.arff'):
        if not file_path.name.startswith('.'):
            df = load_arff(file_path)
            if df is not None:
                datasets[file_path.name] = df
    return datasets

def load_all_datasets(root_path):
    # Carga todos los datasets y los almacena en un diccionario
    all_datasets = load_datasets(root_path)
    return all_datasets

def print_dataset_summary(datasets):
    """Imprime un resumen de los datasets cargados."""
    summary = [(name, df.shape[0], df.shape[1]) for name, df in datasets.items()]
    print(tabulate(summary, headers=['Dataset', 'Rows', 'Columns']))

# Si este script se ejecuta directamente, entonces carga los datasets
if __name__ == "__main__":
    
    # Obtener la ruta al directorio actual donde se encuentra este script
    current_dir = Path(__file__).parent
    
    # Construir la ruta al directorio de datos, subiendo un nivel y luego bajando a 'data/DS-DefectPrediction'
    data_dir = current_dir.parent.parent / 'data' / 'DS-DefectPrediction'

    # Cargar todos los datasets y almacenarlos en un diccionario
    all_datasets = load_all_datasets(data_dir)
    
    # Imprimir un resumen de los datasets cargados
    print_dataset_summary(all_datasets)
