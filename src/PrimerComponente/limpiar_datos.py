# limpiar_datos.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from load_data import load_all_datasets

class DataCleaner:
    def __init__(self, datasets):
        self.datasets = datasets

    @staticmethod
    def clean_dataframe(df):
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        threshold = len(df) * 0.5
        df = df.dropna(thresh=threshold, axis=1)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        object_cols = df.select_dtypes(include=['object']).columns
        df[object_cols] = df[object_cols].applymap(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
        )

        if numeric_cols.any():
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        df = pd.get_dummies(df, columns=categorical_cols)

        return df

    def clean_all_datasets(self):
        return {name: self.clean_dataframe(df) for name, df in self.datasets.items()}

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent / 'data' / 'DS-DefectPrediction'
    all_datasets = load_all_datasets(data_dir)
    cleaner = DataCleaner(all_datasets)
    cleaned_datasets = cleaner.clean_all_datasets()

    base_path = current_dir.parent.parent / 'data' / 'cleaned'
    base_path.mkdir(parents=True, exist_ok=True)

    for name, df in cleaned_datasets.items():
        file_path = base_path / f"{name}_cleaned.csv"
        df.to_csv(file_path, index=False)
        print(f"Archivo guardado: {file_path}")