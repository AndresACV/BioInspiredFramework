# load_data.py

from pathlib import Path
import pandas as pd
from scipy.io import arff
import traceback
from tabulate import tabulate

class DataLoader:
    def __init__(self, root_path):
        self.root_path = root_path

    def load_arff_file(self, file_path):
        """Loads an .arff file and converts it into a pandas DataFrame."""
        if file_path.stat().st_size == 0:
            print(f"The file {file_path} is empty. Skipping.")
            return None
        try:
            data = arff.loadarff(file_path)
            return pd.DataFrame(data[0])
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            traceback.print_exc()
            return None

    def load_datasets(self):
        """Loads all .arff files in a folder and its subfolders, excluding those with 'isDefective' or 'Defective' columns."""
        datasets = {}
        for file_path in Path(self.root_path).rglob('*.arff'):
            if not file_path.name.startswith('.'):
                df = self.load_arff_file(file_path)
                if df is not None and 'isDefective' not in df.columns and 'Defective' not in df.columns:
                    datasets[file_path.name] = df
                else:
                    print(f"Excluding dataset '{file_path.name}' as it contains 'isDefective' or 'Defective' column.")
        return datasets

    @staticmethod
    def print_dataset_summary(datasets):
        """Prints a summary of the loaded datasets."""
        summary = [(name, df.shape[0], df.shape[1]) for name, df in datasets.items()]
        print(tabulate(summary, headers=['Dataset', 'Rows', 'Columns']))

# If this script is run directly, load the datasets
if __name__ == "__main__":
    # Get the path to the current directory where this script is located
    current_dir = Path(__file__).parent
    
    # Build the path to the data directory, going up one level and then down to 'data/DS-DefectPrediction'
    data_dir = current_dir.parent.parent / 'data' / 'DS-DefectPrediction'

    # Create an instance of DataLoader and load all datasets
    data_loader = DataLoader(data_dir)
    all_datasets = data_loader.load_datasets()
    
    # Print a summary of the loaded datasets
    DataLoader.print_dataset_summary(all_datasets)




