# clean_data.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from load_data import DataLoader

class DataCleaner:
    def __init__(self, datasets):
        self.datasets = datasets

    @staticmethod
    def clean_dataframe(df):
        """Cleans a pandas DataFrame by handling missing values and applying standardization"""
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        # Fill missing values in numeric columns with the mean if more than 50% data is present
        for col in numeric_cols:
            if df[col].isnull().sum() <= 0.5 * len(df):
                df[col] = df[col].fillna(df[col].mean())

        # Standardize numeric data
        if numeric_cols.any():
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Convert bytes to string in object columns
        object_cols = df.select_dtypes(include=['object']).columns
        df[object_cols] = df[object_cols].applymap(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
        )

        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        df = pd.get_dummies(df, columns=categorical_cols)

        return df
    
    def visualize_and_save_plots(self, df, dataset_name):
        """Visualizes data and saves plots."""
        print(f"Visualizing and saving plots for {dataset_name}...")

        # Main directory for saving plots
        script_dir = Path(__file__).parent
        main_plots_dir = script_dir.parent.parent / 'data' / 'cleaned_plots'
        main_plots_dir.mkdir(parents=True, exist_ok=True)

        # Create a subdirectory for each dataset
        dataset_plots_dir = main_plots_dir / dataset_name
        dataset_plots_dir.mkdir(parents=True, exist_ok=True)

        # Numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        # Histograms and Box Plots for Numeric Columns
        for col in numeric_cols:
            print(f"Creating histogram and box plot for {col}...")
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            df[col].hist()
            plt.title(f"Histogram of {col}")

            plt.subplot(1, 2, 2)
            sns.boxplot(y=df[col])
            plt.title(f"Box Plot of {col}")

            combined_file = dataset_plots_dir / f"{dataset_name}_{col}_combined.png"
            plt.savefig(combined_file)
            plt.close()
            print(f"Saved histogram and box plot for {col}.")

        # Correlation Heatmap
        if len(numeric_cols) > 1:  # Only if there are more than one numeric columns
            print("Creating correlation heatmap...")
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title("Correlation Heatmap")
            corr_heatmap_file = dataset_plots_dir / f"{dataset_name}_correlation_heatmap.png"
            plt.savefig(corr_heatmap_file)
            plt.close()
            print("Saved correlation heatmap.")

        # Bar Charts for Categorical Columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            print(f"Creating bar chart for {col}...")
            plt.figure()
            df[col].value_counts().plot(kind='bar')
            plt.title(f"Bar Chart of {col}")
            bar_chart_file = dataset_plots_dir / f"{dataset_name}_{col}_bar_chart.png"
            plt.savefig(bar_chart_file)
            plt.close()
            print(f"Saved bar chart for {col}.")
            
        print(f"Completed visualizing and saving plots for {dataset_name}.")

    def clean_all_datasets(self):
        """Cleans all datasets and returns a dictionary of cleaned DataFrames."""
        cleaned_datasets = {}
        for name, df in self.datasets.items():
            cleaned_df = self.clean_dataframe(df)
            self.visualize_and_save_plots(cleaned_df, name)
            cleaned_datasets[name] = cleaned_df
        return cleaned_datasets

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent / 'data' / 'DS-DefectPrediction'
    
    data_loader = DataLoader(data_dir)
    all_datasets = data_loader.load_datasets()
    
    cleaner = DataCleaner(all_datasets)
    cleaned_datasets = cleaner.clean_all_datasets()

    base_path = current_dir.parent.parent / 'data' / 'cleaned'
    base_path.mkdir(parents=True, exist_ok=True)

    for name, df in cleaned_datasets.items():
        file_path = base_path / f"{name}_cleaned.csv"
        df.to_csv(file_path, index=False)
        print(f"File saved: {file_path}")

