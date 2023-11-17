import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from load_data import DataLoader
from pathlib import Path

class DataExplorer:
    def __init__(self, datasets):
        self.datasets = datasets

    def explore_dataframe(self, df, dataset_name):
        """Explores a pandas DataFrame and displays relevant information."""
        print(f"Exploring dataset: {dataset_name}")

        print("First rows of the DataFrame:")
        print(df.head())

        print(f"\nThe DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")

        # Main directory for saving plots
        script_dir = Path(__file__).parent
        main_plots_dir = script_dir.parent.parent / 'data' / 'dataset_plots'
        main_plots_dir.mkdir(parents=True, exist_ok=True)

        # Create a subdirectory for each dataset
        dataset_plots_dir = main_plots_dir / dataset_name
        dataset_plots_dir.mkdir(parents=True, exist_ok=True)

        # Visualize null values
        print("Creating null value heatmap...")
        plt.figure()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title("Null Value Heatmap")
        heatmap_file = dataset_plots_dir / f"{dataset_name}_null_heatmap.png"
        plt.savefig(heatmap_file)
        plt.close()
        print("Null value heatmap saved.")

        # Descriptive statistics
        print("\nDescriptive statistics:")
        print(df.describe())

        # Numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        # Scatter Plot Matrix for Numeric Columns
        if len(numeric_cols) <= 5:
            print("Creating scatter plot matrix...")
            sns.pairplot(df[numeric_cols])
            plt.suptitle("Scatter Plot Matrix")
            scatter_matrix_file = dataset_plots_dir / f"{dataset_name}_scatter_matrix.png"
            plt.savefig(scatter_matrix_file)
            plt.close()
            print("Scatter plot matrix saved.")

        # PCA Visualization for High-Dimensional Data
        if len(numeric_cols) > 5:
            print("Creating PCA 2D projection...")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df[numeric_cols].dropna())
            pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
            plt.figure()
            sns.scatterplot(x="PCA1", y="PCA2", data=pca_df)
            plt.title("PCA - 2D Projection")
            pca_file = dataset_plots_dir / f"{dataset_name}_PCA_projection.png"
            plt.savefig(pca_file)
            plt.close()
            print("PCA 2D projection saved.")

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
            print(f"Histogram and box plot for {col} saved.")

        # Correlation Heatmap
        if len(numeric_cols) > 1:
            print("Creating correlation heatmap...")
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title("Correlation Heatmap")
            corr_heatmap_file = dataset_plots_dir / f"{dataset_name}_correlation_heatmap.png"
            plt.savefig(corr_heatmap_file)
            plt.close()
            print("Correlation heatmap saved.")

        # Bar Charts for Categorical Columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            print(f"Creating bar chart for {col}...")
            plt.figure()
            df[col].value_counts().plot(kind='bar')
            plt.title(f"Bar Chart of {col}")
            bar_chart_file = dataset_plots_dir / f"{dataset_name}_{col}_bar_chart.png"
            plt.savefig(bar_chart_file)
            plt.close()
            print(f"Bar chart for {col} saved.")

        print(f"Completed exploring dataset: {dataset_name}\n" + "-"*50 + "\n")

    def explore_all_datasets(self):
        """Explores all DataFrames in the provided dictionary."""
        for name, df in self.datasets.items():
            print(f"Exploring dataset: {name}")
            self.explore_dataframe(df, name)
            print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent / 'data' / 'DS-DefectPrediction'

    data_loader = DataLoader(data_dir)
    all_datasets = data_loader.load_datasets()

    data_explorer = DataExplorer(all_datasets)
    data_explorer.explore_all_datasets()



