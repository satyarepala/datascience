import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Load the Wine dataset
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target

# Data analysis and insights
def wine_data_analysis():
    # Basic information about the dataset
    print("Dataset Info:")
    print(wine_df.info())

    # Summary statistics of numerical columns
    print("\nSummary Statistics:")
    print(wine_df.describe())

    # Count of each target class
    print("\nTarget Class Count:")
    print(wine_df['target'].value_counts())

    # Visualization: Pair plot colored by target class
    sns.pairplot(wine_df, hue='target')
    plt.title("Pair Plot")
    plt.show()

    # Visualization: Box plot of alcohol content by target class
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='target', y='alcohol', data=wine_df)
    plt.title("Box Plot of Alcohol Content by Target Class")
    plt.show()

    # Visualization: Violin plot of flavanoids by target class
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='target', y='flavanoids', data=wine_df)
    plt.title("Violin Plot of Flavanoids by Target Class")
    plt.show()

    # Visualization: Scatter plot of color_intensity vs. hue
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='color_intensity', y='hue', hue='target', data=wine_df)
    plt.title("Scatter Plot of Color Intensity vs. Hue")
    plt.show()

    # Bar plot: Count of each target class
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=wine_df)
    plt.title("Count of Each Target Class")
    plt.show()

if __name__ == "__main__":
    wine_data_analysis()
