import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing # for Socioeconomic/geographic census data
import numpy as np

def main():
    # Load dataset 1
    data = fetch_california_housing(as_frame=True).frame
    # The function fetch_california_housing is specific to the scikit-learn library, which only includes a small set of "toy" and "real-world" datasets directly in its package. A specific dataset like Netflix Movies and TV Shows is not built into scikit-learn, so you must load it using general data processing functions.

    # Plotting
    # sns.histplot(data=data, x='MedInc', bins=30, kde=True)
    # plt.title('CH - Distribution of Median Income')
    # plt.xlabel('Income (Tens of Thousands)')
    # plt.ylabel('Frequency')
    # plt.show()

    # Plot all numerical features in a grid
    # data.hist(figsize=(12, 10), bins=30, edgecolor='black')
    # plt.title('CH - All numerical features')
    # plt.tight_layout()
    # plt.show()

    # Load dataset 2
    iris = sns.load_dataset('iris') # Physical measurements of flowers

    # Compare petal length across different species
    # sns.histplot(data=iris, x='petal_length', hue='species', multiple='stack', bins=20)
    # plt.title('IRIS - Petal Length Distribution by Species')
    # plt.xlabel('Petal Length')
    # plt.show()

    # Load dataset 3
    # Load the dataset from a local CSV file
    df = pd.read_csv('netflix_titles.csv')

    sns.histplot(data=df, x='release_year', bins=30, kde=True)
    plt.title('Netflix - Release Year Distribution')
    plt.xlabel('Release Year')
    plt.show()

    # Plot all numerical features in a grid
    sns.histplot(data=df, x='duration', bins=30, kde=True)
    plt.title('Netflix - Duration Distribution')
    plt.tight_layout()
    plt.show()

    # Simple frequency count of a single category
    # For features with many categories or long labels, swap x for y to draw the bars horizontally.
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x='type')
    plt.title('Total Count: Movies vs TV Shows')
    plt.show()

    # Analyze Age Ratings
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='rating', hue='type', order=df['rating'].value_counts().index)
    plt.title('Content Distribution by Rating')
    plt.xticks(rotation=45)
    plt.show()

    # Compare Type distribution by country
    sns.countplot(data=df, x='type', hue='country')
    plt.xlabel('Country')
    plt.show()

    # Convert duration to numeric (for Movies only)
    movies_df = df[df['type'] == 'Movie'].copy()
    movies_df['duration_min'] = movies_df['duration'].str.replace(' min', '').astype(float)

    # Plotting Box Plot to see outliers
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=movies_df['duration_min'], color='darkred')
    plt.title('Box Plot of Movie Durations (Identifying Outliers)')
    plt.xlabel('Duration (minutes)')
    plt.show()

    # 1. Feature Engineering: Convert 'duration' to numeric
    # Extract the number from '90 min' or '1 Season'
    df['duration_num'] = df['duration'].str.extract('(\\d+)').astype(float)

    # 2. Select numerical features
    # Common features: release_year, duration_num, and added IMDb scores if available
    netflix_numeric = df.select_dtypes(include=[np.number])

    # 3. Compute Correlation Matrix
    corr_matrix = netflix_numeric.corr()

    # 4. Plot Heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle for clarity

    sns.heatmap(corr_matrix,
                annot=True,  # Shows the correlation coefficients in each cell
                fmt=".2f",  # Rounds numbers to 2 decimal places
                cmap='coolwarm',  # Diverging palette (red for positive, blue for negative)
                center=0,  # Centers the colorbar at 0
                square=True,  # Makes cells square
                linewidths=0.5)  # Adds lines between cells for better readability

    plt.title('Correlation Heatmap: Netflix Content Features (2026)')
    plt.show()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', square=True)
    plt.title('Reduced Correlation Heatmap (Lower Triangle Only)')
    plt.show()

    pd.get_dummies(netflix_numeric).head()
if __name__ == '__main__':
    main()
