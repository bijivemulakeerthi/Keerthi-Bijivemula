import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

data = pd.read_csv('/Users/apple/Downloads/Electricity.csv')

# Drop the 'Unnamed: 66' column
data = data.drop(['Unnamed: 66', '1960', '2021'], axis=1)

data = data.loc[:, ['Country Name'] + list(data.columns[42:])]

# Extract the columns we want to cluster on
X = data.iloc[:, 1:]

# Normalize the data
X_norm = (X - X.mean()) / X.std()

# Create imputer and fill in missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Perform k-means clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(X_imputed)


# Add the cluster labels to the original dataframe
data['cluster'] = kmeans.labels_

# Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(data.iloc[:, -2], data.iloc[:, -3], c=data['cluster'])
plt.scatter(kmeans.cluster_centers_[:, -2], kmeans.cluster_centers_[:, -3], s=300, c='red')
plt.title('KMeans Clustering Results')
plt.xlabel('2020')
plt.ylabel('2019')


# Save the plot as a PNG file
plt.savefig('ClusteringResults.png')

