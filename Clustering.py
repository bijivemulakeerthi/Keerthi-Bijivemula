import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.optimize import curve_fit
from scipy import stats


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





# Define the exponential growth model
def exp_model(x, a, b, c):
    return a * np.exp(b * x) + c

# Define the function to estimate the confidence range
def err_ranges(x, y, pcov, conf_level=0.95):
    perr = np.sqrt(np.diag(pcov))
    t_value = abs(stats.t.ppf((1 - conf_level) / 2, len(x) - 2))
    lower = y - t_value * perr[1] * np.sqrt(1 + 1/len(x) + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    upper = y + t_value * perr[1] * np.sqrt(1 + 1/len(x) + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    return lower, upper

# Set the name of the country to analyze
country = 'South Africa'

# Extract the data for the country from the data DataFrame
y = data.loc[data['Country Name'] == country].iloc[:, 1:].values[0]

# Create xdata for the years
x = np.arange(2001, 2023)

# Fit the exponential growth model to the data
popt, pcov = curve_fit(exp_model, x, y)

# Define the future time points to predict
future_x = np.arange(2021, 2033)

# Predict the future values using the fitted model
future_y = exp_model(future_x, *popt)

# Estimate the lower and upper limits of the confidence range
lower, upper = err_ranges(future_x, future_y, pcov, conf_level=0.95)

# Plot the results
plt.plot(x, y, 'bo', label='Data')
plt.plot(future_x, future_y, 'r-', label='Fitted Model')
plt.fill_between(future_x, lower, upper, alpha=0.1, label='Confidence Range')
plt.title('Exponential Growth Model Fitting for ' + country)
plt.xlabel('Year')
plt.ylabel('Access to electricity')
plt.legend()
plt.show()

# Save the plot as a PNG file
plt.savefig('ExponentialGrowth.png')






