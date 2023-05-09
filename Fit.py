import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

data = pd.read_csv('/Users/apple/Downloads/Electricity.csv')

# Drop the 'Unnamed: 66' column
data = data.drop(['Unnamed: 66', '1960', '2021'], axis=1)

data = data.loc[:, ['Country Name'] + list(data.columns[42:])]

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

# Save the plot as a PNG file
plt.savefig('ExponentialGrowth.png')
