#%%
# Initializing
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt  # Added import statement
from pmdarima import auto_arima
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error

file_P = os.path.join(os.getcwd(), 'Elspotprices2.csv')
df_prices = pd.read_csv(file_P)
#%%
# Data Preprocessing
df_prices["HourUTC"] = pd.to_datetime(df_prices["HourUTC"])
df_prices = df_prices.loc[(df_prices['PriceArea'] == "DK2")][["HourUTC", "SpotPriceDKK"]]
df_prices = df_prices.loc[df_prices["HourUTC"].dt.year.isin([2019, 2020, 2021, 2022, 2023])]
df_prices = df_prices.set_index('HourUTC')
df_prices = df_prices.dropna().asfreq('H')
df_prices = df_prices.reset_index()

df_prices['SpotPriceDKK'].plot(title='Electricity Spot Price')
plt.show()

#%%
# Split the data
train, test = model_selection.train_test_split(df_prices, train_size=200)
#%%
# Check for stationarity



#result = adfuller(df_prices['SpotPriceDKK'])
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])

#seasonal = (seasonal_decompose(df_prices['SpotPriceDKK'], model='additive', period=24)).seasonal
#seasonal.plot()
#plt.ylim(-5, 5)
#plt.show()

model_auto_arima = auto_arima(train['SpotPriceDKK'], 
                              start_p=1, start_q=1, 
                              max_p=3, max_q=3, 
                              seasonal=True, m=24, 
                              trace=False, 
                              suppress_warnings=True, 
                              stepwise=True, 
                              error_action="ignore")
print(model_auto_arima.summary())

#%%
# Fit the model
model = ARIMA(train['SpotPriceDKK'], order=model_auto_arima.order)
model_fit = model.fit()

#%%
# Forecast
forecast = model_fit.forecast(steps=24) 
actual = df_prices['SpotPriceDKK'].iloc[-24:]  
rmse = np.sqrt(mean_squared_error(actual, forecast))
print('RMSE: ', rmse)
#%%
# plot something

plt.figure(figsize=(12, 6))
plt.plot(actual.index, actual, color='blue', label='Actual')
plt.plot(actual.index, forecast, color='red', linestyle='--', label='Forecast')
plt.title('ARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.legend()
plt.show()

# %%
