
import pandas as pd
import matplotlib.pyplot as plt
import warnings  

# --- Data Loading and Preparation ---
file_path = r"C:\Users\Asus\Desktop\DS\Time Series\04 - Statistical Models for Time Series Forecasting\002 Temp-Data.csv"
df = pd.read_csv(file_path, index_col="DATE", parse_dates=True)

# Set the frequency of the index
df.index.freq = "D"

# Drop rows with missing values
df.dropna(inplace=True)

# Select the 'Temp' column and convert back to a DataFrame
df = pd.DataFrame(df["Temp"])

# --- Data Splitting ---
train = df.iloc[:510, 0]
test = df.iloc[510:, 0]

# --- Time Series Decomposition ---
# Visualization of time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose as sa

decompose_result = sa(df)
decompose_result.plot()
decompose_result.seasonal.plot()  # Plot the seasonal component separately

# --- Autocorrelation and Partial Autocorrelation Analysis ---
# Finding potential parameters (p, d, q) using ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train, lags=50)
plot_pacf(train, lags=50)

# --- Auto ARIMA for Parameter Selection ---
# Suppressing future warnings for better readability
warnings.simplefilter(action='ignore', category=FutureWarning)

# Finding optimal parameters (p,d,q) using auto_arima with grid search
from pmdarima import auto_arima

# Note: Running auto_arima on the full dataframe 'df'
auto_arima(df, trace=True)

# --- ARIMA Model Development and Forecasting ---
# Developing the ARIMA model using the training data
from statsmodels.tsa.arima.model import ARIMA

A_model = ARIMA(train, order=(1, 1, 2))
predictor = A_model.fit()

# Display the model summary
predictor.summary()

# Predict values on the test set
predicted_results = predictor.predict(start=len(train), end=len(train) + len(test) - 1, typ="levels")

# --- Plotting Results ---
plt.figure(figsize=(12, 6))  # Optional: Add figure size for better visibility
plt.plot(test, color="red", label="Actual Values") # Corrected label
plt.plot(predicted_results, color="blue", label="Forecasted Values") # Corrected typo and color
plt.xlabel("Day")  # Capitalized 'Day' for better labeling
plt.ylabel("Temperature")  # Capitalized 'Temperature' for better labeling
plt.title("ARIMA Forecast vs Actual Values")  # Optional: Add a title
plt.legend()
plt.grid(True)  # Optional: Add a grid for better readability
plt.show()

# --- Model Performance Evaluation ---
# Model performance metrics for comparison
print("\n--- ARIMA Performance ---")  # Added a header for clarity
print(f"Mean of actual test values: {test.mean():.2f}")  # Formatted output
print(f"Mean of forecasted values: {predicted_results.mean():.2f}")  # Formatted output

import math
from sklearn.metrics import mean_squared_error as mse

# Calculate Root Mean Squared Error (RMSE)
rmse = math.sqrt(mse(test, predicted_results))  
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")  

