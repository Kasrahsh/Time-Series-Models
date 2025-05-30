

import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sn

# -- Settings and Options --
warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress future warnings for cleaner output

# -- Data Loading and Preparation --
file_path = r"C:\Users\Asus\Desktop\DS\Time Series\04 - Statistical Models for Time Series Forecasting\002 Temp-Data.csv"

df = pd.read_csv(file_path, index_col="DATE", parse_dates=True)
df.index.freq = "D"   # Set the index frequency as daily
df.dropna(inplace=True)    # Remove missing values

# -- Correlation Heatmap --
corr = df.corr()

# Define a custom colormap for better visualization
cmap = sn.diverging_palette(10, 120, as_cmap=True)

plt.figure(figsize=(10, 8))
sn.heatmap(corr, annot=True, cmap=cmap, fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap of Temperature Data")
plt.tight_layout()                           
plt.savefig("correlation_heatmap.png")
plt.show()
print("Heatmap saved as: correlation_heatmap.png")

# -- Data Splitting (Train/Test) --
target_col = "Temp"                          
n_train = 510

train = df.iloc[:n_train][target_col]
test = df.iloc[n_train:][target_col]

exo = df.iloc[:, 1:4]                        
exo_train = exo.iloc[:n_train]
exo_test = exo.iloc[n_train:]

# -- Time Series Decomposition --
from statsmodels.tsa.seasonal import seasonal_decompose

decompose_result = seasonal_decompose(df[target_col])

# Plot full decomposition
decompose_result.plot()
plt.suptitle("Time Series Decomposition", y=1.02)
plt.tight_layout()
plt.show()

# Optional: Plot only the seasonal component
decompose_result.seasonal.plot(title="Seasonal Component")
plt.tight_layout()
plt.show()

# -- Model Selection with Auto ARIMA --
from pmdarima import auto_arima

arima_model = auto_arima(
    df[target_col],
    exogenous=exo,
    m=7,
    trace=True,
    D=1,
    suppress_warnings=True,
)
print(arima_model.summary())


from statsmodels.tsa.statespace.sarimax import SARIMAX
Model = SARIMAX(train, exog=exo_train, order =(2,0,2),seasonal_order =(0,1,1,7) )
Model = Model.fit(maxiter=1000,method='powell')
prediction = Model.predict(len(train), len(train)+len(test)-1, exog= exo_test, typ="levels")

# --- Plotting Results ---
plt.figure(figsize=(12, 6))  # Optional: Add figure size for better visibility
plt.plot(test, color="red", label="Actual Values") # Corrected label
plt.plot(prediction, color="blue", label="Forecasted Values") # Corrected typo and color
plt.xlabel("Day")  # Capitalized 'Day' for better labeling
plt.ylabel("Temperature")  # Capitalized 'Temperature' for better labeling
plt.title("ARIMA Forecast vs Actual Values")  # Optional: Add a title
plt.legend()
plt.grid(True)  # Optional: Add a grid for better readability
plt.show()

# --- Model Performance Evaluation ---
# Model performance metrics for comparison
print("\n--- SARIMAX Performance ---")  # Added a header for clarity
print(f"Mean of actual test values: {test.mean():.2f}")  # Formatted output
print(f"Mean of forecasted values: {prediction.mean():.2f}")  # Formatted output

import math
from sklearn.metrics import mean_squared_error as mse

# Calculate Root Mean Squared Error (RMSE)
rmse = math.sqrt(mse(test, prediction))  
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")