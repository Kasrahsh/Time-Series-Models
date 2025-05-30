import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -- Settings and Options --
warnings.simplefilter(action='ignore', category=FutureWarning)

# -- Data Loading and Preparation --
file_path = r"C:\Users\Asus\Desktop\DS\Time Series\05 - Deep Learning for Time Series Forecasting\007 Solar-Data-Set.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
df.dropna(inplace=True)

train_set = df.iloc[:8712].values
test_set = df.iloc[8712:].values

sc = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = sc.fit_transform(train_set)
test_set_scaled = sc.transform(test_set)  # use transform, not fit_transform

# Prepare windowed data
ws = 24
x_train, y_train = [], []
for i in range(ws, len(train_set_scaled)):
    x_train.append(train_set_scaled[i - ws:i, 0])  # if univariate
    y_train.append(train_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape for LSTM: (samples, timesteps, features)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# -- Model Definition --
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=60))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1)



# Plot training loss
plt.plot(history.history["loss"])
plt.xlabel("Epoch number")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# Load 
from keras.models import load_model
model_path = r"c:\users\asus\desktop\python scripts\ULSTM.keras"
model.save(model_path)        
model = load_model(model_path)


prediction_test = []
batch_new = train_set_scaled[-ws:].reshape((1, ws, 1))
for _ in range(48):
    pred = model.predict(batch_new, verbose=0)[0][0]           
    prediction_test.append(pred)
    next_step = np.array([[[pred]]])                          
    batch_new = np.concatenate((batch_new[:, 1:, :], next_step), axis=1)  

prediction_test = np.array(prediction_test).reshape(-1, 1)
prediction = sc.inverse_transform(prediction_test)

plt.plot(prediction, color="red", label="Forecast")
plt.plot(test_set[:48], color="blue", label="Actual Values")
plt.xlabel("Time (h)")
plt.ylabel("Solar Irradiance")
plt.legend(loc="best")
plt.title("Forecast vs Actual")
plt.show()


import math 
from sklearn.metrics import mean_squared_error
RSME = math.sqrt(mean_squared_error(test_set,prediction))

from sklearn.metrics import r2_score 
Rsquare =r2_score(test_set,prediction)