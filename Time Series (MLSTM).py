import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

warnings.simplefilter(action='ignore', category=FutureWarning)

# ---- Data Loading ----
file_path = r"C:\Users\Asus\Desktop\DS\Time Series\05 - Deep Learning for Time Series Forecasting\Electricity-Consumption.csv"
df = pd.read_csv(file_path, index_col="DATE", parse_dates=True)
df.dropna(inplace=True)

# ---- Visualization & Correlation ----
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    annot_kws={"size": 12},
    cbar_kws={"shrink": 0.8},
    linewidths=0.5,
    square=True
)
plt.xticks(fontsize=12, rotation=45, ha="right")
plt.yticks(fontsize=12, rotation=0)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()

# ---- Data Preparation ----
WINDOW_SIZE = 24
FEATURES = 3   # adjust if you want to use more/less features

train_set = df.iloc[:8712, :].values
test_set = df.iloc[8712:, :].values

scaler = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = scaler.fit_transform(train_set)
test_set_scaled  = scaler.transform(test_set)
test_set_scaled =test_set_scaled[:,0:2]
# X, y window creation for training
x_train, y_train = [], []
for i in range(WINDOW_SIZE, len(train_set_scaled)):
    x_train.append(train_set_scaled[i - WINDOW_SIZE:i, :FEATURES])
    y_train.append(train_set_scaled[i, 2])  # target column (index 2)

x_train = np.array(x_train)
y_train = np.array(y_train)

# ---- Model ----
model = Sequential([
    LSTM(70, return_sequences=True, input_shape=(x_train.shape[1], FEATURES)),
    Dropout(0.2),
    LSTM(70, return_sequences=True),
    Dropout(0.2),
    LSTM(70, return_sequences=True),
    Dropout(0.2),
    LSTM(70),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="AdamW", loss="mean_squared_error")
history = model.fit(x_train, y_train, epochs=40, batch_size=32)

# ---- Training Loss Plot ----
plt.plot(history.history["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()
plt.show()

# ---- Save & Load Model Example ----
model_path = r"c:\users\asus\desktop\python scripts\MLSTM.keras"
model.save(model_path)        
model = load_model(model_path)


prediction_test = []
batch_one = train_set_scaled[-WINDOW_SIZE:]
batch_new = batch_one.reshape((1, WINDOW_SIZE, 3))
for i in range(48):
    pred = model.predict(batch_new, verbose=0)[0][0]           
    prediction_test.append(pred)
    New_Var = test_set_scaled[i,:]
    New_Var = New_Var.reshape(1,2)
    New_test = np.insert(New_Var,2,[pred],axis=1)
    New_test = New_test.reshape(1,1,3)
    batch_new = np.append(batch_new[:,1:,:],New_test,axis=1)
    
prediction_test = np.array(prediction_test).reshape(-1, 1)
SI = MinMaxScaler(feature_range = (0,1))
y_scale = train_set[:,2:3]
SI.fit_transform(y_scale)
prediction = SI.inverse_transform(prediction_test)
real_values = test_set[:,2]
plt.plot(real_values,color="red", label ="actual values")
plt.plot(prediction,color="blue", label ="forecast values")
plt.title("electrical consumption prediction") 
plt.xlabel("time(hr)")
plt.ylabel("electricity demand")
plt.legend()
plt.show()


import math 
from sklearn.metrics import mean_squared_error

RSME = math.sqrt(mean_squared_error(real_values, prediction))

from sklearn.metrics import r2_score 
Rsquare =r2_score(real_values,prediction)


def mean_absolute_percentage(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage(real_values, prediction)
