import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("airline_sales_data.csv")

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values(by='date')

# Select features and target
features = ['ticket_price', 'season', 'fuel_price', 'economic_index']
target = 'tickets_sold'

# Normalize features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Prepare data for LSTM model
X = df[features].values
y = df[target].values

# Reshape input for LSTM (samples, time steps, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, len(features))),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Plot actual vs predicted ticket sales
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Ticket Sales', color='blue')
plt.plot(predictions, label='Predicted Ticket Sales', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Tickets Sold')
plt.title('Actual vs Predicted Ticket Sales')
plt.show()
