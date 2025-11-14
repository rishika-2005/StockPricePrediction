import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('/content/tata motors (1).xlsx')

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Use only the 'close' column
df = df[['close']]

# Normalize the close prices
scaler = MinMaxScaler()
df['close'] = scaler.fit_transform(df[['close']])

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Set sequence length based on dataset size
seq_length = 5  # Adjusted for small dataset

# Create sequences
X, y = create_sequences(df['close'].values, seq_length)

# Reshape X for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=20, batch_size=1, verbose=1)

# Predict next value
last_sequence = df['close'].values[-seq_length:]
last_sequence = last_sequence.reshape((1, seq_length, 1))
predicted_scaled = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_scaled)

print("\n🔮 Predicted next day price:", predicted_price[0][0])

# Predict on the training set for plotting
predictions = model.predict(X)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))
predicted_prices = scaler.inverse_transform(predictions)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title('Tata Motors Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
