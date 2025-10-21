import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
import plotly.graph_objects as go

# Assuming df is your DataFrame with 'Close' prices and datetime index

# 1. Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# 2. Prepare the dataset for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60

# 3. Split data into train and test
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:]

# 4. Create datasets
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 5. Reshape input for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 6. Build Stacked Bidirectional LSTM Model
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50)))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# 7. Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# 8. Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)

# 9. Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 10. Inverse transform to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

# 11. Calculate RMSE and MAE
train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_predict))
train_mae = mean_absolute_error(y_train_orig, train_predict)
test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_predict))
test_mae = mean_absolute_error(y_test_orig, test_predict)

print(f'Train RMSE: {train_rmse}, Train MAE: {train_mae}')
print(f'Test RMSE: {test_rmse}, Test MAE: {test_mae}')

# 12. Prepare data for plotting
train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, 0] = train_predict[:, 0]

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, 0] = test_predict[:, 0]

# 13. Plot the results
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df.index, y=train_plot[:, 0], mode='lines', name='Train Predict', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df.index, y=test_plot[:, 0], mode='lines', name='Test Predict', line=dict(color='red')))

fig.update_layout(title='Stock Price Prediction using Stacked Bidirectional LSTM',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  template='plotly_dark')

fig.show()
