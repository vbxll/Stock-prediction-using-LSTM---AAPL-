# Install necessary libraries
!pip install tensorflow keras yfinance mplfinance -qqq

# Import necessary libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mplfinance as mpf

# Check TensorFlow version
print("TensorFlow Version: ", tf.__version__)

# Fetch AAPL data
aapl_data = yf.download('AAPL', start='2020-01-01', end='2024-09-15')

# Display the first few rows of the dataframe
aapl_data.head()

# Plot the 'Low' price
aapl_data['Low'].plot(kind='line', figsize=(8, 4), title='Low')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Checking for missing values
aapl_data.isnull().sum()

# Filling missing values, if any
aapl_data.fillna(method='ffill', inplace=True)

# Scaling the data (focusing on the 'Close' price)
scaler = MinMaxScaler(feature_range=(0,1))
aapl_data_scaled = scaler.fit_transform(aapl_data['Close'].values.reshape(-1,1))

# Creating the dataset
X = []
y = []
for i in range(60, len(aapl_data_scaled)):
    X.append(aapl_data_scaled[i-60:i, 0])
    y.append(aapl_data_scaled[i, 0])


# Split the data into train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model with attention mechanism
model = Sequential()

# LSTM layers with return_sequences=True
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))

# Adding Dropout and Batch Normalization
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary
model.summary()

# Early Stopping and Model Checkpoint callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
tensorboard = TensorBoard(log_dir='./logs')
csv_logger = CSVLogger('training_log.csv')

callbacks_list = [early_stopping, model_checkpoint, reduce_lr, tensorboard, csv_logger]

# Build the corrected LSTM model
model = Sequential()

# First LSTM layer with return_sequences=True
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# Second LSTM layer without return_sequences=True (since we want the final output)
model.add(LSTM(units=50, return_sequences=False))

# Adding Dropout and Batch Normalization for regularization
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Output layer - Dense layer with 1 unit (predicting one value)
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary
model.summary()

# Train the model (using callbacks for early stopping and other features)
history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=callbacks_list)

# Prepare test data
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Evaluate the model on test data
test_loss = model.evaluate(X_test, y_test)
print("Test Loss: ", test_loss)

# Making predictions on test data
y_pred = model.predict(X_test)

# Calculating MAE and RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fetch the latest 60 days (using a valid period, e.g., 3 months of data)
data = yf.download('AAPL', period='3mo', interval='1d')  # 3 months of daily data

# Check if data is empty
if data.empty:
    raise ValueError("No data was fetched. Please check the ticker or period.")

# Select the 'Close' price and scale it
closing_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Prepare the latest data for prediction
X_latest = np.array([scaled_data[-60:].reshape(60)])
X_latest = np.reshape(X_latest, (X_latest.shape[0], X_latest.shape[1], 1))

# Make predictions for the next 4 days
predicted_stock_price = model.predict(X_latest)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

print("Predicted Stock Prices for the next 4 days: ", predicted_stock_price)

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fetch the latest 3 months of AAPL stock data
data = yf.download('AAPL', period='3mo', interval='1d')

# Select 'Close' price and scale it
closing_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Prepare the latest data for prediction (most recent 60 days)
X_latest = np.array([scaled_data[-60:].reshape(60)])
X_latest = np.reshape(X_latest, (X_latest.shape[0], X_latest.shape[1], 1))

# Predict the next 4 days iteratively
predicted_prices = []
current_batch = X_latest  # Initialize with the latest batch of data

for i in range(4):  # Predicting 4 future days
    next_prediction = model.predict(current_batch)
    
    # Append the predicted price to the results (inverse transform back to original scale)
    predicted_price = scaler.inverse_transform(next_prediction)[0, 0]
    predicted_prices.append(predicted_price)

    # Update the batch to include the predicted value (shift window by 1 day)
    next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
    current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)

# Print the predictions formatted to 2 decimal places
for i, price in enumerate(predicted_prices, 1):
    print(f"Day {i} prediction: ${price:.2f}")


# Ensure predicted_stock_price is a flattened array
predicted_stock_price = np.array(predicted_prices)  # This is now (4,) from the earlier loop

# Create a list of dates for the predictions
last_date = data.index[-1]
next_day = last_date + pd.Timedelta(days=1)
prediction_dates = pd.date_range(start=next_day, periods=4)

# Plot the actual data
plt.figure(figsize=(10,6))
plt.plot(data.index[-60:], data['Close'][-60:], linestyle='-', marker='o', color='blue', label='Actual Data')

# Plot the predicted data
plt.plot(prediction_dates, predicted_stock_price, linestyle='-', marker='o', color='red', label='Predicted Data')

plt.title("AAPL Stock Price: Last 60 Days and Next 4 Days Predicted")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming predicted_prices is your list of predicted stock prices for the next 4 days

# Create a list of dates for the predictions
last_date = data.index[-1]  # Last available date in the dataset
next_day = last_date + pd.Timedelta(days=1)
prediction_dates = pd.date_range(start=next_day, periods=4)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot actual data for the last 7 days with a thicker, dashed line and large blue markers
plt.plot(data.index[-7:], data['Close'][-7:], linestyle='--', marker='o', markersize=10, linewidth=2, color='#1f77b4', label='Actual Data (Last Week)', alpha=0.8)

# Plot predicted data for the next 4 days with a thicker, solid line and large red markers
plt.plot(prediction_dates, predicted_prices, linestyle='-', marker='D', markersize=10, linewidth=2, color='#ff6347', label='Predicted Data (Next 4 Days)', alpha=0.9)

# Add labels to each point on the plot with a vibrant background and larger font size
for i, (date, price) in enumerate(zip(prediction_dates, predicted_prices)):
    plt.text(date, price, f'{price:.2f}', ha='center', va='bottom', fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.7))

for i, (date, price) in enumerate(zip(data.index[-7:], data['Close'][-7:])):
    plt.text(date, price, f'{price:.2f}', ha='center', va='bottom', fontsize=12, color='white', bbox=dict(facecolor='blue', alpha=0.7))

# Add vibrant gridlines
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# Customize the title and axis labels with a vibrant font style
plt.title("AAPL Stock Price: Last 7 Days and Next 4 Days Predicted", fontsize=18, fontweight='bold', color='#333333')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.ylabel('Price (USD)', fontsize=14, fontweight='bold')

# Customize the legend
plt.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', edgecolor='gray')

# Show the plot with tight layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()
