import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# App Title
st.title("Stock Price Prediction with LSTM")

# Get user input for stock symbol
ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL for Apple):", "AAPL")

# Download stock data
if st.button("Predict"):
    st.write(f"Fetching stock data for {ticker}...")
    data = yf.download(ticker, start="2020-01-01", end="2024-09-15")
    
    # Display the raw data
    st.subheader('Raw Data')
    st.write(data.head())
    
    # Prepare the data for LSTM model
    st.write("Preparing data for LSTM model...")
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create the training dataset
    def create_dataset(dataset, time_step=60):
        x_data, y_data = [], []
        for i in range(time_step, len(dataset)):
            x_data.append(dataset[i-time_step:i, 0])
            y_data.append(dataset[i, 0])
        return np.array(x_data), np.array(y_data)
    
    time_step = 60  # Number of previous days to consider for prediction
    x_data, y_data = create_dataset(scaled_data)

    # Reshape input to be [samples, time steps, features]
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_data.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))  # Predicting the next price
    
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the model
    st.write("Training the LSTM model...")
    model.fit(x_data, y_data, epochs=5, batch_size=32, verbose=0)
    
    # Predicting future stock prices
    st.write("Making predictions...")
    predictions = model.predict(x_data)
    predictions = scaler.inverse_transform(predictions)  # Reverse scaling

    # Visualize the results
    st.subheader("Prediction vs Actual")
    
    # Fixing the SettingWithCopyWarning using .loc[] and proper slicing
    data['Prediction'] = np.nan
    data.loc[data.index[time_step:], 'Prediction'] = predictions.flatten()
    
    st.line_chart(data[['Close', 'Prediction']])

    # Display the predicted data
    st.subheader('Predicted Data')
    st.write(data[['Close', 'Prediction']].tail())
    
    # Forecast the next 7 days
    st.subheader("Next 7 Days Forecast")
    
    # Get the last 60 days from the dataset
    last_60_days = scaled_data[-time_step:]
    
    # Forecasting next 7 days
    forecast = []
    current_input = last_60_days.reshape(1, time_step, 1)
    
    for _ in range(7):
        next_price = model.predict(current_input)
        forecast.append(next_price[0, 0])
        # Update current input by adding the predicted price and removing the oldest one
        next_price = next_price.reshape(1, 1, 1)  # Reshape the next predicted price
        current_input = np.append(current_input[:, 1:, :], next_price, axis=1)
    
    # Inverse transform the forecasted values
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    
    # Create a dataframe to show the forecast
    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecast'])
    
    st.write(forecast_df)
    
    # Plot forecasted data
    st.subheader("Forecast for the Next 7 Days")
    st.line_chart(forecast_df)
