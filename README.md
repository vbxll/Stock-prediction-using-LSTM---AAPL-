
# Time Series Analysis using LSTM on Apple Stock Data

## Overview
This project aims to perform time series analysis and prediction on Apple stock data using Long Short-Term Memory (LSTM) neural networks. The data is fetched using the `yfinance` library, and the model is built using `TensorFlow` and `Keras`.

## Requirements
The following libraries are required to run the notebook:
- TensorFlow
- Keras
- yfinance
- mplfinance
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

You can install the necessary packages with the following command:
```bash
pip install tensorflow keras yfinance mplfinance
```

## Dataset
The stock price data for Apple (AAPL) is fetched from Yahoo Finance for the period between January 1, 2020, and September 15, 2024.

## Steps Involved
1. **Data Collection**: 
   - The data is collected using the `yfinance` API.
   ```python
   aapl_data = yf.download('AAPL', start='2020-01-01', end='2024-09-15')
   ```

2. **Data Preprocessing**: 
   - Scaling the data using `MinMaxScaler` to normalize the input features.
   
3. **Model Building**:
   - A Sequential model with LSTM layers is constructed using Keras. Additional layers like Batch Normalization, Dropout, and Dense are used to improve performance and prevent overfitting.

4. **Model Training**:
   - The model is trained using the training data with callbacks like `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` to monitor and improve training efficiency.

5. **Prediction and Visualization**:
   - Predictions are made on the test data, and the results are visualized using Matplotlib and mplfinance.

## Running the Notebook
To run the notebook:
1. Install the necessary dependencies.
2. Execute the notebook cells in sequence. The stock data will be fetched, processed, and the LSTM model will be trained and evaluated.

## Results
The trained model predicts future stock prices for Apple based on historical data. The performance of the model is visualized by plotting both the actual and predicted stock prices.


## License
This project is open-source and available under the MIT License.

---
