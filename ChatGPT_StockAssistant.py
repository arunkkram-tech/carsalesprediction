import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to predict next day's price range
def predict_next_day_range(data):
    
    """ data['Date'] = np.arange(len(data))  # Add numeric dates for linear regression
    X = data[['Date']]
    y_high = data['High']
    y_low = data['Low']

    # Train linear regression models for High and Low prices
    model_high = LinearRegression().fit(X, y_high)
    model_low = LinearRegression().fit(X, y_low)

    next_day = [[len(data)]]  # Numeric value for the next day
    predicted_high = model_high.predict(next_day)[0]
    predicted_low = model_low.predict(next_day)[0] """

    data = data.dropna()
    data['Date'] = np.arange(len(data))  # Add numeric dates for training
    X = data[['Date']]
    y_high = data['High']
    y_low = data['Low']

    # Train Random Forest Regressor for High and Low prices
    model_high = RandomForestRegressor(n_estimators=100, random_state=42)
    model_low = RandomForestRegressor(n_estimators=100, random_state=42)
    model_high.fit(X, y_high)
    model_low.fit(X, y_low)

    next_day = [[len(data)]]  # Numeric value for the next day
    predicted_high = model_high.predict(next_day)[0]
    predicted_low = model_low.predict(next_day)[0]



    return predicted_low, predicted_high

# Streamlit App
st.title("Stock Price Analysis App")

# Ticker Input
st.sidebar.header("Stock Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")

# Date Range Selection
today = datetime.today()
default_start = today - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date:", default_start)
end_date = st.sidebar.date_input("End Date:", today)

# Fetch Data
if st.sidebar.button("Fetch Data"):
    if ticker:
        # Get data from Yahoo Finance
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if not stock_data.empty:
            st.subheader(f"Stock Price Data for {ticker.upper()}")
            st.dataframe(stock_data.tail())

            # Candle Stick Chart for Last 30 Days
            st.subheader("Candlestick Chart with SMA (Last 30 Days)")
            stock_data["SMA50"] = stock_data["Close"].rolling(window=50).mean()
            stock_data["SMA100"] = stock_data["Close"].rolling(window=100).mean()

            last_30_days = stock_data[-30:]

            fig = go.Figure()
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=last_30_days.index,
                open=last_30_days['Open'],
                high=last_30_days['High'],
                low=last_30_days['Low'],
                close=last_30_days['Close'],
                name='Candlestick'
            ))
            # Add SMA50 and SMA100
            fig.add_trace(go.Scatter(
                x=last_30_days.index, y=last_30_days['SMA50'],
                mode='lines', name='SMA 50 Days',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=last_30_days.index, y=last_30_days['SMA100'],
                mode='lines', name='SMA 100 Days',
                line=dict(color='orange')
            ))
            fig.update_layout(
                title=f"Candlestick Chart with SMA for {ticker.upper()} (Last 30 Days)",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig)

            # Average and Standard Deviation of Returns (Last 90 Days)
            st.subheader("Statistics: Average and Standard Deviation of Returns (Last 90 Days)")
            stock_data['Daily Return'] = stock_data['Close'].pct_change()
            last_90_days = stock_data[-90:]
            avg_return = last_90_days['Daily Return'].mean()
            std_return = last_90_days['Daily Return'].std()

            stats = pd.DataFrame({
                'Statistic': ['Average Return', 'Standard Deviation of Return'],
                'Value': [avg_return, std_return]
            })
            st.table(stats)

            # RSI Indicator
            st.subheader("RSI Indicator (Last 30 Days)")
            stock_data['RSI'] = calculate_rsi(stock_data)
            last_30_days['RSI'] = stock_data['RSI'][-30:]

            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=last_30_days.index, y=last_30_days['RSI'],
                mode='lines', name='RSI',
                line=dict(color='purple')
            ))
            fig_rsi.update_layout(
                title=f"RSI Indicator for {ticker.upper()} (Last 30 Days)",
                xaxis_title="Date",
                yaxis_title="RSI",
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_rsi)

            # Next Day Price Range Prediction
            #st.subheader("Next Day Price Range Prediction")
            #predicted_low, predicted_high = predict_next_day_range(stock_data)
            #st.write(f"Predicted Low: {predicted_low:.2f}")
            #st.write(f"Predicted High: {predicted_high:.2f}")

            st.subheader("Next Day Price Range Prediction")
            predicted_low, predicted_high = predict_next_day_range(stock_data)
            st.write(f"Predicted Low: {float(predicted_low):.2f}")
            st.write(f"Predicted High: {float(predicted_high):.2f}")
        else:
            st.error("No data found. Please check the ticker symbol and date range.")
    else:
        st.error("Please enter a valid ticker symbol.")
