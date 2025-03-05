import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import datetime

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("Shanghai.csv", parse_dates=["Date"], index_col="Date")

df = load_data()

# Sidebar for user input
st.sidebar.title("Shanghai Stock Analysis")
company = st.sidebar.selectbox("Select a Company", df["Symbol"].unique())

company_data = df[df["Symbol"] == company]

# Convert index to DateTime and set frequency
company_data.index = pd.to_datetime(company_data.index)
company_data = company_data.asfreq("D")
company_data["Close"] = company_data["Close"].interpolate()  # Handle missing values

# Plot historical stock prices
st.subheader(f"Stock Price of {company}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=company_data.index, y=company_data["Close"], mode="lines", name="Close Price"))
fig.update_layout(title="Historical Stock Prices", xaxis_title="Date", yaxis_title="Stock Price")
st.plotly_chart(fig)

# Stock Price Forecast using ARIMA
st.subheader(f"Stock Price Forecast for {company} using ARIMA")

train_data = company_data["Close"].dropna()

# ADF Test before differencing
result_before = adfuller(train_data)
st.write("**ADF Test Before Differencing:**")
st.write(f"ADF Statistic: {result_before[0]}")
st.write(f"p-value: {result_before[1]}")

# Apply differencing if not stationary
diff_needed = result_before[1] > 0.05
if diff_needed:
    st.write("Data is NOT stationary. Applying Differencing...")
    train_data_diff = train_data.diff().dropna()
    
    # ADF Test after differencing
    result_after = adfuller(train_data_diff)
    st.write("**ADF Test After Differencing:**")
    st.write(f"ADF Statistic: {result_after[0]}")
    st.write(f"p-value: {result_after[1]}")
else:
    st.write("Data is already stationary. No differencing applied.")
    train_data_diff = train_data  # No differencing needed

# Train ARIMA Model
model = ARIMA(train_data_diff, order=(5,1,0))  # (p,d,q) values can be fine-tuned
model_fit = model.fit()

# User input for forecast period
forecast_steps = st.sidebar.slider("Select Forecast Days", min_value=5, max_value=90, value=30, step=5)

# Generate Forecast
forecast_diff = model_fit.forecast(steps=forecast_steps)

# Convert differenced forecast back to original scale
if diff_needed:
    forecast = train_data.iloc[-1] + forecast_diff.cumsum()
else:
    forecast = forecast_diff

# Create forecast index
forecast_index = pd.date_range(start=company_data.index[-1], periods=forecast_steps+1, freq="D")[1:]

# Create Plotly figure for forecast
fig_forecast = go.Figure()

# Actual Data
fig_forecast.add_trace(go.Scatter(x=company_data.index, y=company_data["Close"], 
                                  mode="lines", name="Actual Prices"))

# Forecast Data
fig_forecast.add_trace(go.Scatter(x=forecast_index, y=forecast, 
                                  mode="lines", name="Forecasted Prices", line=dict(dash="dash")))

fig_forecast.update_layout(title=f"ARIMA Forecast for {company}", 
                           xaxis_title="Date", yaxis_title="Stock Price")

# Display forecast graph
st.plotly_chart(fig_forecast)

# Display forecasted data
forecast_df = pd.DataFrame({"Date": forecast_index, "Forecasted Close": forecast.values})
forecast_df.set_index("Date", inplace=True)
st.subheader("Forecasted Data")
st.dataframe(forecast_df)

# ----------------------------------------------------------------------------------
# ✅ Fetch actual stock prices for the forecast period & Compare with Predictions
# ----------------------------------------------------------------------------------

# Get start and end date from forecast index
s = forecast_index.min()
e = forecast_index.max()
# Download actual stock data
# Create a dictionary for easy lookup of stock indices

stocks = [
    ('600028', 'SYNOPEC_CORP'),
    ('600030', 'CITIC'),
    ('600031', 'SANY'),
    ('600036', 'CMB'),
    ('600048', 'PDH'),
    ('600050', 'CHINA_UNICOM'),
    ('600150', 'CSSC_HOLDINGS'),
    ('600276', 'HENGRUI_PHARMA'),
    ('600309', 'WANHUA'),
    ('600406', 'NARI_TECH'),
    ('600438', 'TONGWEI'),
    ('600519', 'KWEICHOW_MOUTAI'),
    ('600690', 'HAIER_SMART_HOME'),
    ('600809', 'SHANXI_FEN_WINE'),
    ('600887', 'YILI'),
    ('600900', 'CYPC'),
    ('601012', 'LONGI'),
    ('601088', 'CSEC'),
    ('601166', 'INDUSTRIAL_BANK'),
    ('601225', 'SHCI'),
    ('601288', 'AGRICULTURAL_BANK_OF_CHINA'),
    ('601318', 'PING_AN'),
    ('601328', 'BANKCOMM'),
    ('601390', 'CHINA_RAILWAY'),
    ('601398', 'ICBC')

]

stock_dict = {name: index for index, name in stocks}

# Get the stock index from the dictionary
company_index = stock_dict.get(company)  # This gives the correct stock index
if company_index:
    # Fetch actual stock prices for the forecast period
    actual_data = yf.download(company_index + ".SS", start=s, end=e)
    # Reset index to make "Date" a normal column
    actual_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    actual_data = actual_data[['Close']].reset_index()
    # Convert "Date" column to datetime format
    actual_data['Date'] = pd.to_datetime(actual_data['Date'])

    # Merge with forecasted data
    forecast_df = forecast_df.reset_index()
    merged_df = pd.merge(forecast_df, actual_data, on="Date", how="outer")

    # Calculate residuals
    merged_df['Residual'] = merged_df['Close'] - merged_df['Forecasted Close']

    # Display merged DataFrame
    st.subheader("Actual vs. Forecasted Prices & Residuals")
    st.dataframe(merged_df.dropna())

    # Plot comparison
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(x=merged_df["Date"], y=merged_df["Forecasted Close"], 
                                        mode="lines", name="Forecasted Prices", line=dict(dash="dot")))
    fig_comparison.add_trace(go.Scatter(x=merged_df["Date"], y=merged_df["Close"], 
                                        mode="lines", name="Actual Prices"))
    fig_comparison.add_trace(go.Scatter(x=merged_df["Date"], y=merged_df["Residual"], 
                                        mode="lines", name="Residuals", line=dict(color="red")))

    fig_comparison.update_layout(title=f"Actual vs. Forecasted Prices for {company}", 
                                 xaxis_title="Date", yaxis_title="Stock Price")

    st.plotly_chart(fig_comparison)
else:
    st.error("Selected company not found in stock list. Please check your selection.")