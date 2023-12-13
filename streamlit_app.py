import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
from forex_python.converter import CurrencyRates
from datetime import datetime, timedelta
import plotly.express as px
import arch
import plotly.graph_objects as go
import os
from streamlit_option_menu import option_menu


# Get the path to the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Set the working directory
os.chdir(script_directory)

# Now you can load the image
# image_path = 'logo.png'
# image = Image.open(image_path)
# st.image(image, width=390)

# Create container for the sticky header https://discuss.streamlit.io/t/is-it-possible-to-create-a-sticky-header/33145/3
header = st.container()
header.title("💴 💵 💶 Currency Conversion App 💶 💵 💴")
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

# Header CSS https://discuss.streamlit.io/t/is-it-possible-to-create-a-sticky-header/33145/3
st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 2.875rem;
        background-color: green;
        z-index: 999;
        text-align: center;
    }
    .fixed-header {
        # border-bottom: 1px solid black;
    }
</style>
    """,
    unsafe_allow_html=True
)

# st.title('Currency Converter App')
# st.markdown("""This app interconverts the values of foreign currencies!""")

# Sidebar + Main panel
st.sidebar.header('Input Options')

# Sidebar - Currency price unit
currency_list = ['AUD', 'BGN', 'BRL', 'CAD', 'CHF', 'CNY', 'CZK', 'DKK', 'EUR', 'GBP', 'HKD', 'HRK', 'HUF', 'IDR', 'ILS', 'INR', 'ISK', 'JPY', 'KRW', 'MXN', 'MYR', 'NOK', 'NZD', 'PHP', 'PLN', 'RON', 'RUB', 'SEK', 'SGD', 'THB', 'TRY', 'USD', 'ZAR']
base_price_unit = st.sidebar.selectbox('Select the base currency for conversion', currency_list)
symbols_price_unit = st.sidebar.selectbox('Select the target currency to convert to', currency_list)

# Input amount to convert
amount_to_convert = st.sidebar.number_input('Enter the amount to convert', value=1.0, min_value=0.0)

# Make API request
api_url = "http://data.fixer.io/api/latest"
access_key = "f7bcc5b6e5fccc8aa3c372b3ef4d589e"
symbols = f"{base_price_unit},{symbols_price_unit}"

params = {"access_key": access_key, "symbols": symbols}
response = requests.get(api_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()

    # Create DataFrame from exchange rates
    rates_df = pd.DataFrame.from_dict(data['rates'], orient='index', columns=['Exchange Rate'])

    # Convert the amount
    one_eur_to_one_base_currency = rates_df.loc[base_price_unit, 'Exchange Rate']
    one_eur_to_one_target_currency = rates_df.loc[symbols_price_unit, 'Exchange Rate']
    one_base_to_one_target_currency = one_eur_to_one_target_currency / one_eur_to_one_base_currency
    converted_amount = amount_to_convert * one_base_to_one_target_currency

    # Display conversion result
    st.markdown("<h2 style='color:green; font-size:32px;'>Conversion Result</h2>", unsafe_allow_html=True)
    st.write(f"{amount_to_convert} {base_price_unit} is equal to: <div style='font-size:24px;'>{converted_amount:.2f} {symbols_price_unit}</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:green; font-size:32px;'>Exchange Rates</h2>", unsafe_allow_html=True)
    st.write("Equivalent to 1 Euro")
    st.dataframe(rates_df.style.format("{:.4f}"))
   

    
    #******************************************************************#
    #*************************Historical Data**************************#
    #******************************************************************#
    
    # Button to show exchange rate time series using Historical Data
    time_range_historical = st.sidebar.selectbox('Select time range for historical data', ['','1 Month', '3 Months', '6 Months', '1 Year', '5 Years'])
    
    # Calculate start date based on the selected time range
    end_date_historical = datetime.now()        
    if time_range_historical == '':
        pass
    elif time_range_historical == '1 Month':
        start_date_historical = end_date_historical - timedelta(days=30)
    elif time_range_historical == '3 Months':
        start_date_historical = end_date_historical - timedelta(days=30 * 3)
    elif time_range_historical == '6 Months':
        start_date_historical = end_date_historical - timedelta(days=30 * 6)
    elif time_range_historical == '1 Year':
        start_date_historical = end_date_historical - timedelta(days=365)
    elif time_range_historical == '5 Years':
        start_date_historical = end_date_historical - timedelta(days=365 * 5)
    else:
        st.warning("Invalid time range for historical data selected")
        start_date_historical = datetime.now() - timedelta(days=30)  # Default value

    if time_range_historical != '':
        try:
            # Get historical exchange rates from Forex Python
            c = CurrencyRates()

            # Get historical exchange rates for the specified date range
            rates_historical = {}
            current_date_historical = start_date_historical
            total_days_historical = (end_date_historical - start_date_historical).days  # Progress bar

            progress_bar = st.progress(0)  # Initialize progress bar

            with st.spinner("Downloading historical data..."):  # Progress bar
                while current_date_historical <= end_date_historical:
                    # Update progress bar
                    progress_percentage = int(((current_date_historical - start_date_historical).days / total_days_historical) * 100)  # Progress bar
                    progress_bar.progress(progress_percentage)  # Progress bar

                    rates = c.get_rate(base_price_unit, symbols_price_unit, current_date_historical)
                    rates_historical[current_date_historical] = rates
                    current_date_historical += timedelta(days=1)

            # Clear progress bar when done
            progress_bar.empty()

            # Create DataFrame for the time series
            df_historical = pd.DataFrame(list(rates_historical.items()), columns=['Date', 'Exchange Rate'])
            df_historical['Date'] = pd.to_datetime(df_historical['Date'])
            df_historical["Date"] = df_historical["Date"].dt.strftime('%Y-%m-%d')  # Change time format

            # Plot exchange rate time series for 1 unit
            st.markdown("<h2 style='color:green; font-size:32px;'>Exchange Rate Historical Data</h2>", unsafe_allow_html=True)
            fig = px.line()
            fig.add_trace(go.Scatter(x=df_historical['Date'], y=df_historical['Exchange Rate'], mode='lines', name='Historical Rates', line=dict(color='rgba(60, 114, 168, 1)')))
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error: {e}")
            
    #******************************************************************#
    #****************************AR-Forecast***************************#
    #******************************************************************#
            
    # Select time range for AR forecast
    time_range_forecast = st.sidebar.selectbox('Select time range for AR forecast', ['', '1 Month', '3 Months', '6 Months', '1 Year', '5 Years'])
    
    # Calculate steps based on the selected time range
    if time_range_forecast == '':
        time_step_forecast = 1
    elif time_range_forecast == '1 Month':
        time_step_forecast = 30
    elif time_range_forecast == '3 Months':
        time_step_forecast = 30 * 3
    elif time_range_forecast == '6 Months':
        time_step_forecast = 30 * 6
    elif time_range_forecast == '1 Year':
        time_step_forecast = 365
    elif time_range_forecast == '5 Years':
        time_step_forecast = 365 * 5
    else:
        st.warning("Invalid time range for forecast data selected")
        start_date_forecast = datetime.now() - timedelta(days=30)  # Default value

    
    if time_range_forecast != '':
        try:           
            # Get fit exchange rates from Forex Python
            c = CurrencyRates()

            # Get fit exchange rates for the specified date range
            rates_forecast = {}
            end_date_forecast = datetime.now()
            start_date_forecast = end_date_forecast - timedelta(days=365)  # 1 year default
            current_date_forecast = start_date_forecast
            total_days_forecast = (end_date_forecast - start_date_forecast).days

            progress_bar = st.progress(0)  # Initialize progress bar

            with st.spinner("Fit the model and make the forecast..."):
                while current_date_forecast <= end_date_forecast:
                    # Update progress bar
                    progress_percentage = int(((current_date_forecast - start_date_forecast).days / total_days_forecast) * 100)
                    progress_bar.progress(progress_percentage)

                    rates = c.get_rate(base_price_unit, symbols_price_unit, current_date_forecast)
                    rates_forecast[current_date_forecast] = rates
                    current_date_forecast += timedelta(days=1)

            # Clear progress bar when done
            progress_bar.empty()

            # Create DataFrame for the time series
            df_forecast_train = pd.DataFrame(list(rates_forecast.items()), columns=['Date', 'Exchange Rate'])
            df_forecast_train['Date'] = pd.to_datetime(df_forecast_train['Date'])
             
            # -----------------------------------------------------------------------------
            # ------------------------------ ARMA-GARCH ----------------------------------
            # -----------------------------------------------------------------------------
            
            # Make the Log Returns for better fit
            df_forecast_train_logret = np.diff(np.log(df_forecast_train['Exchange Rate']))  # calculate log-return
            df_forecast_train_logret = pd.DataFrame(df_forecast_train_logret)  # make a dataframe
            df_forecast_train_logret = df_forecast_train_logret.rename(columns={df_forecast_train_logret.columns[0]: 'Log Returns'})  # rename the column
            df_forecast_train_logret = df_forecast_train_logret.dropna()  # delete the NaN value
            
            # Specify the ARMA-GARCH model
            model = arch.arch_model(df_forecast_train_logret, vol='Garch', mean='AR', p=2, q=1, rescale=False)
            
            # Fit the model
            result = model.fit()
            
            # Forecast future volatility and returns
            forecast = result.forecast(horizon=time_step_forecast)
            
            # Extract the forecasted volatility
            forecast_volatility = np.sqrt(forecast.variance.dropna().values[-1, :])
            
            future_dates = pd.date_range(start=end_date_forecast + pd.DateOffset(1), periods=time_step_forecast, freq='D')
            
            # Simulate future returns based on forecasted volatility
            np.random.seed(42)
            forecast_returns = np.random.normal(0, 1, time_step_forecast) * forecast_volatility
            
            # Calculate future exchange rate values
            forecast_exchange_rates_arma = df_forecast_train['Exchange Rate'].iloc[-1] * forecast_returns
            forecast_exchange_rates_arma = np.exp(forecast_exchange_rates_arma.cumsum())
            
            # Make dataframe
            df_forecast_predict_arma = pd.DataFrame(forecast_exchange_rates_arma)
            df_forecast_predict_arma = df_forecast_predict_arma.rename(columns={df_forecast_predict_arma.columns[0]: 'Exchange Rate'})
            df_forecast_predict_arma["Date"] = pd.date_range(end_date_forecast + timedelta(days=1), end_date_forecast + timedelta(days=time_step_forecast), freq='D')
            
            # Move the "Date" column to the first position
            column_order = ["Date"] + [col for col in df_forecast_predict_arma if col != "Date"]
            df_forecast_predict_arma = df_forecast_predict_arma[column_order]
            
            df_forecast_total_arma = pd.concat([df_forecast_train, df_forecast_predict_arma])
            df_forecast_total_arma['Date'] = pd.to_datetime(df_forecast_total_arma['Date'])
            
            # Plot both df_forecast and forecast_df in the same plot
            st.markdown("<h2 style='color:green; font-size:32px;'>Forecast</h2>", unsafe_allow_html=True)
            fig = px.line()
            fig.add_trace(go.Scatter(x=df_forecast_total_arma['Date'], y=df_forecast_total_arma['Exchange Rate'], mode='lines', name='Historical', line=dict(color='rgba(60, 114, 168, 1)')))
            fig.add_trace(go.Scatter(x=df_forecast_predict_arma['Date'], y=df_forecast_predict_arma['Exchange Rate'], mode='lines', name='Forecast (ARMA)', line=dict(color='rgba(255, 255, 202 1)')))
           
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")
     
    # ******************************************************************#
    # ************************* Trading Strategy ************************#
    # ******************************************************************#
    
    # Select if MA Cross or not
    trading_strategy = st.sidebar.selectbox('Select Trading-Strategy', ['', 'MA-Cross', 'Bollinger-Bands'])
    
    if trading_strategy == '':
        pass
    
    if trading_strategy == 'MA-Cross':
        try:
            # Get fit exchange rates from Forex Python
            c = CurrencyRates()
    
            # Get fit exchange rates for the specified date range
            rates_ma_cross = {}
            end_date_ma_cross = datetime.now()
            start_date_ma_cross = end_date_ma_cross - timedelta(days=730)  # 2 year default
            current_date_ma_cross = start_date_ma_cross
            total_days_ma_cross = (end_date_ma_cross - start_date_ma_cross).days
    
            progress_bar = st.progress(0)  # Initialize progress bar
    
            with st.spinner("Compute the trading strategy..."):
                while current_date_ma_cross <= end_date_ma_cross:
                    # Update progress bar
                    progress_percentage = int(((current_date_ma_cross - start_date_ma_cross).days / total_days_ma_cross) * 100)
                    progress_bar.progress(progress_percentage)
    
                    rates = c.get_rate(base_price_unit, symbols_price_unit, current_date_ma_cross)
                    rates_ma_cross[current_date_ma_cross] = rates
                    current_date_ma_cross += timedelta(days=1)
    
            # Clear progress bar when done
            progress_bar.empty()
    
            # Create DataFrame for the time series
            df_ma_cross = pd.DataFrame(list(rates_ma_cross.items()), columns=['Date', 'Exchange Rate'])
            df_ma_cross['Date'] = pd.to_datetime(df_ma_cross['Date'])
    
            short_window = 50
            long_window = 200
    
            df_ma_cross['Short_MA'] = df_ma_cross['Exchange Rate'].rolling(window=short_window, min_periods=1).mean()
            df_ma_cross['Long_MA'] = df_ma_cross['Exchange Rate'].rolling(window=long_window, min_periods=1).mean()
    
            st.markdown("<h2 style='color:green; font-size:32px;'>MA-Cross</h2>", unsafe_allow_html=True)
            fig = px.line()
    
            # Add traces for Exchange Rate, Short MA, and Long MA
            fig.add_trace(go.Scatter(x=df_ma_cross['Date'], y=df_ma_cross['Exchange Rate'], mode='lines', name='Exchange Rate', line=dict(color='rgba(60, 114, 168, 1)')))
            fig.add_trace(go.Scatter(x=df_ma_cross['Date'], y=df_ma_cross['Short_MA'], mode='lines', name=f'Short {short_window}-Day MA', line=dict(color='rgba(203, 224, 199, 1)')))
            fig.add_trace(go.Scatter(x=df_ma_cross['Date'], y=df_ma_cross['Long_MA'], mode='lines', name=f'Long {long_window}-Day MA', line=dict(color='rgba(245, 199, 198, 1)')))
    
            # Add markers for Buy and Sell signals
            ma_cross_buy = df_ma_cross[(df_ma_cross['Short_MA'] > df_ma_cross['Long_MA']) & (df_ma_cross['Short_MA'].shift(1) <= df_ma_cross['Long_MA'].shift(1))]
            ma_cross_sell = df_ma_cross[(df_ma_cross['Short_MA'] < df_ma_cross['Long_MA']) & (df_ma_cross['Short_MA'].shift(1) >= df_ma_cross['Long_MA'].shift(1))]
    
            fig.add_trace(go.Scatter(x=ma_cross_buy['Date'], y=ma_cross_buy['Exchange Rate'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')))
            fig.add_trace(go.Scatter(x=ma_cross_sell['Date'], y=ma_cross_sell['Exchange Rate'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red')))
    
            # Show the plot
            st.plotly_chart(fig)
    
        except Exception as e:
            st.error(f"Error: {e}")
    
    if trading_strategy == 'Bollinger-Bands':
        try:
            # Get fit exchange rates from Forex Python
            c = CurrencyRates()
    
            # Get fit exchange rates for the specified date range
            rates_bollinger_bands = {}
            end_date_bollinger_bands = datetime.now()
            start_date_bollinger_bands = end_date_bollinger_bands - timedelta(days=730)  # 2 year default
            current_date_bollinger_bands = start_date_bollinger_bands
            total_days_bollinger_bands = (end_date_bollinger_bands - start_date_bollinger_bands).days
    
            progress_bar = st.progress(0)  # Initialize progress bar
    
            with st.spinner("Compute the trading strategy..."):
                while current_date_bollinger_bands <= end_date_bollinger_bands:
                    # Update progress bar
                    progress_percentage = int(((current_date_bollinger_bands - start_date_bollinger_bands).days / total_days_bollinger_bands) * 100)
                    progress_bar.progress(progress_percentage)
    
                    rates = c.get_rate(base_price_unit, symbols_price_unit, current_date_bollinger_bands)
                    rates_bollinger_bands[current_date_bollinger_bands] = rates
                    current_date_bollinger_bands += timedelta(days=1)
    
            # Clear progress bar when done
            progress_bar.empty()
    
            # Create DataFrame for the time series
            df_bollinger_bands = pd.DataFrame(list(rates_bollinger_bands.items()), columns=['Date', 'Exchange Rate'])
            df_bollinger_bands['Date'] = pd.to_datetime(df_bollinger_bands['Date'])
    
            # Calculate the 20-period SMA (X)
            df_bollinger_bands['SMA_20'] = df_bollinger_bands['Exchange Rate'].rolling(window=20).mean()
    
            # Calculate standard deviation
            std_dev = df_bollinger_bands['Exchange Rate'].rolling(window=20).std()
    
            # Calculate upper and lower Bollinger Bands
            df_bollinger_bands['Upper_2stdv'] = df_bollinger_bands['SMA_20'] + 2 * std_dev
            df_bollinger_bands['Upper_1stdv'] = df_bollinger_bands['SMA_20'] + std_dev
            df_bollinger_bands['Lower_1stdv'] = df_bollinger_bands['SMA_20'] - std_dev
            df_bollinger_bands['Lower_2stdv'] = df_bollinger_bands['SMA_20'] - 2 * std_dev
    
            # Create signals column
            df_bollinger_bands['Signal'] = 0
    
            # Buy Signal (Long Entry)
            df_bollinger_bands.loc[df_bollinger_bands['Exchange Rate'] > df_bollinger_bands['Upper_1stdv'], 'Signal'] = 1
    
            # Sell Signal (Short Entry)
            df_bollinger_bands.loc[df_bollinger_bands['Exchange Rate'] < df_bollinger_bands['Lower_1stdv'], 'Signal'] = -1

            st.markdown("<h2 style='color:green; font-size:32px;'>Bollinger-Bands</h2>", unsafe_allow_html=True)
            fig = px.line()
    
            # Create an interactive plot
            fig = go.Figure()
    
            # Plot exchange rate and SMA
            fig.add_trace(go.Scatter(x=df_bollinger_bands['Date'], y=df_bollinger_bands['Exchange Rate'], mode='lines', name='Exchange Rate', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df_bollinger_bands['Date'], y=df_bollinger_bands['SMA_20'], mode='lines', name='20-period SMA', line=dict(dash='dash')))
    
            # Plot Buy signals
            buy_signals = df_bollinger_bands[df_bollinger_bands['Signal'] == 1]
            fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Exchange Rate'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')))
    
            # Plot Sell signals
            sell_signals = df_bollinger_bands[df_bollinger_bands['Signal'] == -1]
            fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Exchange Rate'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red')))
    
            # Plot Bollinger Bands
            fig.add_trace(go.Scatter(x=df_bollinger_bands['Date'].tolist() + df_bollinger_bands['Date'].tolist()[::-1],
                                     y=df_bollinger_bands['Upper_2stdv'].tolist() + df_bollinger_bands['Upper_1stdv'].tolist()[::-1],
                                     fill='toself', fillcolor='rgba(0, 128, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                                     name='Buy Zone'))
            fig.add_trace(go.Scatter(x=df_bollinger_bands['Date'].tolist() + df_bollinger_bands['Date'].tolist()[::-1],
                                     y=df_bollinger_bands['Upper_1stdv'].tolist() + df_bollinger_bands['Lower_1stdv'].tolist()[::-1],
                                     fill='toself', fillcolor='rgba(255,255,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                     name='Neutral Zone'))
            fig.add_trace(go.Scatter(x=df_bollinger_bands['Date'].tolist() + df_bollinger_bands['Date'].tolist()[::-1],
                                     y=df_bollinger_bands['Lower_2stdv'].tolist() + df_bollinger_bands['Lower_1stdv'].tolist()[::-1],
                                     fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                     name='Sell Zone'))
    
            # Show the Plotly figure
            st.plotly_chart(fig)
    
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.error(f"Error: {response.status_code}")
