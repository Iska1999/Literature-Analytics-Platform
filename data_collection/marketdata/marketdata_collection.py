# data_collection/marketdata/marketdata_collection.py
import yfinance as yf
import pandas as pd
from datetime import date
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from utils.db_mgr import DatabaseManager

def fetch_sector_prices(etfs, start=str, end=None):
    """Fetch ETF close prices from YFinance then calculate the monthly average"""
    if end is None:
        end = pd.to_datetime("today").strftime("%Y-%m-%d")

    all_data = []

    for sector, ticker in etfs.items():
        print(f"Fetching {sector} ({ticker})...")
        df = yf.Ticker(ticker).history(start=start, end=end)

        if df.empty:
            print(f"No data for {ticker}")
            continue

        df["month"] = df.index.to_period("M").astype(str)
        monthly_avg = (
            df.groupby("month")["Close"]
            .mean()
            .reset_index()
            .rename(columns={"Close": "avg_close_price"})
        )
        monthly_avg["sector"] = sector
        all_data.append(monthly_avg)

    final_df = pd.concat(all_data, ignore_index=True)
    return final_df[["month", "sector", "avg_close_price"]]

# Validate configuration
try:
    Config.validate()
except ValueError as e:
    print(f"Configuration error: {e}")
    print("Please ensure your .env file is properly configured with DB_HOST, DB_USER, and DB_PASSWORD")
    sys.exit(1)

# Get today's date and calculate start date
today = date.today()
start = today.replace(year=today.year - Config.MARKET_LOOKBACK_YEARS)

# Fetch sector prices
print(f"Fetching market data for the last {Config.MARKET_LOOKBACK_YEARS} years...")
sector_prices = fetch_sector_prices(Config.SECTOR_ETFS, start)

# Calculate monthly % change in price
sector_prices["price_change"] = sector_prices.groupby("sector")["avg_close_price"].pct_change()

# Drop first row per sector because we are calculating % change
sector_prices = sector_prices.dropna()

# Upload to database
try:
    db_mgr = DatabaseManager()

    print("Uploading market data to database...")
    
    # Create database if it doesn't exist
    db_mgr.create_database_if_not_exists()
    
    # Upload to server
    db_mgr.write_table(sector_prices, 'marketdata')
    
    print("Market data successfully uploaded to database.")
    
except Exception as e:
    print(f"Error uploading to database: {e}")
    
finally:
    # Dispose of database connections
    db_mgr.dispose()
    print("Database connections closed.")