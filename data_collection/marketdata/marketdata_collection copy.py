import yfinance as yf
import pandas as pd

#Fetch_sector_prices fetches ETF close prices from YFinance then calculates the monthly average
def fetch_sector_prices(etfs, start=str, end=None):
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

from datetime import date, timedelta

#Fetch market ETFs for the five following sectors
sector_etfs = {
    "technology": "XLK",
    "healthcare": "XLV",
    "financials": "XLF",
    "energy": "XLE",
    "industrials": "XLI"
}

#Get today's date, then calculate date for 5 years ago
today = date.today()
start = today.replace(year=today.year - 5)
#Fetch sector prices for the last 5 years
sector_prices = fetch_sector_prices(sector_etfs,start)

#Calculate monthly % change in price
sector_prices["price_change"] = sector_prices.groupby("sector")["avg_close_price"].pct_change()
#Drop first row per sector because we are calculating % change
sector_prices = sector_prices.dropna()

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy import text

# parameters
host_ip = "3.148.234.227" # my server's IP address
id = "test1"
pw = "Test1234#"

# connect to mysql server
url = URL.create(
    drivername="mysql+pymysql",
    host=host_ip,
    port=3306,
    username= id,
    password=pw)

sqlEngine = create_engine(url)
sql_connection = sqlEngine.connect()

#sql_connection.execute(text('DROP DATABASE IF EXISTS literature_analytics_platform'))

#It's already been created in the previous script
#sql_connection.execute(text("CREATE DATABASE IF NOT EXISTS literature_analytics_platform"))

db_url = URL.create(
    drivername="mysql+pymysql",
    host=host_ip,
    port=3306,
    username=id,
    password=pw,
    database="literature_analytics_platform"
)

db_engine = create_engine(db_url)

# Upload to server
sector_prices.to_sql(con= db_engine, name= 'marketdata', if_exists = 'replace')

# Close connection
sql_connection.close()
sqlEngine.dispose()
db_engine.dispose()

print ("Connection closed.")
