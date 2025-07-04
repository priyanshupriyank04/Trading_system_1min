#  Step 1: Import Required Libraries
import os                # For environment variables and file handling
import time              # For adding delays where needed
import datetime          # To handle timestamps
import pandas as pd      # For working with dataframes
import math              # For working with math related functions
import psycopg2          # PostgreSQL database connection
import logging           # For structured logging
from kiteconnect import KiteConnect  # Zerodha API connection


# Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logging.info(" Required libraries imported successfully.")

#  Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#  API Credentials
API_KEY = "8re7mjcm2btaozwf"  #  Replace with your actual API key
API_SECRET = "fw8gm7wfeclcic9rlkp0tbzx4h2ss2n1"  # Replace with your actual API secret
ACCESS_TOKEN_FILE = "access_token.txt"

#  Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)


def get_access_token():
    """
    Checks if the access token exists and is valid. If not, prompts the user to manually enter a new one.
    """
    #  Step 1: Check if access_token.txt exists
    if os.path.exists(ACCESS_TOKEN_FILE):
        with open(ACCESS_TOKEN_FILE, "r") as file:
            access_token = file.read().strip()
            kite.set_access_token(access_token)
            logging.info(" Found existing access token. Attempting authentication...")

            #  Step 2: Validate access token
            try:
                profile = kite.profile()  #  API call to validate token
                logging.info(f"API Authentication Successful! User: {profile['user_name']}")
                return access_token  # ✅Return the valid token
            except Exception as e:
                logging.warning(f" Invalid/Expired Access Token: {e}")
    
    #  Step 3: If token is invalid or file does not exist, ask the user for a new one
    logging.info(" Fetching new access token...")

    request_token_url = kite.login_url()
    logging.info(f" Go to this URL, authorize, and retrieve the request token: {request_token_url}")
    
    request_token = input(" Paste the request token here: ").strip()

    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]

        # 🔹 Step 4: Save the new access token
        with open(ACCESS_TOKEN_FILE, "w") as file:
            file.write(access_token)

        logging.info(" New access token saved successfully!")
        return access_token
    except Exception as e:
        logging.error(f" Failed to generate access token: {e}")
        return None

#  Get Access Token
access_token = get_access_token()

if access_token:
    logging.info(" API is now authenticated and ready to use!")
else:
    logging.error(" API authentication failed. Please check credentials and try again.")

from psycopg2 import sql

#  Database Configuration
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "admin123"
DB_HOST = "localhost"
DB_PORT = "5432"

#  Connect to PostgreSQL
def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True  # Enable autocommit mode
        return conn
    except Exception as e:
        logging.error(f" Failed to connect to database: {e}")
        return None
connect_to_db()

#Fetch nifty index price 
def get_nifty50_price():
    """
    Fetches the real-time Nifty 50 index price with retry logic.
    """
    retries = 5
    for attempt in range(retries):
        try:
            nifty_data = kite.ltp("NSE:NIFTY 50")
            nifty_price = nifty_data["NSE:NIFTY 50"]["last_price"]
            logging.info(f"Fetched Nifty 50 Index Price: {nifty_price}")
            return nifty_price
        except Exception as e:
            logging.warning(f" Attempt {attempt + 1}/{retries}: Error fetching Nifty 50 price: {e}")
            time.sleep(1)  # Wait before retrying
    
    logging.error(" Failed to fetch Nifty 50 price after retries.")
    return None

# Get all available option instruments
option_instruments = kite.instruments("NFO")


#Fetch nifty 50 option price 
def get_nifty50_option_price(option_token):
    """
    Fetches the real-time price of the Nifty 50 option contract from Zerodha API.
    :param option_token: Instrument token of the Nifty 50 option contract.
    :return: Last traded price (LTP) of the option contract.
    """
    try:
        logging.info(f"Fetching LTP for token: {option_token}")
        
        #  Fetch LTP from Zerodha API
        option_data = kite.ltp(option_token)

        #  Log full API response to check the structure
        logging.info(f" Full LTP API response: {option_data}")

        #  Use token as a string directly, without "NFO:"
        token_str = str(option_token)

        if token_str in option_data:
            option_price = option_data[token_str]["last_price"]
            logging.info(f" Fetched Nifty 50 Option Price: {option_price}")
            return option_price
        else:
            logging.error(f" LTP response does not contain expected token: {option_token}")
            return None

    except Exception as e:
        logging.error(f" Error fetching Nifty 50 option price: {e}")
        return None  # Return None if fetching fails


import datetime
import logging

#Fetch nifty custom weekly expiry date
def get_custom_nifty_expiry():
    """
    Returns a manually mapped expiry date based on today's date for Nifty 50 contracts.
    """
    today = datetime.date.today()



    if today <= datetime.date(today.year, 7, 10):
        return datetime.date(today.year, 7, 10)  # Weekly expiry
    elif today <= datetime.date(today.year, 7, 17):
        return datetime.date(today.year, 7, 17)  # Weekly expiry
    elif today <= datetime.date(today.year, 7, 24):
        return datetime.date(today.year, 7, 24)  # Weekly expiry
    elif today <= datetime.date(today.year, 7, 31):
        return datetime.date(today.year, 7, 31)  # Monthly expiry (default)
    elif today <= datetime.date(today.year, 8, 7):
        return datetime.date(today.year, 8, 7)  # Weekly expiry
    else:
        logging.error(" No predefined expiry date available for current date.")
        return None


# Find the nearest OTM CE contract based on the Nifty index price
def get_nearest_otm_ce_contract(nifty_index_price):
    try:
        expiry = get_custom_nifty_expiry()
        logging.info(f"Expiry date is {expiry}")
        if not expiry:
            return None, None

        ce_options = [
            inst for inst in option_instruments
            if inst["name"] == "NIFTY"
            and inst["instrument_type"] == "CE"
            and inst["expiry"] == expiry
        ]

        if not ce_options:
            logging.warning(" No CE contracts found for selected expiry.")
            return None, None

        #  Nearest OTM CE: Next strike ABOVE the current price
        otm_ce_strike = int(math.ceil(nifty_index_price / 50) * 50)

        best_ce = min(ce_options, key=lambda x: abs(x["strike"] - otm_ce_strike))

        ltp = get_nifty50_option_price(best_ce["instrument_token"])
        logging.info(f" CE OTM Contract: {best_ce['tradingsymbol']} | Token: {best_ce['instrument_token']} | 💰 LTP: {ltp}")
        return best_ce["tradingsymbol"], best_ce["instrument_token"]

    except Exception as e:
        logging.error(f" Error in get_nearest_otm_ce_contract: {e}")
        return None, None

# Find the nearest OTM PE contract based on the Nifty index price
def get_nearest_otm_pe_contract(nifty_index_price):
    try:
        expiry = get_custom_nifty_expiry()
        if not expiry:
            return None, None

        pe_options = [
            inst for inst in option_instruments
            if inst["name"] == "NIFTY"
            and inst["instrument_type"] == "PE"
            and inst["expiry"] == expiry
        ]

        if not pe_options:
            logging.warning(" No PE contracts found for selected expiry.")
            return None, None

        #  Nearest OTM PE: Next strike BELOW the current price
        otm_pe_strike = int(math.floor(nifty_index_price / 50) * 50)

        best_pe = min(pe_options, key=lambda x: abs(x["strike"] - otm_pe_strike))

        ltp = get_nifty50_option_price(best_pe["instrument_token"])
        logging.info(f" PE OTM Contract: {best_pe['tradingsymbol']} | Token: {best_pe['instrument_token']} | 💰 LTP: {ltp}")
        return best_pe["tradingsymbol"], best_pe["instrument_token"]

    except Exception as e:
        logging.error(f" Error in get_nearest_otm_pe_contract: {e}")
        return None, None

def get_nearest_otm_ce_pe_tables(nifty_price):
    """
    Fetch nearest OTM CE & PE contracts based on latest NIFTY price
    and return their respective 1-minute OHLC table names and metadata.
    """
    ce_symbol, ce_token = get_nearest_otm_ce_contract(nifty_price)
    pe_symbol, pe_token = get_nearest_otm_pe_contract(nifty_price)

    if not ce_symbol or not pe_symbol:
        logging.error(" Failed to fetch nearest OTM CE/PE contracts.")
        return None

    return {
        "CE": {
            "symbol": ce_symbol,
            "token": ce_token,
            "table_1min": f"{ce_symbol.lower()}_ohlc_1min"
        },
        "PE": {
            "symbol": pe_symbol,
            "token": pe_token,
            "table_1min": f"{pe_symbol.lower()}_ohlc_1min"
        }
    }

def create_nearest_otm_ohlc_tables(ce_symbol, pe_symbol):
    """
    Create 1-minute OHLC tables for nearest OTM CE and PE contracts.
    Drops existing tables and recreates with required structure for:
    - ADX and DI indicators
    - CBOE (Stoch RSI + Market Index + Odds) indicator
    """

    conn = connect_to_db()
    if conn:
        try:
            cur = conn.cursor()

            # Define table names
            ce_table_1min = f"{ce_symbol.lower()}_ohlc_1min"
            pe_table_1min = f"{pe_symbol.lower()}_ohlc_1min"

            # Drop existing 1-min tables if exist
            for table in [ce_table_1min, pe_table_1min]:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                logging.info(f" Dropped existing table if present: {table}")

            # Create CE 1-min table
            cur.execute(f"""
                CREATE TABLE {ce_table_1min} (
                    timestamp TIMESTAMPTZ PRIMARY KEY,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume FLOAT,
                    ema_5 FLOAT,                    
                    max_channel FLOAT,
                    min_channel FLOAT,
                    supertrend_avg FLOAT,
                    stoch_rsi_k FLOAT,
                    stoch_rsi_d FLOAT
                );
            """)

            # Create PE 1-min table
            cur.execute(f"""
                CREATE TABLE {pe_table_1min} (
                    timestamp TIMESTAMPTZ PRIMARY KEY,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume FLOAT,
                    ema_5 FLOAT,
                    max_channel FLOAT,
                    min_channel FLOAT,
                    supertrend_avg FLOAT,
                    stoch_rsi_k FLOAT,
                    stoch_rsi_d FLOAT
                );
            """)

            conn.commit()
            logging.info(" Created fresh 1-minute OHLC tables for Nearest OTM CE/PE successfully.")

            cur.close()
            conn.close()

        except Exception as e:
            logging.error(f" Failed to create 1-minute OHLC tables for Nearest OTM CE/PE: {e}")


#  Step 1: Fetch Latest Nifty 50 Price
nifty_price = get_nifty50_price()

#  Step 2: Get Nearest OTM CE/PE Contract Details
nearest_contracts = get_nearest_otm_ce_pe_tables(nifty_price)

if nearest_contracts:
    print("\n Nearest OTM CE Contract Details:")
    print(nearest_contracts["CE"])

    print("\n Nearest OTM PE Contract Details:")
    print(nearest_contracts["PE"])

    #  Step 3: Create Fresh OHLC Tables for Nearest OTM CE & PE
    create_nearest_otm_ohlc_tables(
        nearest_contracts["CE"]["symbol"],
        nearest_contracts["PE"]["symbol"]
    )

else:
    print(" Failed to fetch nearest CE/PE contracts. Exiting...")

#Fetch CE contracts in range nifty-500,nifty+500
def get_ce_contracts(nifty_index_price):
    """
    Fetches all CE contracts in the range of NIFTY_INDEX +/- 500 with step of 50
    Returns list of dict with symbol and instrument_token
    """
    try:
        expiry = get_custom_nifty_expiry()
        if not expiry:
            return []

        ce_options = [
            inst for inst in option_instruments
            if inst["name"] == "NIFTY"
            and inst["instrument_type"] == "CE"
            and inst["expiry"] == expiry
        ]

        if not ce_options:
            logging.warning(" No CE contracts found for selected expiry.")
            return []

        lower_limit = int(math.floor((nifty_index_price - 500) / 50) * 50)
        upper_limit = int(math.floor((nifty_index_price + 500) / 50) * 50)

        selected_contracts = []

        for strike in range(lower_limit, upper_limit + 1, 50):
            for option in ce_options:
                if option["strike"] == strike:
                    selected_contracts.append({
                        "symbol": option["tradingsymbol"],
                        "token": option["instrument_token"]
                    })

        logging.info(f" Total CE Contracts Found: {len(selected_contracts)}")
        return selected_contracts

    except Exception as e:
        logging.error(f" Error in get_ce_contracts: {e}")
        return []

#Fetch PE contracts in the range nifty-500,nifty+500
def get_pe_contracts(nifty_index_price):
    """
    Fetches all PE contracts in the range of NIFTY_INDEX +/- 500 with step of 50
    Returns list of dict with symbol and instrument_token
    """
    try:
        expiry = get_custom_nifty_expiry()
        if not expiry:
            return []

        pe_options = [
            inst for inst in option_instruments
            if inst["name"] == "NIFTY"
            and inst["instrument_type"] == "PE"
            and inst["expiry"] == expiry
        ]

        if not pe_options:
            logging.warning(" No PE contracts found for selected expiry.")
            return []

        lower_limit = int(math.floor((nifty_index_price - 500) / 50) * 50)
        upper_limit = int(math.floor((nifty_index_price + 500) / 50) * 50)

        selected_contracts = []

        for strike in range(lower_limit, upper_limit + 1, 50):
            for option in pe_options:
                if option["strike"] == strike:
                    selected_contracts.append({
                        "symbol": option["tradingsymbol"],
                        "token": option["instrument_token"]
                    })

        logging.info(f" Total PE Contracts Found: {len(selected_contracts)}")
        return selected_contracts

    except Exception as e:
        logging.error(f" Error in get_pe_contracts: {e}")
        return []

#Fetch all the contracts 
def fetch_contracts(nifty_index_price):
    """
    Fetches all CE and PE contracts in the range of NIFTY_INDEX +/- 500 with step of 50
    Returns a dictionary with lists of CE and PE contracts
    """
    return {
        "ce_contracts": get_ce_contracts(nifty_index_price),
        "pe_contracts": get_pe_contracts(nifty_index_price)
    }

nifty_price = get_nifty50_price()
contracts = fetch_contracts(nifty_price)

# Print to see the result
print("CE Contracts:")
for contract in contracts['ce_contracts']:
    print(contract)

print("PE Contracts:")
for contract in contracts['pe_contracts']:
    print(contract)

#List of market holidays
#  Market Holidays for 2025
MARKET_HOLIDAYS = {
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10",
    "2025-04-14", "2025-04-18", "2025-05-01", "2025-08-15",
    "2025-08-27", "2025-10-02", "2025-10-21", "2025-10-22",
    "2025-11-05", "2025-12-25"
}

#  Fetch last trading day's OHLC data for a single table (dynamic)
def fetch_last_trading_day_ohlc_for_table(table_name, instrument_token, interval="minute"):
    """
    Fetches last trading day's OHLC data for the given table and instrument token.
    Used when switching nearest ITM CE/PE contracts dynamically.

    Args:
        table_name (str): Name of the database table (example: nifty25apr23850ce_ohlc_1min)
        instrument_token (int): Instrument token of the option contract
        interval (str): "minute" for 1-min data, "5minute" for 5-min data
    """

    try:
        conn = connect_to_db()
        if not conn:
            return None

        cur = conn.cursor()

        now = datetime.datetime.now()
        last_trading_day = now - datetime.timedelta(days=1)

        #  Ensure we pick previous working day (skip weekends and holidays)
        while last_trading_day.strftime("%Y-%m-%d") in MARKET_HOLIDAYS or last_trading_day.weekday() in [5, 6]:
            last_trading_day -= datetime.timedelta(days=1)

        from_date = last_trading_day.strftime("%Y-%m-%d 09:15:00")
        to_date = last_trading_day.strftime("%Y-%m-%d 15:30:00")

        logging.info(f" Fetching last trading day's {interval} data for token {instrument_token}: {from_date} to {to_date}")

        # Fetch Historical Data
        historical_data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )

        if historical_data:
            df = pd.DataFrame(historical_data)
            df["timestamp"] = pd.to_datetime(df["date"])
            df.drop(columns=["date"], inplace=True)

            for _, row in df.iterrows():
                cur.execute(f"""
                    INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO NOTHING;
                """, (
                    row["timestamp"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"]
                ))

            conn.commit()
            logging.info(f"Last trading day's {interval} data inserted into {table_name} successfully.")
        else:
            logging.warning(f" No historical data found for token {instrument_token}.")

        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f" Error fetching last trading day's {interval} data for table {table_name}: {e}")
        return None


#  Fetch & Merge Today's Data for a Specific 1-Min Table (CE or PE)
def fetch_and_merge_ohlc_for_table(table_name, instrument_token, interval="minute"):
    """
    Fetch and merge today's OHLC data (including last trading day's) for a 1-minute table.
    """

    conn = connect_to_db()
    if not conn:
        return

    cur = conn.cursor()

    # Step 1: Fetch Last Trading Day's Data
    fetch_last_trading_day_ohlc_for_table(table_name, instrument_token, interval)

    # Step 2: Fetch Today's Data up to current completed candle
    now = datetime.datetime.now()
    from_date = now.replace(hour=9, minute=15, second=0).strftime("%Y-%m-%d %H:%M:%S")
    to_date = now.replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


    logging.info(f" Fetching today's {interval} data for {table_name} from {from_date} to {to_date}")

    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )

        if data:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["date"])
            df.drop(columns=["date"], inplace=True)

            for _, row in df.iterrows():
                cur.execute(f"""
                    INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO NOTHING;
                """, (
                    row["timestamp"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"]  
                ))

            conn.commit()
            logging.info(f" Today's {interval} data merged into {table_name} successfully!")

        else:
            logging.warning(f" No today's {interval} data fetched for {table_name}")

    except Exception as e:
        logging.error(f" Error fetching today's {interval} data for {table_name}: {e}")

    finally:
        cur.close()
        conn.close()

fetch_and_merge_ohlc_for_table(nearest_contracts["CE"]["table_1min"], nearest_contracts["CE"]["token"], "minute")
fetch_and_merge_ohlc_for_table(nearest_contracts["PE"]["table_1min"], nearest_contracts["PE"]["token"], "minute")

import datetime

def create_nearest_otm_contracts_table():
    """
    Drops (if exists) and creates the nearest_otm_contracts table cleanly,
    tailored for 1-minute strategies (no 5-min tables).
    """
    try:
        conn = connect_to_db()
        if not conn:
            return

        cur = conn.cursor()

        # Drop table if exists
        cur.execute("DROP TABLE IF EXISTS nearest_otm_contracts;")
        logging.info(" Dropped existing nearest_otm_contracts table.")

        # Create fresh table with only 1-min references
        cur.execute("""
            CREATE TABLE nearest_otm_contracts (
                ce_symbol TEXT,
                ce_token BIGINT,
                ce_table_1min TEXT,
                pe_symbol TEXT,
                pe_token BIGINT,
                pe_table_1min TEXT,
                update_timestamp TIMESTAMPTZ
            );
        """)
        conn.commit()

        logging.info(" Created fresh nearest_otm_contracts table successfully!")

        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f" Error creating nearest_otm_contracts table: {e}")


create_nearest_otm_contracts_table()


def update_nearest_otm_contracts():
    """
    Fetches the latest Nifty price, finds nearest OTM CE/PE contracts,
    and updates the nearest_otm_contracts table with only 1min table info.
    """
    try:
        # Step 1: Fetch live Nifty price
        nifty_price = get_nifty50_price()
        if nifty_price is None:
            logging.warning(" Failed to fetch Nifty 50 price while updating nearest OTM contracts.")
            return

        # Step 2: Get nearest OTM CE/PE contracts
        nearest_otm = get_nearest_otm_ce_pe_tables(nifty_price)
        if nearest_otm is None:
            logging.warning(" Failed to fetch nearest OTM CE/PE contracts.")
            return

        current_timestamp = datetime.datetime.now()

        # Step 3: Connect to DB
        conn = connect_to_db()
        if not conn:
            logging.error(" Database connection failed during nearest OTM update.")
            return
        
        cur = conn.cursor()

        # Step 4: Clear old row
        cur.execute("TRUNCATE TABLE nearest_otm_contracts;")

        # Step 5: Insert new row
        cur.execute("""
            INSERT INTO nearest_otm_contracts (
                ce_symbol, ce_token, ce_table_1min,
                pe_symbol, pe_token, pe_table_1min,
                update_timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (
            nearest_otm["CE"]["symbol"],
            nearest_otm["CE"]["token"],
            nearest_otm["CE"]["table_1min"],
            nearest_otm["PE"]["symbol"],
            nearest_otm["PE"]["token"],
            nearest_otm["PE"]["table_1min"],
            current_timestamp
        ))

        conn.commit()
        cur.close()
        conn.close()

        logging.info(f" Nearest OTM CE/PE contracts updated successfully at {current_timestamp}!")

    except Exception as e:
        logging.error(f" Error while updating nearest OTM contracts: {e}")


update_nearest_otm_contracts()

#  Initialize Current CE and PE Tokens from nearest_otm_contracts
def initialize_current_tokens():
    """
    Fetches the latest CE and PE tokens from the nearest_otm_contracts table
    and initializes global variables: current_ce_token and current_pe_token.
    """
    global current_ce_token, current_pe_token

    try:
        conn = connect_to_db()
        if not conn:
            logging.error(" Failed to connect to DB while initializing current tokens.")
            return False

        cur = conn.cursor()

        cur.execute("""
            SELECT ce_token, pe_token 
            FROM nearest_otm_contracts
            ORDER BY update_timestamp DESC
            LIMIT 1;
        """)
        result = cur.fetchone()

        if result and len(result) == 2:
            current_ce_token = result[0]
            current_pe_token = result[1]
            logging.info(f" Initialized Current CE Token: {current_ce_token}, PE Token: {current_pe_token}")
            success = True
        else:
            logging.error(" No data found in nearest_otm_contracts table to initialize tokens.")
            success = False

        cur.close()
        conn.close()
        return success

    except Exception as e:
        logging.error(f" Error initializing current tokens: {e}")
        return False


#  Call this during setup
initialize_current_tokens()

import pandas as pd
import numpy as np

def calculate_ema_for_table(table_name: str, length: int):
    """
    Calculates Exponential Moving Average (EMA) of 'close' for a given length
    and updates the specified table's corresponding column (ema_<length>).
    """
    column_name = f"ema_{length}"
    try:
        conn = connect_to_db()
        if not conn:
            logging.error(" DB connection failed for EMA calculation.")
            return

        cur = conn.cursor()

        #  Step 1: Ensure EMA column exists
        cur.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s;
        """, (table_name, column_name))
        result = cur.fetchone()

        if not result:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} FLOAT;")
            logging.info(f" Added missing column: {column_name} to table {table_name}")

        #  Step 2: Fetch close prices
        cur.execute(f"SELECT timestamp, close FROM {table_name} ORDER BY timestamp ASC;")
        rows = cur.fetchall()
        if not rows:
            logging.warning(f" No data found in table {table_name} for EMA-{length} calculation.")
            return

        df = pd.DataFrame(rows, columns=["timestamp", "close"])

        #  Step 3: Calculate EMA
        df[column_name] = df["close"].ewm(span=length, adjust=False).mean()

        #  Step 4: Update table
        for _, row in df.iterrows():
            cur.execute(
                f"""
                UPDATE {table_name}
                SET {column_name} = %s
                WHERE timestamp = %s;
                """,
                (
                    round(row[column_name], 4) if not pd.isna(row[column_name]) else None,
                    row["timestamp"]
                )
            )

        conn.commit()
        logging.info(f" EMA-{length} updated for table {table_name}")

        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f" Error in calculate_ema_for_table({table_name}, {length}): {e}")


calculate_ema_for_table(nearest_contracts["CE"]["table_1min"], length=5)
calculate_ema_for_table(nearest_contracts["PE"]["table_1min"], length=5)

def calculate_supertrend_channel_for_table(table_name):
    """
    Calculates HL2, ATR, Initial Bands, Supertrend Upper/Lower, Oscillation State,
    Supertrend Pivot, Max/Min Channels, and Supertrend Average for a single OHLC table.
    """

    try:
        conn = connect_to_db()
        if not conn:
            logging.error(" DB connection failed.")
            return

        cur = conn.cursor()

        logging.info(f" Starting Supertrend Channel calculation for table: {table_name}")

        # Step 1: Fetch required OHLC data
        cur.execute(f"""
            SELECT timestamp, high, low, close
            FROM {table_name}
            ORDER BY timestamp;
        """)
        rows = cur.fetchall()

        if not rows:
            logging.warning(f" No OHLC data found in {table_name}. Skipping...")
            return

        df = pd.DataFrame(rows, columns=["timestamp", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Step 2: HL2 Calculation
        df["hl2"] = (df["high"] + df["low"]) / 2

    

        # --- ATR Calculation ---
        ATR_LENGTH = 10
        ATR_MULTIPLIER = 3
        df["true_range"] = 0.0
        df["atr"] = 0.0

        for i in range(1, len(df)):
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            prev_close = df.iloc[i - 1]['close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            df.at[df.index[i], 'true_range'] = tr

        for i in range(len(df)):
            if i == 0:
                df.at[df.index[i], 'atr'] = 0.0
            elif i < ATR_LENGTH:
                df.at[df.index[i], 'atr'] = df['true_range'][:i+1].mean()
            else:
                prev_atr = df.iloc[i-1]['atr']
                tr = df.iloc[i]['true_range']
                df.at[df.index[i], 'atr'] = ((prev_atr * (ATR_LENGTH - 1)) + tr) / ATR_LENGTH

        df['atr'] *= ATR_MULTIPLIER
        df.drop(columns=['true_range'], inplace=True)

        # Initial Upper Band Calculation
        df["initial_upper_bar"] = df["hl2"] + df["atr"]

        # Initial Lower Band Calculation
        df["initial_lower_bar"] = df["hl2"] - df["atr"]

        # Supertrend Upper Band Calculation
        df["supertrend_upper"] = 0.0
        for i in range(len(df)):
            if i == 0:
                df.at[i, "supertrend_upper"] = df.at[i, "initial_upper_bar"]
            else:
                prev_upper = df.at[i - 1, "supertrend_upper"]
                initial_upper = df.at[i, "initial_upper_bar"]
                prev_close = df.at[i - 1, "close"]

                if prev_close < prev_upper:
                    df.at[i, "supertrend_upper"] = min(initial_upper, prev_upper)
                else:
                    df.at[i, "supertrend_upper"] = initial_upper


        #Supertrend Lower Band Calculation
        df["supertrend_lower"] = 0.0
        for i in range(len(df)):
            if i == 0:
                df.at[i, "supertrend_lower"] = df.at[i, "initial_lower_bar"]
            else:
                prev_lower = df.at[i - 1, "supertrend_lower"]
                initial_lower = df.at[i, "initial_lower_bar"]
                prev_close = df.at[i - 1, "close"]

                if prev_close >= prev_lower:
                    df.at[i, "supertrend_lower"] = max(initial_lower, prev_lower)
                else:
                    df.at[i, "supertrend_lower"] = initial_lower

        # --- Oscillation State (os) Calculation ---
        df["os"] = 0  # Default Bearish

        for i in range(len(df)):
            close = df.at[i, "close"]
            upper_band = df.at[i, "supertrend_upper"]
            lower_band = df.at[i, "supertrend_lower"]

            if close > upper_band:
                df.at[i, "os"] = 1  # Bullish
            elif close < lower_band:
                df.at[i, "os"] = 0  # Bearish
            else:
                df.at[i, "os"] = df.at[i - 1, "os"] if i > 0 else 0

        # --- Supertrend Pivot (spt) Calculation ---
        df['spt'] = df.apply(
            lambda row: row['supertrend_lower'] if row['os'] == 1 else row['supertrend_upper'],
            axis=1
        )

        # --- Max Channel Calculation ---
        df['max_channel'] = 0.0

        for i in range(len(df)):
            close = df.at[i, 'close']
            os = df.at[i, 'os']
            spt = df.at[i, 'spt']

            if i == 0:
                df.at[i, 'max_channel'] = close
            else:
                prev_max_channel = df.at[i - 1, 'max_channel']
                prev_os = df.at[i - 1, 'os']

                if close > spt:
                    df.at[i, 'max_channel'] = max(prev_max_channel, close)
                elif os == 1:
                    df.at[i, 'max_channel'] = max(close, prev_max_channel)
                else:
                    df.at[i, 'max_channel'] = min(spt, prev_max_channel)

        # --- Min Channel Calculation ---
        df['min_channel'] = 0.0

        for i in range(len(df)):
            close = df.at[i, 'close']
            os = df.at[i, 'os']
            spt = df.at[i, 'spt']

            if i == 0:
                df.at[i, 'min_channel'] = close
            else:
                prev_min_channel = df.at[i - 1, 'min_channel']
                prev_os = df.at[i - 1, 'os']

                if close < spt:
                    df.at[i, 'min_channel'] = min(prev_min_channel, close)
                elif os == 0:
                    df.at[i, 'min_channel'] = min(close, prev_min_channel)
                else:
                    df.at[i, 'min_channel'] = max(spt, prev_min_channel)
        
        # --- Supertrend Average Channel Calculation ---
        df['supertrend_avg'] = (df['max_channel'] + df['min_channel']) / 2


        df['timestamp'] = df['timestamp'].astype(str)

        update_query = f"""
            UPDATE {table_name}
            SET max_channel = %s,
                min_channel = %s,
                supertrend_avg = %s
            WHERE timestamp = %s;
        """

        for _, row in df.iterrows():
            cur.execute(update_query, (
                row['max_channel'],
                row['min_channel'],
                row['supertrend_avg'],
                row['timestamp']
            ))

        conn.commit()
        logging.info(f" Supertrend Channel (max, min, avg) values updated successfully in {table_name}")

        # Drop unnecessary columns from the table
        cur.execute(f"""
            ALTER TABLE {table_name}
            DROP COLUMN IF EXISTS hl2,
            DROP COLUMN IF EXISTS atr,
            DROP COLUMN IF EXISTS initial_upper_bar,
            DROP COLUMN IF EXISTS initial_lower_bar,
            DROP COLUMN IF EXISTS supertrend_upper,
            DROP COLUMN IF EXISTS supertrend_lower,
            DROP COLUMN IF EXISTS os,
            DROP COLUMN IF EXISTS spt;
        """)
        conn.commit()

        # We'll update DB only once after all calculations
        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f" Error in calculate_supertrend_channel_for_table({table_name}): {e}")



calculate_supertrend_channel_for_table(nearest_contracts['CE']['table_1min'])
calculate_supertrend_channel_for_table(nearest_contracts['PE']['table_1min'])

def calculate_stoch_rsi_for_table(table_name: str, rsi_length=14, stoch_length=14, smooth_k=3, smooth_d=3):
    """
    Calculates the Stochastic RSI %K and %D and updates the given table.
    Adds new columns: stoch_rsi_k, stoch_rsi_d
    """
    try:
        conn = connect_to_db()
        if not conn:
            logging.error(" DB connection failed for Stoch RSI calculation.")
            return

        cur = conn.cursor()

        # Step 1: Ensure required columns exist
        for column in ['stoch_rsi_k', 'stoch_rsi_d']:
            cur.execute(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s;
            """, (table_name, column))
            if not cur.fetchone():
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} FLOAT;")
                logging.info(f" Added missing column: {column} to table {table_name}")

        # Step 2: Fetch close prices
        cur.execute(f"SELECT timestamp, close FROM {table_name} ORDER BY timestamp ASC;")
        rows = cur.fetchall()
        if not rows:
            logging.warning(f" No data found in table {table_name} for Stoch RSI calculation.")
            return

        df = pd.DataFrame(rows, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Step 3: Calculate RSI first
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/rsi_length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_length, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df["rsi"] = rsi

        # Step 4: Calculate Stochastic RSI
        min_rsi = df["rsi"].rolling(window=stoch_length).min()
        max_rsi = df["rsi"].rolling(window=stoch_length).max()

        stoch_rsi = (df["rsi"] - min_rsi) / (max_rsi - min_rsi)
        stoch_rsi_k = stoch_rsi.rolling(window=smooth_k).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d).mean()

        df["stoch_rsi_k"] = stoch_rsi_k * 100
        df["stoch_rsi_d"] = stoch_rsi_d * 100

        # Step 5: Update values into DB
        for _, row in df.iterrows():
            cur.execute(
                f"""
                UPDATE {table_name}
                SET stoch_rsi_k = %s,
                    stoch_rsi_d = %s
                WHERE timestamp = %s;
                """,
                (
                    round(row["stoch_rsi_k"], 4) if not pd.isna(row["stoch_rsi_k"]) else None,
                    round(row["stoch_rsi_d"], 4) if not pd.isna(row["stoch_rsi_d"]) else None,
                    row["timestamp"]
                )
            )

        conn.commit()
        logging.info(f" Stoch RSI (%K and %D) updated for table {table_name}")

        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f" Error in calculate_stoch_rsi_for_table({table_name}): {e}")


calculate_stoch_rsi_for_table(nearest_contracts["CE"]["table_1min"])
calculate_stoch_rsi_for_table(nearest_contracts["PE"]["table_1min"])



def get_1min_table_for_token(token):
    """
    Returns the correct 1-min OHLC table name based on token (CE or PE).
    """
    conn = connect_to_db()
    cur = conn.cursor()

    cur.execute("SELECT ce_token, ce_table_1min, pe_token, pe_table_1min FROM nearest_otm_contracts LIMIT 1;")
    result = cur.fetchone()

    ce_token, ce_table_1min, pe_token, pe_table_1min = result

    cur.close()
    conn.close()

    if token == ce_token:
        return ce_table_1min
    elif token == pe_token:
        return pe_table_1min
    else:
        return None


def get_symbol_from_token(token):
    """
    Returns the symbol (trading symbol) based on token (CE or PE).
    """
    conn = connect_to_db()
    cur = conn.cursor()

    cur.execute("SELECT ce_token, ce_symbol, pe_token, pe_symbol FROM nearest_otm_contracts LIMIT 1;")
    result = cur.fetchone()

    ce_token, ce_symbol, pe_token, pe_symbol = result

    cur.close()
    conn.close()

    if token == ce_token:
        return ce_symbol
    elif token == pe_token:
        return pe_symbol
    else:
        return None

from kiteconnect import KiteConnect

#  Initialize this global variable once before starting infinite loop
last_5min_processed = None


#Setting up live websocket connection, tick by tick handling and aggregation
from kiteconnect import KiteTicker
import logging
import time
import os
import datetime
from collections import defaultdict, deque

#  Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#  Zerodha API Credentials
API_KEY = "8re7mjcm2btaozwf"  # Replace with your API key

#  Fetch access token dynamically from the file
with open("access_token.txt", "r") as f:
    ACCESS_TOKEN = f.read().strip()

#  Dynamically Prepare All Instrument Tokens for Subscription
# From fetched contracts
INSTRUMENT_TOKENS = [256265]  # NIFTY Index Token

# Add all CE + PE contract tokens
INSTRUMENT_TOKENS += [contract['token'] for contract in contracts['ce_contracts'] + contracts['pe_contracts']]

logging.info(f" All Tokens to Subscribe for Live Ticks: {INSTRUMENT_TOKENS}")

#  Initialize KiteTicker WebSocket
kws = KiteTicker(API_KEY, ACCESS_TOKEN)


#  Store Realtime OHLC for Current Candle for Each Symbol
ohlc_data = {}  
# Structure → {symbol: {'timestamp': current, 'open': o, 'high': h, 'low': l, 'close': c}}

#  Kill Existing WebSocket Processes (Optional - Safe Cleanup)
def kill_existing_websockets():
    try:
        os.system("pkill -f kiteconnect")  
        logging.info(" Existing WebSocket instances killed successfully.")
    except Exception as e:
        logging.error(f" Error while killing existing WebSocket processes: {e}")

#  Tick Buffers for 1-Minute OHLC Calculation
tick_buffer = defaultdict(lambda: defaultdict(deque))
# Structure → {symbol: {minute: deque([tick1, tick2, ...])}}

#  Tick Buffers for 5-Minute OHLC Calculation
tick_buffer_5min = defaultdict(lambda: deque())

def get_verified_1min_volume(token, start_time_str):
    """
    Fetches the historical 1-minute candle volume using Kite API for a specific token and timestamp.
    """
    try:
        from_datetime = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
        to_datetime = from_datetime + datetime.timedelta(minutes=1)

        candles = kite.historical_data(
            instrument_token=token,
            from_date=from_datetime,
            to_date=to_datetime,
            interval="minute"
        )

        if candles:
            return candles[0]['volume']
        else:
            logging.warning(f"No historical candle found for token {token} at {start_time_str}")
            return 0
    except Exception as e:
        logging.error(f" Error fetching historical volume for token {token}: {e}")
        return 0
    

#  Process Live OHLC Candles and Handle Nearest OTM Switching
def process_ohlc_candle():
    global current_ce_token, current_pe_token
    global last_5min_processed  #Important to add this

    current_time = datetime.datetime.now()
    current_minute = current_time.strftime("%Y-%m-%d %H:%M")  #"YYYY-MM-DD HH:MM"

    try:
        conn = connect_to_db()
        if not conn:
            logging.error(" Cannot connect to database inside process_ohlc_candle()")
            return

        cur = conn.cursor()

        # Step 2: Process 1-min Candle and Switch Nearest OTM if NEW 1-min window
        if last_5min_processed != current_time.strftime("%Y-%m-%d %H:%M"):
            # New 1-min window detected
            last_5min_processed = current_time.strftime("%Y-%m-%d %H:%M")
            logging.info(f"Detected new 1-minute window at {current_time}")

            previous_minute = (current_time - datetime.timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M")

            for token in [current_ce_token, current_pe_token]:
                if previous_minute not in tick_buffer[token]:
                    continue

                ticks = list(tick_buffer[token][previous_minute])
                if not ticks:
                    continue

                prices = [tick['last_price'] for tick in ticks]

                one_min_entry = {
                    "timestamp": previous_minute,
                    "open": prices[0],
                    "high": max(prices),
                    "low": min(prices),
                    "close": prices[-1],
                    "volume": get_verified_1min_volume(token, previous_minute)
                }

                logging.info(f"1-min OHLC for Token {token}: {one_min_entry}")

                # Determine table name for 1-min data
                table_name = get_1min_table_for_token(token)

                # Insert 1-min OHLC into table
                cur.execute(f"""
                    INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO NOTHING;
                """, (
                    one_min_entry["timestamp"], one_min_entry["open"], one_min_entry["high"],
                    one_min_entry["low"], one_min_entry["close"], one_min_entry["volume"]
                ))
                conn.commit()

                # Recalculate indicators on updated table
                calculate_ema_for_table(table_name, length=5)
                calculate_supertrend_channel_for_table(table_name)
                calculate_stoch_rsi_for_table(table_name)

                # Clear tick buffer for that minute
                del tick_buffer[token][previous_minute]
            
            # Step 3: After processing both tokens → Check nearest OTM contract switching
            update_nearest_otm_contracts()

            # Fetch latest nearest_otm_contracts from database
            conn_check = connect_to_db()
            if conn_check:
                cur_check = conn_check.cursor()
                cur_check.execute("""
                    SELECT ce_token, pe_token
                    FROM nearest_otm_contracts
                    ORDER BY update_timestamp DESC
                    LIMIT 1;
                """)
                new_result = cur_check.fetchone()

                if new_result:
                    new_ce_token, new_pe_token = new_result

                    if new_ce_token != current_ce_token or new_pe_token != current_pe_token:
                        logging.info(" Nearest OTM Contract Changed! Switching...")

                        # Update current CE/PE tokens
                        current_ce_token = new_ce_token
                        current_pe_token = new_pe_token

                        # Create new tables
                        ce_symbol = get_symbol_from_token(current_ce_token)
                        pe_symbol = get_symbol_from_token(current_pe_token)
                        create_nearest_otm_ohlc_tables(ce_symbol, pe_symbol)

                        # Fetch historical into new tables
                        fetch_and_merge_ohlc_for_table(f"{ce_symbol.lower()}_ohlc_1min", current_ce_token, "minute")
                        fetch_and_merge_ohlc_for_table(f"{pe_symbol.lower()}_ohlc_1min", current_pe_token, "minute")

                        # Calculate indicators for fresh tables
                        calculate_ema_for_table(f"{ce_symbol.lower()}_ohlc_1min", length=5)
                        calculate_supertrend_channel_for_table(f"{ce_symbol.lower()}_ohlc_1min")
                        calculate_stoch_rsi_for_table(f"{ce_symbol.lower()}_ohlc_1min")

                        calculate_ema_for_table(f"{pe_symbol.lower()}_ohlc_1min", length=5)
                        calculate_supertrend_channel_for_table(f"{pe_symbol.lower()}_ohlc_1min")
                        calculate_stoch_rsi_for_table(f"{pe_symbol.lower()}_ohlc_1min")

                        logging.info(" New Nearest OTM Switching completed successfully!")

                cur_check.close()
                conn_check.close()

        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f" Error inside process_ohlc_candle(): {e}")
        if conn:
            conn.rollback()
            cur.close()
            conn.close()

#  Handle WebSocket Connection
def on_connect(ws, response):
    logging.info(" WebSocket Connected. Attempting subscription...")

    try:
        time.sleep(1)  # Small delay before subscribing (prevents race condition)
        ws.subscribe(INSTRUMENT_TOKENS)
        logging.info(f" Subscription request sent for: {INSTRUMENT_TOKENS}")

        ws.set_mode(ws.MODE_FULL, INSTRUMENT_TOKENS)
        logging.info(f" Mode set to FULL for: {INSTRUMENT_TOKENS}")

        logging.info(" WebSocket is now actively listening for tick data...")

    except Exception as e:
        logging.error(f" Subscription failed: {e}")

#  Handle Incoming Tick Data & Assign to Correct Minute
def on_ticks(ws, ticks):
    for tick in ticks:
        
        token = tick['instrument_token']

        #  Small improvement: check if token belongs to the tracked INSTRUMENT_TOKENS
        if token not in INSTRUMENT_TOKENS:
            continue  # Ignore ticks for any irrelevant tokens (safety)

        tick_time = tick['exchange_timestamp'].strftime("%Y-%m-%d %H:%M")  # Extract minute part
        tick_buffer[token][tick_time].append(tick)  # Append tick to its respective minute

#  Handle WebSocket Closure & Reconnection
def on_close(ws, code, reason):
    logging.warning(f" WebSocket Closed: {code}, Reason: {reason}")
    logging.info(" Reconnecting in 5 seconds...")
    time.sleep(5)
    ws.connect(reconnect=True)

#  Handle WebSocket Errors
def on_error(ws, code, reason):
    logging.error(f" WebSocket Error Occurred! Code: {code}, Reason: {reason}")

    if "token" in reason.lower():
        logging.error(" Possible access token issue! Fetch a new one and restart.")

#  Handle Reconnection Attempts
def on_reconnect(ws, attempts):
    logging.warning(f" WebSocket Reconnecting... Attempt {attempts}")

#  Assign Event Handlers to WebSocket
kws.on_connect = on_connect
kws.on_ticks = on_ticks
kws.on_close = on_close
kws.on_error = on_error
kws.on_reconnect = on_reconnect

#  Start WebSocket After Killing Any Existing WebSocket Processes
kill_existing_websockets()
logging.info(" Starting WebSocket connection...")
kws.connect(threaded=True)

#  Process OHLC Data Every half-second (Live loop)
while True:
    process_ohlc_candle()
    time.sleep(0.5)
