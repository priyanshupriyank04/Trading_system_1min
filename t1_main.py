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



    if today <= datetime.date(today.year, 6, 5):
        return datetime.date(today.year, 6, 5)  # Weekly expiry
    elif today <= datetime.date(today.year, 6, 12):
        return datetime.date(today.year, 6, 12)  # Weekly expiry
    elif today <= datetime.date(today.year, 6, 19):
        return datetime.date(today.year, 6, 19)  # Monthly expiry
    elif today <= datetime.date(today.year, 6, 26):
        return datetime.date(today.year, 6, 26)  # Weekly expiry
    elif today <= datetime.date(today.year, 7, 3):
        return datetime.date(today.year, 7, 3)  # Weekly expiry
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
                    ema_22 FLOAT,
                    ema_33 FLOAT,
                    hl2 FLOAT,
                    atr FLOAT,
                    initial_upper_bar FLOAT,
                    initial_lower_bar FLOAT,
                    supertrend_upper FLOAT,
                    supertrend_lower FLOAT,
                    os FLOAT,
                    spt FLOAT,
                    max_channel FLOAT,
                    min_channel FLOAT,
                    supertrend_avg FLOAT,
                    adx FLOAT,
                    di_plus FLOAT,
                    di_minus FLOAT

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
                    ema_22 FLOAT,
                    ema_33 FLOAT,
                    hl2 FLOAT,
                    atr FLOAT,
                    initial_upper_bar FLOAT,
                    initial_lower_bar FLOAT,
                    supertrend_upper FLOAT,
                    supertrend_lower FLOAT,
                    os FLOAT,
                    spt FLOAT,
                    max_channel FLOAT,
                    min_channel FLOAT,
                    supertrend_avg FLOAT,
                    adx FLOAT,
                    di_plus FLOAT,
                    di_minus FLOAT
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

