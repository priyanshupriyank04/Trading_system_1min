#  Step 1: Import Required Libraries
import os                # For environment variables and file handling
import time              # For adding delays where needed
import datetime          # To handle timestamps
import pandas as pd      # For working with dataframes
import psycopg2          # PostgreSQL database connection
import logging           # For structured logging
from kiteconnect import KiteConnect  # Zerodha API connection
import math             # for using math functions 



#  Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logging.info(" Required libraries imported successfully.")

#  Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#  API Credentials
API_KEY = "8re7mjcm2btaozwf"  #  Replace with your actual API key
API_SECRET = "fw8gm7wfeclcic9rlkp0tbzx4h2ss2n1"  #  Replace with your actual API secret
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
                logging.info(f" API Authentication Successful! User: {profile['user_name']}")
                return access_token  #  Return the valid token
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

        #  Step 4: Save the new access token
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

import psycopg2

#  Database Configuration
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "admin123"
DB_HOST = "localhost"
DB_PORT = "5432"

def connect_to_db():
    try:
        logging.info(f"Connecting to database with:\nDB_NAME={DB_NAME}\nDB_USER={DB_USER}\nDB_PASSWORD={DB_PASSWORD}\nDB_HOST={DB_HOST}\nDB_PORT={DB_PORT}")
        
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True  #  Enable autocommit mode
        logging.info("Successfully connected to PostgreSQL!")
        return conn
    except Exception as e:
        logging.error(f" Failed to connect to database: {e}")
        return None

#  Establish Database Connection
conn = connect_to_db()
if conn:
    cur = conn.cursor()
else:
    logging.error("Database connection failed, exiting...")
    exit(1)

#  Stores the latest 1-min OHLC data for NIFTY 50, CE & PE
latest_1min_data = {
    "NIFTY": {
        "timestamp": None,
        "1min_high": None,
        "1min_low": None,
        "1min_min_channel": None,
        "1min_max_channel": None,
        "1min_5wma": None,
        "1min_supertrend_avg": None
    },
    "CE": {
        "symbol": None,
        "token": None,
        "timestamp": None,
        "1min_high": None,
        "1min_low": None,
        "1min_min_channel": None,
        "1min_max_channel": None,
        "1min_5wma": None,
        "1min_supertrend_avg": None
    },
    "PE": {
        "symbol": None,
        "token": None,
        "timestamp": None,
        "1min_high": None,
        "1min_low": None,
        "1min_min_channel": None,
        "1min_max_channel": None,
        "1min_5wma": None,
        "1min_supertrend_avg": None
    }
}


#  Keeps track of active alerts
active_alert = {
    "type": None,  # "CE" or "PE"
    "timestamp": None  # Stores the timestamp when alert was triggered
}

#  Stores the current trade state
trade_state = {
    "active": False,
    "entry_price": None,
    "exit_price": None,
    "trade_type": None,  # "CE" or "PE"
    "quantity": None,
    "stop_loss": None,
    "target": None
}

#  Establish Database Connection
conn = connect_to_db()
cur = conn.cursor()

#  Variable to Store Last Logged Time (Prevents Log Flooding)
last_logged_time = None  


import time

#Fetch nifty price 
def get_nifty50_price():
    """
    Fetches the real-time Nifty 50 index price with retry logic.
    """
    retries = 5
    for attempt in range(retries):
        try:
            nifty_data = kite.ltp("NSE:NIFTY 50")
            nifty_price = nifty_data["NSE:NIFTY 50"]["last_price"]
            logging.info(f" Fetched Nifty 50 Index Price: {nifty_price}")
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
        logging.info(f" Fetching LTP for token: {option_token}")
        
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

# Find the nearest OTM CE contract based on the nifty index price
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

        otm_strike = int(math.ceil(nifty_index_price / 50) * 50)
        best_ce = min(ce_options, key=lambda x: abs(x["strike"] - otm_strike))

        ltp = get_nifty50_option_price(best_ce["instrument_token"])
        logging.info(f" CE OTM Contract: {best_ce['tradingsymbol']} | Token: {best_ce['instrument_token']} | 💰 LTP: {ltp}")
        return best_ce["tradingsymbol"], best_ce["instrument_token"]

    except Exception as e:
        logging.error(f" Error in get_nearest_otm_ce_contract: {e}")
        return None, None

# Find the nearest OTM PE contract based on the nifty index price
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

        otm_strike = int(math.floor(nifty_index_price / 50) * 50 )
        best_pe = min(pe_options, key=lambda x: abs(x["strike"] - otm_strike))

        ltp = get_nifty50_option_price(best_pe["instrument_token"])
        logging.info(f" PE OTM Contract: {best_pe['tradingsymbol']} | Token: {best_pe['instrument_token']} | 💰 LTP: {ltp}")
        return best_pe["tradingsymbol"], best_pe["instrument_token"]

    except Exception as e:
        logging.error(f" Error in get_nearest_otm_pe_contract: {e}")
        return None, None
    

# Fetch the nearest OTM CE/PE contract table names
def get_nearest_otm_ce_pe_tables(nifty_price):
    """
    Fetch nearest OTM CE & PE contracts based on latest NIFTY price
    and return their respective table names for OHLC 1min.
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

#  Fetch latest nifty price
nifty_price = get_nifty50_price()

#  Get nearest OTM CE & PE contract details and respective ohlc table names
nearest_contracts = get_nearest_otm_ce_pe_tables(nifty_price)

#  Print Output
if nearest_contracts:
    print("Nearest OTM CE Contract Details:")
    print(nearest_contracts["CE"])

    print("\nNearest OTM PE Contract Details:")
    print(nearest_contracts["PE"])
else:
    print("Failed to fetch nearest OTM CE/PE contracts.")

#  Fetch CE contracts in range nifty-500 to nifty+500 (step of 50)
def get_ce_contracts(nifty_index_price):
    """
    Fetches all CE contracts in the range of NIFTY_INDEX +/- 500 with step of 50
    Returns list of dicts with symbol and instrument_token
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

#  Fetch PE contracts in range nifty-500 to nifty+500 (step of 50)
def get_pe_contracts(nifty_index_price):
    """
    Fetches all PE contracts in the range of NIFTY_INDEX +/- 500 with step of 50
    Returns list of dicts with symbol and instrument_token
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

# Fetch All CE and PE contracts together
def fetch_contracts(nifty_index_price):
    """
    Fetches all CE and PE contracts in the range of NIFTY_INDEX +/- 500 with step of 50
    Returns a dictionary with lists of CE and PE contracts
    """
    return {
        "ce_contracts": get_ce_contracts(nifty_index_price),
        "pe_contracts": get_pe_contracts(nifty_index_price)
    }

import csv
import datetime

def log_trade_to_csv(trade_type, contract_symbol):
    """
    Appends trade trigger info to trades.csv with timestamp, trade type (CE/PE), and contract symbol.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, trade_type, contract_symbol]

    file_path = "trades.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "trade_type", "contract_symbol"])
        writer.writerow(row)



#  Get live Nifty 50 Index Price
nifty_price = get_nifty50_price()

#  Fetch all CE & PE contracts around that price
contracts = fetch_contracts(nifty_price)

#  Extract instrument tokens to subscribe via WebSocket
all_tokens = [contract['token'] for contract in contracts['ce_contracts'] + contracts['pe_contracts']]

#  Get live Nifty 50 Index Price
nifty_price = get_nifty50_price()

#  Fetch all CE & PE contracts around that price
contracts = fetch_contracts(nifty_price)

#  Extract instrument tokens to subscribe via WebSocket
all_tokens = [contract['token'] for contract in contracts['ce_contracts'] + contracts['pe_contracts']]

#  WebSocket for Option Contract Strategy Execution
from kiteconnect import KiteTicker
import logging
import time
import os

#  Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#  Load Access Token from file
with open("access_token.txt", "r") as f:
    ACCESS_TOKEN = f.read().strip()

#  Define Index Token
NIFTY_50_TOKEN = 256265  # Nifty 50 Index Token

#  Initialize KiteTicker WebSocket
strategy_ws = KiteTicker(API_KEY, ACCESS_TOKEN)

#  Store Latest Prices for Monitoring
latest_nifty_price = None
latest_ce_price = None
latest_pe_price = None
ce_token = None
pe_token = None

#  Combine all tokens to subscribe (Nifty 50 + Option contracts)
tokens_to_subscribe = [NIFTY_50_TOKEN] + all_tokens
logging.info(f" Total tokens to subscribe via WebSocket: {len(tokens_to_subscribe)}")


#  WebSocket Event Handlers for Option Strategy Execution
def on_connect(ws, response):
    """Handles WebSocket connection."""
    logging.info("WebSocket Connected. Subscribing to all tokens...")

    try:
        time.sleep(1)  # Optional: short delay before subscribing
        ws.subscribe(tokens_to_subscribe)
        ws.set_mode(ws.MODE_FULL, tokens_to_subscribe)  # Tick-by-tick with full market depth
        logging.info(f" Subscribed to {len(tokens_to_subscribe)} tokens for strategy execution.")
    except Exception as e:
        logging.error(f" Subscription failed: {e}")

def on_ticks(ws, ticks):
    """Handles incoming tick data for live CE/PE trade execution."""
    global latest_nifty_price, latest_ce_price, latest_pe_price
    global ce_token, pe_token
    for tick in ticks:
        token = tick['instrument_token']
        ltp = tick.get("last_price")

        if token == NIFTY_50_TOKEN:
            latest_nifty_price = ltp
            # logging.info(f" Live Nifty 50 Price: {ltp}")

        elif token == ce_token:
            latest_ce_price = ltp
            #logging.info(f" Live CE Price: {ltp}")

        elif token == pe_token:
            latest_pe_price = ltp
            #logging.info(f" Live PE Price: {ltp}")

        

def on_close(ws, code, reason):
    """Handles WebSocket closure and restarts connection."""
    logging.warning(f" WebSocket Closed: Code={code}, Reason={reason}")
    logging.info(" Reconnecting WebSocket in 5 seconds...")
    time.sleep(5)
    ws.connect(threaded=True, reconnect=True)

def on_error(ws, code, reason):
    """Handles WebSocket error events."""
    logging.error(f" WebSocket Error: Code={code}, Reason={reason}")
    if "token" in reason.lower():
        logging.error(" Access token issue detected! Please refresh and restart.")

def on_reconnect(ws, attempts):
    """Handles reconnection attempts."""
    logging.warning(f" WebSocket Reconnecting... Attempt #{attempts}")

#  Assign Handlers to strategy WebSocket
strategy_ws.on_connect = on_connect
strategy_ws.on_ticks = on_ticks
strategy_ws.on_close = on_close
strategy_ws.on_error = on_error
strategy_ws.on_reconnect = on_reconnect

#  Launch WebSocket (non-blocking)
logging.info(" Launching WebSocket for Option Strategy Execution...")
strategy_ws.connect(threaded=True)

#  Infinite Loop to Continuously Check for New 1-Minute Data
while True:
    # Step 1: Get the current time's minute value
    now = datetime.datetime.now()
    curtime = now.minute  # Extracting the minute value

    # Step 2: Compute reqd_time as the largest multiple of 1 that is LESS THAN curtime
    if curtime % 1 == 0:
        time.sleep(1)
        reqd_time = (curtime - 1) % 60  # If curtime is a multiple of 1, take the previous minute
    else:
        reqd_time = (curtime // 1) * 1  # This else will never hit, but structure retained

    # Step 3: Fetch the latest CE/PE 1min tables and update timestamp from nearest_otm_contracts
    cur.execute("""
        SELECT ce_table_1min, pe_table_1min, update_timestamp,
               ce_symbol, ce_token, pe_symbol, pe_token
        FROM nearest_otm_contracts
        ORDER BY update_timestamp DESC
        LIMIT 1;
    """)

    result = cur.fetchone()

    if result:
        ce_table, pe_table, update_timestamp, ce_symbol, ce_token, pe_symbol, pe_token = result
        latest_1min_data["CE"]["symbol"] = ce_symbol
        latest_1min_data["CE"]["token"] = ce_token
        latest_1min_data["PE"]["symbol"] = pe_symbol
        latest_1min_data["PE"]["token"] = pe_token

        # Step 4: Fetch the last row timestamp from CE 1min table
        cur.execute(f"SELECT timestamp FROM {ce_table} ORDER BY timestamp DESC LIMIT 1;")
        last_row = cur.fetchone()

        if last_row:
            last_timestamp = last_row[0]
            fetched_time = last_timestamp.minute  # Extract the minute part from the timestamp
        else:
            if last_logged_time != curtime:  # Prevent repeated logs
                last_logged_time = curtime
            time.sleep(1)
            continue

    else:
        if last_logged_time != curtime:
            last_logged_time = curtime
        time.sleep(1)
        continue

    
    #Step 5: Compare fetched_time with reqd_time
    if fetched_time != reqd_time:
        time.sleep(1)
        continue

    
    # Step 6: If new data is available, start fetching complete 1-min candle data
    if last_logged_time != reqd_time:  # Log only when a new candle appears
        last_logged_time = reqd_time  # Update last_logged_time

        time.sleep(5)  # Slight wait to ensure DB update is complete

        # CE: Fetch latest 8 complete 1-min candles
        cur.execute(f"""
            SELECT timestamp, open, high, low, close, volume, ema_5, 
                max_channel, min_channel, supertrend_avg,
                stoch_rsi_k, stoch_rsi_d
            FROM {ce_table}
            WHERE ema_5 IS NOT NULL AND max_channel IS NOT NULL AND 
                min_channel IS NOT NULL AND supertrend_avg IS NOT NULL AND 
                stoch_rsi_k IS NOT NULL AND stoch_rsi_d IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 8;
        """)
        ce_rows = cur.fetchall()

        if len(ce_rows) < 8:
            logging.warning(" Not enough CE 1-min candles to evaluate alert condition.")
            time.sleep(1)
            continue

        ce_rows.reverse()

        ce_candles = []
        for row in ce_rows:
            ce_candles.append({
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "ema_5": row[6],
                "max_channel": row[7],
                "min_channel": row[8],
                "supertrend_avg": row[9],
                "stoch_rsi_k": row[10],
                "stoch_rsi_d": row[11],
            })

        ce_c7 = ce_candles[-8] #oldest candle 
        ce_c6 = ce_candles[-7]
        ce_c5 = ce_candles[-6]
        ce_c4 = ce_candles[-5]
        ce_c3 = ce_candles[-4]  
        ce_c2 = ce_candles[-3]
        ce_c1 = ce_candles[-2]
        ce_c0 = ce_candles[-1]  # Latest candle

        ce_alert = False

        ema_drops = 0
        ema_list = [ce_c7, ce_c6, ce_c5, ce_c4, ce_c3, ce_c2, ce_c1]
        for i in range(len(ema_list) - 1):
            if ema_list[i]["ema_5"] - ema_list[i + 1]["ema_5"] >= 0.7:
                ema_drops += 1



        if (
            ce_c0["ema_5"] > abs(ce_c1["ema_5"]+ 0.7) and  # 5ema(0) > 5ema(-1)
            ce_c0["stoch_rsi_k"] >= abs(ce_c0["stoch_rsi_d"] + 4) and  # K > D
            ce_c0["stoch_rsi_k"] < 55 and  # K < 55 means D will also be less than 55
            ema_drops >= 3 and  # minimum 3 0.7pts ema drops in the 7 candle window condition
            ce_c0["ema_5"] < ce_c5["ema_5"] and  # 5ema(0) < 5ema(-5)
            ce_c0["max_channel"] >= ce_c0["close"] + (ce_c0["close"] - ce_c0["low"])  # atleast close - low gap available up till before supertrend upper channel
        ):
            logging.info("✅ All alert conditions satisfied for 1-min CE strategy. Proceeding to trigger condition.")

            try:
                ce_ltp_data = kite.ltp(ce_symbol)
                ce_ltp = ce_ltp_data.get(ce_symbol, {}).get("last_price")

                if ce_ltp is None:
                    logging.warning(" Unable to fetch CE LTP. Skipping alert.")
                else:
                    risk = abs(ce_c0["close"] - ce_c0["low"])
                    t1_target = round(ce_ltp + risk, 2)
                    t2_target = round(ce_ltp + 2 * risk, 2)

                    ce_alert = True
                    active_alert["timestamp"] = ce_c0["timestamp"]
                    active_alert["trigger_high"] = ce_c0["close"]
                    active_alert["stop_loss"] = ce_c0["low"]
                    active_alert["target"] = t1_target

                    trigger_monitoring_start = datetime.datetime.now()

                    logging.info(f" CE LTP: {ce_ltp}, T1: {t1_target}, T2: {t2_target}")

            except Exception as e:
                logging.error(f" Error fetching CE LTP or calculating targets: {e}")

        else:
            logging.info("❌ CE alert condition not met for 1-min strategy. Skipping.")


        if ce_alert:
            logging.info("✅ CE Alert Condition Satisfied (on OTM CE) for 1-min strategy! Executing CE Buy Order immediately...")

            try:
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NFO,
                    tradingsymbol=ce_symbol,
                    transaction_type=kite.TRANSACTION_TYPE_BUY,
                    quantity=0,  # Replace with actual quantity
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_MIS
                )

                logging.info(f" CE Buy Order Placed: {ce_symbol} | Order ID: {order_id}")
                log_trade_to_csv("CE", ce_symbol)

                entry_price = ce_ltp  # LTP already fetched earlier
                stop_loss = active_alert["stop_loss"]
                risk = abs(entry_price - stop_loss)
                target_price = round(entry_price + risk, 2)

                trade_active_ce = True
                logging.info(f" CE SL: {stop_loss}, Target: {target_price}")

                # Start trade monitoring
                while trade_active_ce:
                    try:
                        ce_ltp_data = kite.ltp(ce_symbol)
                        latest_ce_ltp = ce_ltp_data.get(ce_symbol, {}).get("last_price")

                        if latest_ce_ltp is None:
                            logging.warning(" Failed to fetch CE LTP during trade. Retrying...")
                            time.sleep(1)
                            continue

                        logging.info(f" CE LTP in Trade: {latest_ce_ltp}")

                        if latest_ce_ltp <= stop_loss:
                            logging.info(f"❌ SL Hit on CE trade! Exiting at CMP ({latest_ce_ltp})")
                            kite.place_order(
                                variety=kite.VARIETY_REGULAR,
                                exchange=kite.EXCHANGE_NFO,
                                tradingsymbol=ce_symbol,
                                transaction_type=kite.TRANSACTION_TYPE_SELL,
                                quantity=0,  # Replace with actual quantity
                                order_type=kite.ORDER_TYPE_MARKET,
                                product=kite.PRODUCT_MIS
                            )
                            trade_active_ce = False
                            break

                        if latest_ce_ltp >= target_price:
                            logging.info(f"✅ CE Target Hit! Booking at CMP ({latest_ce_ltp})")
                            kite.place_order(
                                variety=kite.VARIETY_REGULAR,
                                exchange=kite.EXCHANGE_NFO,
                                tradingsymbol=ce_symbol,
                                transaction_type=kite.TRANSACTION_TYPE_SELL,
                                quantity=0,  # Replace with actual quantity
                                order_type=kite.ORDER_TYPE_MARKET,
                                product=kite.PRODUCT_MIS
                            )
                            trade_active_ce = False
                            break

                        time.sleep(1)

                    except Exception as e:
                        logging.error(f" Error while monitoring CE trade: {e}")
                        time.sleep(1)

            except Exception as e:
                logging.error(f" CE Buy Order Failed: {e}")
        else:
            logging.info("❌ CE Alert NOT satisfied for 1-min strategy. Skipping.")



        #  PE: Fetch latest 8 complete 1-min candles
        cur.execute(f"""
            SELECT timestamp, open, high, low, close, volume, ema_5,
                max_channel, min_channel, supertrend_avg,
                stoch_rsi_k, stoch_rsi_d
            FROM {pe_table}
            WHERE ema_5 IS NOT NULL AND max_channel IS NOT NULL AND min_channel IS NOT NULL AND supertrend_avg IS NOT NULL
                AND stoch_rsi_k IS NOT NULL AND stoch_rsi_d IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 8;
        """)
        pe_rows = cur.fetchall()

        if len(pe_rows) < 8:
            logging.warning(" Not enough PE 1-min candles to evaluate alert condition.")
            time.sleep(1)
            continue

        pe_rows.reverse()

        pe_candles = []
        for row in pe_rows:
            pe_candles.append({
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "ema_5": row[6],
                "max_channel": row[7],
                "min_channel": row[8],
                "supertrend_avg": row[9],
                "stoch_rsi_k": row[10],
                "stoch_rsi_d": row[11]
            })



        pe_c7 = pe_candles[-8]
        pe_c6 = pe_candles[-7]
        pe_c5 = pe_candles[-6]  # Oldest
        pe_c4 = pe_candles[-5]
        pe_c3 = pe_candles[-4]
        pe_c2 = pe_candles[-3]
        pe_c1 = pe_candles[-2]
        pe_c0 = pe_candles[-1]  # Latest candle

        pe_alert = False
        ema_drops = 0
        ema_list = [pe_c7, pe_c6, pe_c5, pe_c4, pe_c3, pe_c2, pe_c1]
        for i in range(len(ema_list) - 1):
            if ema_list[i]["ema_5"] - ema_list[i + 1]["ema_5"] >= 0.7:
                ema_drops += 1


        if (
            pe_c0["ema_5"] > abs(pe_c1["ema_5"]+0.7) and  # Latest 5 EMA > previous
            pe_c0["stoch_rsi_k"] >= abs(pe_c0["stoch_rsi_d"] + 4) and  # K >= D + 4
            pe_c0["stoch_rsi_k"] < 55 and # K < 55 means D will also be greater than 55
            ema_drops >= 3 and  # min 3 0.7 pts drop in the ema for 7 candle window
            pe_c0["ema_5"] < pe_c5["ema_5"] and  # 5ema(0)<5ema(-5)
            pe_c0["max_channel"] >= pe_c0["close"] + (pe_c0["close"] - pe_c0["low"]) # close - low gap up till before supertrend upper channel 
        ):
            logging.info("✅ All alert conditions satisfied for 1-min PE strategy. Proceeding to trigger condition.")

            try:
                pe_ltp_data = kite.ltp(pe_symbol)
                pe_ltp = pe_ltp_data.get(pe_symbol, {}).get("last_price")

                if pe_ltp is None:
                    logging.warning(" Unable to fetch PE LTP. Skipping alert.")
                else:
                    risk = abs(pe_c0["close"] - pe_c0["low"])
                    t1_target = round(pe_ltp + risk, 2)
                    t2_target = round(pe_ltp + 2 * risk, 2)

                    pe_alert = True
                    active_alert["timestamp"] = pe_c0["timestamp"]
                    active_alert["trigger_high"] = pe_c0["close"]
                    active_alert["stop_loss"] = pe_c0["low"]
                    active_alert["target"] = t1_target

                    trigger_monitoring_start = datetime.datetime.now()

                    logging.info(f" PE LTP: {pe_ltp}, T1: {t1_target}, T2: {t2_target}")

            except Exception as e:
                logging.error(f" Error fetching PE LTP or calculating targets: {e}")

        else:
            logging.info("❌ PE alert condition not met for 1-min strategy. Skipping.")

        
        if pe_alert:
            logging.info("✅ PE Alert Condition Satisfied (on OTM PE) for 1-min strategy! Executing PE Buy Order immediately...")

            try:
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NFO,
                    tradingsymbol=pe_symbol,
                    transaction_type=kite.TRANSACTION_TYPE_BUY,
                    quantity=0,  # Replace with actual quantity
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_MIS
                )

                logging.info(f" PE Buy Order Placed: {pe_symbol} | Order ID: {order_id}")
                log_trade_to_csv("PE", pe_symbol)  # Save to trades.csv

                entry_price = pe_ltp  # LTP already fetched during alert
                stop_loss = active_alert["stop_loss"]
                risk = abs(entry_price - stop_loss)
                target_price = round(entry_price + risk, 2)

                trade_active_pe = True
                logging.info(f" PE SL: {stop_loss}, Target: {target_price}")

                # Trade Monitoring Loop
                while trade_active_pe:
                    try:
                        pe_ltp_data = kite.ltp(pe_symbol)
                        latest_pe_ltp = pe_ltp_data.get(pe_symbol, {}).get("last_price")

                        if latest_pe_ltp is None:
                            logging.warning(" Failed to fetch PE LTP during trade. Retrying...")
                            time.sleep(1)
                            continue

                        logging.info(f" PE LTP in Trade: {latest_pe_ltp}")

                        if latest_pe_ltp <= stop_loss:
                            logging.info(f"❌ SL Hit on PE trade! Exiting at CMP ({latest_pe_ltp})")
                            kite.place_order(
                                variety=kite.VARIETY_REGULAR,
                                exchange=kite.EXCHANGE_NFO,
                                tradingsymbol=pe_symbol,
                                transaction_type=kite.TRANSACTION_TYPE_SELL,
                                quantity=0,  # Replace with actual quantity
                                order_type=kite.ORDER_TYPE_MARKET,
                                product=kite.PRODUCT_MIS
                            )
                            trade_active_pe = False
                            break

                        if latest_pe_ltp >= target_price:
                            logging.info(f"✅ PE Target Hit! Booking at CMP ({latest_pe_ltp})")
                            kite.place_order(
                                variety=kite.VARIETY_REGULAR,
                                exchange=kite.EXCHANGE_NFO,
                                tradingsymbol=pe_symbol,
                                transaction_type=kite.TRANSACTION_TYPE_SELL,
                                quantity=0,  # Replace with actual quantity
                                order_type=kite.ORDER_TYPE_MARKET,
                                product=kite.PRODUCT_MIS
                            )
                            trade_active_pe = False
                            break

                        time.sleep(1)

                    except Exception as e:
                        logging.error(f" Error while monitoring PE trade: {e}")
                        time.sleep(1)

            except Exception as e:
                logging.error(f" PE Buy Order Failed: {e}")
        else:
            logging.info("❌ PE Alert NOT satisfied for 1-min strategy. Skipping PE trigger monitoring.")

