# 📈 Nifty 50 Options Auto-Trading System (1-Minute Strategy)

This repository implements a **fully automated intraday options trading system** for Nifty 50 using the **Zerodha Kite API**. It scans the **nearest OTM CE and PE contracts**, builds real-time OHLC candles from tick data, calculates indicators, generates alerts, and executes trades automatically.

---

## 🔧 Project Structure


---

## 📌 Features

- 🔁 **Live CE/PE Contract Scanning**: Automatically detects nearest OTM CE and PE from Nifty 50 price.
- 📊 **Live OHLC Candle Construction**: Tick-by-tick aggregation into 1-minute OHLC in PostgreSQL.
- 📈 **Indicator Calculations**:
  - 5 EMA
  - Supertrend Channels (Max, Min, Avg)
  - Stochastic RSI (%K and %D)
- 🚨 **Alert Logic**:
  - Momentum breakout detection via 5EMA slope and Supertrend width.
  - Entry triggers when LTP crosses recent highs with volume confirmation.
- 🤖 **Auto Order Execution** via Kite Connect:
  - Buy Market Orders on triggers
  - Target and SL management with constant monitoring
- 🔄 **Dynamic Contract Switching**: Monitors live prices and switches CE/PE tables as nearest OTM contracts change during the day.

---

## 🧠 System Architecture

1. **Pre-Market (run `t1_main.py`)**
   - Authenticates Kite API.
   - Detects current nearest OTM CE/PE contracts.
   - Creates PostgreSQL tables for OHLC.
   - Fetches historical 1-min OHLC data for CE/PE.
   - Calculates EMA, Supertrend, and StochRSI indicators.

2. **Live Market (run `t1_execute.py`)**
   - Establishes Kite WebSocket for Nifty 50 and CE/PE tokens.
   - Builds 1-min OHLC candles in real-time.
   - Checks for CE/PE alerts and executes trades.
   - Monitors SL/target and places exit orders.
   - Periodically checks for new CE/PE contracts and switches tables.

---

## 💾 Database Schema (PostgreSQL)

### OHLC Table Structure (`<symbol>_ohlc_1min`)
| Column            | Type   | Description                         |
|-------------------|--------|-------------------------------------|
| timestamp         | TIMESTAMP | Candle close timestamp             |
| open              | FLOAT  | Open price                         |
| high              | FLOAT  | High price                         |
| low               | FLOAT  | Low price                          |
| close             | FLOAT  | Close price                        |
| volume            | FLOAT  | Verified 1-min volume              |
| ema_5             | FLOAT  | 5-period EMA                       |
| max_channel       | FLOAT  | Supertrend Max Channel             |
| min_channel       | FLOAT  | Supertrend Min Channel             |
| supertrend_avg    | FLOAT  | Midpoint of max/min channel        |
| stoch_rsi_k       | FLOAT  | Stochastic RSI %K                 |
| stoch_rsi_d       | FLOAT  | Stochastic RSI %D                 |

### Contract Tracking Table (`nearest_otm_contracts`)
Stores the current CE & PE tokens and table references used by the strategy.

---

## 🔑 Setup Instructions

### 1. Prerequisites

- ✅ Zerodha Kite API Account + Credentials
- ✅ PostgreSQL Database (default: `postgres/postgres/admin123`)
- ✅ Python 3.8+
- ✅ Required packages:
```bash
pip install kiteconnect psycopg2 pandas


2. Save Access Token
Visit the login_url printed during the run.

Authorize and paste the request token back when prompted.

The script will fetch and save the access token in access_token.txt.

3. Run Pre-Market Script
bash
Copy
Edit
python t1_main.py
4. Run Live Executor Script
bash
Copy
Edit
python t1_execute.py

🧠 Strategy Logic (CE & PE)
✅ Entry Alert (1-min chart):
Last 3 candles show increasing 5EMA slope.

Latest candle closes above 5EMA.

Stoch RSI K > D and K < 50 (early momentum).

EMA jump ≥ 3% of Supertrend width.

🎯 Trigger:
Live LTP crosses previous candle close (momentum confirmation).

Entry → Market Buy Order.

Target = Entry + Risk.

SL = Current candle low.

📈 Example Output
sql
Copy
Edit
✅ All alert conditions satisfied for 1-min CE strategy.
Live CE LTP: 102.5 | Waiting for trigger above: 101.0
✅ CE Trigger Success! Executing Buy Order at 102.5
✅ CE Target Hit! Exiting at 104.7
🛠 Notes
Adjust the order quantity in the kite.place_order() calls (currently set to 0 as placeholder).

The script handles contract expiry switching and tick buffering efficiently.

Designed for 1-minute scalping. For 5-min strategies, modify the alert logic accordingly.

Ensure system time is synchronized for exact minute-based decisions.

📚 Future Enhancements
Add Telegram/Slack alerts for entry & exit

Add trailing SL / multi-target partial exits

Portfolio PnL monitoring and logging

Live dashboard with candle visualizations

