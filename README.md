import time
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime
from binance.client import Client
from binance.enums import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
import logging
import threading
from telegram import Bot

# =========================
# LOAD CONFIGURATION
# =========================
with open('config.json') as f:
    CONFIG = json.load(f)

# =========================
# SETUP LOGGING
# =========================
logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# DATABASE FOR TRADE HISTORY
# =========================
conn = sqlite3.connect('trade_history.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS trades (timestamp TEXT, signal TEXT, price REAL, position_size REAL, sentiment REAL)''')
conn.commit()

# =========================
# TELEGRAM BOT SETUP
# =========================
telegram_bot = Bot(token=CONFIG['telegram_bot_token'])
TELEGRAM_CHAT_ID = CONFIG['telegram_chat_id']

# =========================
# MARKET DATA MODULE
# =========================
class MarketData:
    def __init__(self, client, symbol, timeframes):
        self.client = client
        self.symbol = symbol
        self.timeframes = timeframes
        self.data = {}

    def fetch_historical_data(self, limit=500):
        for tf in self.timeframes:
            klines = self.client.get_klines(symbol=self.symbol, interval=tf, limit=limit)
            df = pd.DataFrame(klines, columns=["open_time","open","high","low","close","volume","close_time",
                                               "quote_asset_volume","number_of_trades","taker_buy_base",
                                               "taker_buy_quote","ignore"])
            df['close'] = df['close'].astype(float)
            self.data[tf] = df
        return self.data

# =========================
# AI SIGNAL MODULE
# =========================
class AISignalGenerator:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()

    def train_model(self, df):
        logging.info("[AI] Training LSTM model...")
        close_prices = df['close'].values.reshape(-1,1)
        scaled = self.scaler.fit_transform(close_prices)
        X, y = [], []
        for i in range(60, len(scaled)):
            X.append(scaled[i-60:i,0])
            y.append(scaled[i,0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=5, batch_size=32)

    def predict_signal(self, latest_data):
        last_60 = latest_data['close'].values[-60:].reshape(-1,1)
        scaled = self.scaler.transform(last_60)
        X = np.reshape(scaled, (1, scaled.shape[0],1))
        pred = self.model.predict(X)[0][0]
        current_price = latest_data['close'].values[-1]
        signal = "BUY" if pred > current_price else "SELL"
        logging.info(f"[AI] Predicted price: {pred}, Current price: {current_price}, Signal: {signal}")
        return signal

# =========================
# STRATEGIES & NEWS SENTIMENT MODULES
# =========================
class Strategy:
    @staticmethod
    def sma_crossover(df):
        df['SMA10'] = df['close'].rolling(10).mean()
        df['SMA50'] = df['close'].rolling(50).mean()
        if df['SMA10'].iloc[-1] > df['SMA50'].iloc[-1]: return "BUY"
        elif df['SMA10'].iloc[-1] < df['SMA50'].iloc[-1]: return "SELL"
        else: return "HOLD"

    @staticmethod
    def rsi_strategy(df, period=14):
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -1*delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        if rsi.iloc[-1] < 30: return "BUY"
        elif rsi.iloc[-1] > 70: return "SELL"
        else: return "HOLD"

    @staticmethod
    def macd_strategy(df):
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        if macd.iloc[-1] > signal_line.iloc[-1]: return "BUY"
        elif macd.iloc[-1] < signal_line.iloc[-1]: return "SELL"
        else: return "HOLD"

class NewsSentiment:
    @staticmethod
    def get_sentiment():
        sentiment = np.random.uniform(-1,1)
        logging.info(f"[News] Sentiment score: {sentiment}")
        return sentiment

# =========================
# RISK MANAGEMENT MODULE
# =========================
class RiskManager:
    def __init__(self, capital, risk_percent):
        self.capital = capital
        self.risk_percent = risk_percent

    def calculate_position_size(self, price, stop_loss, signal_strength=1.0):
        risk_amount = self.capital * (self.risk_percent / 100) * signal_strength
        position_size = risk_amount / abs(price - stop_loss)
        logging.info(f"[Risk] Position size: {position_size}")
        return position_size

# =========================
# TRADE EXECUTION MODULE
# =========================
class TradeExecutor:
    def __init__(self, client, config, telegram_bot):
        self.client = client
        self.paper_trading = config['paper_trading']
        self.telegram_bot = telegram_bot

    def execute_trade(self, signal, size, symbol, side_type="LONG"):
        message = f"{side_type} {signal} {size} units of {symbol}"
        if self.paper_trading:
            logging.info(f"[Paper Trading] {message}")
        else:
            logging.info(f"[Live Trading] {message}")
            # Execute live order here
        # Send Telegram alert
        try:
            self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            logging.error(f"[Telegram Error] {e}")

# =========================
# DASHBOARD MODULE
# =========================
class Dashboard:
    @staticmethod
    def display(signal, capital, position_size, sentiment):
        print(f"[Dashboard] Signal: {signal}, Capital: {capital}, Position Size: {position_size}, Sentiment: {sentiment}")
        logging.info(f"[Dashboard] Signal: {signal}, Capital: {capital}, Position Size: {position_size}, Sentiment: {sentiment}")

# =========================
# MAIN BOT
# =========================
class AITradingBot:
    def __init__(self, config):
        self.config = config
        self.client = Client(config['api_key'], config['api_secret'])
        self.data_module = MarketData(self.client, config['symbol'], config['timeframes'])
        self.ai_module = AISignalGenerator()
        self.risk_module = RiskManager(10000, config['risk_percent'])
        self.executor = TradeExecutor(self.client, config, telegram_bot)
        self.dashboard = Dashboard()

    def combine_signals(self, signals):
        final_score = 0
        for strat, sig in signals.items():
            weight = self.config['strategy_weights'][strat]
            if sig == "BUY": final_score += weight
            elif sig == "SELL": final_score -= weight
        if final_score > 0.1: return "BUY"
        elif final_score < -0.1: return "SELL"
        else: return "HOLD"

    def retrain_ai_periodically(self, interval=3600):
        while True:
            latest_data = self.data_module.data[self.config['timeframes'][0]]
            self.ai_module.train_model(latest_data)
            logging.info("[AI] Model retrained")
            time.sleep(interval)

    def run(self):
        logging.info("[Bot] Starting AI Trading Bot with Telegram alerts...")
        historical_data = self.data_module.fetch_historical_data()
        self.ai_module.train_model(historical_data[self.config['timeframes'][0]])
        threading.Thread(target=self.retrain_ai_periodically, daemon=True).start()

        while True:
            try:
                latest_data = self.data_module.data[self.config['timeframes'][0]]
                signals = {
                    "AI": self.ai_module.predict_signal(latest_data),
                    "SMA": Strategy.sma_crossover(latest_data),
                    "RSI": Strategy.rsi_strategy(latest_data),
                    "MACD": Strategy.macd_strategy(latest_data)
                }
                sentiment = NewsSentiment.get_sentiment()
                final_signal = self.combine_signals(signals)
                price = latest_data['close'].iloc[-1]
                stop_loss = price * 0.98
                position_size = self.risk_module.calculate_position_size(price, stop_loss, signal_strength=max(0.5, sentiment))
                
                self.executor.execute_trade(final_signal, position_size, self.config['symbol'])
                self.dashboard.display(final_signal, self.risk_module.capital, position_size, sentiment)

                # Save trade to database
                c.execute('INSERT INTO trades VALUES (?,?,?,?,?)', (datetime.now(), final_signal, price, position_size, sentiment))
                conn.commit()

                time.sleep(60)
            except Exception as e:
                logging.error(f"[Error] {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = AITradingBot(CONFIG)
    bot.run()
