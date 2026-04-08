import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import time
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOOK_BACK = 60
TICKERS = ["^GSPC"]
THRESHOLD = 0.005
INITIAL_CAPITAL = 10000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        logger.info("Initializing Trading Bot...")
        self.scaler = self.load_scaler()
        self.model = self.load_model()
        self.position = 0
        self.capital = INITIAL_CAPITAL
        self.portfolio_value = INITIAL_CAPITAL
        
    def load_scaler(self):
        with open(f"{DATA_DIR}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Loaded scaler")
        return scaler
    
    def load_model(self):
        model = load_model(f"{MODEL_DIR}/best_model.keras")
        logger.info("Loaded model")
        return model
    
    def fetch_recent_data(self, ticker, days=90):
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = stock.history(start=start_date, end=end_date)
        
        if len(df) < LOOK_BACK:
            logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
            return None
        
        logger.info(f"Fetched {len(df)} days of data for {ticker}")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def prepare_sequence(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        scaled_data = self.scaler.transform(data)
        
        sequence = scaled_data[-LOOK_BACK:]
        return sequence.reshape(1, LOOK_BACK, 5)
    
    def predict_next_price(self, sequence):
        pred_scaled = self.model.predict(sequence, verbose=0)[0, 0]
        
        dummy = np.zeros((1, 5))
        dummy[0, 3] = pred_scaled
        predicted_price = self.scaler.inverse_transform(dummy)[0, 3]
        
        return predicted_price
    
    def get_current_price(self, ticker):
        stock = yf.Ticker(ticker)
        return stock.history(period='1d')['Close'][-1]
    
    def should_buy(self, predicted_price, current_price):
        change_pct = (predicted_price - current_price) / current_price
        return change_pct > THRESHOLD
    
    def should_sell(self, predicted_price, current_price):
        change_pct = (predicted_price - current_price) / current_price
        return change_pct < -THRESHOLD
    
    def execute_buy(self, ticker, current_price):
        if self.capital > 0:
            shares = self.capital / current_price
            self.position = shares
            self.capital = 0
            logger.info(f"BUY {shares:.2f} shares of {ticker} at ${current_price:.2f}")
            return True
        return False
    
    def execute_sell(self, ticker, current_price):
        if self.position > 0:
            self.capital = self.position * current_price
            logger.info(f"SELL {self.position:.2f} shares of {ticker} at ${current_price:.2f}")
            self.position = 0
            return True
        return False
    
    def run_strategy(self, ticker):
        logger.info(f"\n{'='*50}")
        logger.info(f"Running strategy for {ticker}")
        logger.info(f"{'='*50}")
        
        data = self.fetch_recent_data(ticker)
        if data is None:
            logger.error(f"Skipping {ticker} due to insufficient data")
            return
        
        current_price = data['Close'].iloc[-1]
        logger.info(f"Current price: ${current_price:.2f}")
        
        sequence = self.prepare_sequence(data)
        predicted_price = self.predict_next_price(sequence)
        logger.info(f"Predicted next price: ${predicted_price:.2f}")
        
        change_pct = (predicted_price - current_price) / current_price * 100
        logger.info(f"Predicted change: {change_pct:+.2f}%")
        
        if self.should_buy(predicted_price, current_price):
            self.execute_buy(ticker, current_price)
        elif self.should_sell(predicted_price, current_price):
            self.execute_sell(ticker, current_price)
        else:
            logger.info("No action - holding position")
        
        self.portfolio_value = self.capital + (self.position * current_price)
        logger.info(f"Portfolio Value: ${self.portfolio_value:.2f}")
        logger.info(f"  Cash: ${self.capital:.2f}")
        logger.info(f"  Position: {self.position:.2f} shares")
    
    def run_all_tickers(self):
        for ticker in TICKERS:
            try:
                self.run_strategy(ticker)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
            time.sleep(1)
    
    def get_performance_summary(self):
        total_return = ((self.portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        logger.info(f"\n{'='*50}")
        logger.info("PERFORMANCE SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
        logger.info(f"Current Value: ${self.portfolio_value:.2f}")
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info(f"Current Position: {'Long' if self.position > 0 else 'Flat'}")

def main():
    logger.info("Starting LSTM Trading Bot")
    
    bot = TradingBot()
    
    bot.run_all_tickers()
    
    bot.get_performance_summary()
    
    logger.info("Trading bot cycle complete")

if __name__ == "__main__":
    main()