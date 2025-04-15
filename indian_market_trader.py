import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import configparser
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("indian_market_trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IndianMarketTrader")

class HDFCSecuritiesAPI:
    """Integration with HDFC Securities API"""
    
    def __init__(self, user_id, password,  api_key=None):
        self.user_id = user_id
        self.password = password
        # self.pin = pin
        self.api_key = api_key
        self.session_token = None
        self.base_url = "https://api-uat.hdfcbank.com/API/SMS_Banking"  # Example URL, replace with actual HDFC API URL
        self.session = requests.Session()
        self.headers = {}
    
    def login(self):
        """Log in to HDFC Securities"""
        try:
            # This is a placeholder - you'll need to implement the actual login logic 
            # based on HDFC Securities API documentation
            payload = {
                "user_id": self.user_id,
                "password": self.password,
                # "pin": self.pin,
                "api_key": self.api_key
            }
            
            response = self.session.post(f"{self.base_url}/login", json=payload)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success":
                self.session_token = data.get("session_token")
                self.headers = {
                    "Authorization": f"Bearer {self.session_token}",
                    "Content-Type": "application/json"
                }
                logger.info("Successfully logged in to HDFC Securities")
                return True
            else:
                logger.error(f"Login failed: {data.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"Error during login: {e}")
            return False
    
    def get_profile(self):
        """Get user profile information"""
        try:
            response = self.session.get(f"{self.base_url}/profile", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting profile: {e}")
            return None
    
    def get_funds(self):
        """Get available funds"""
        try:
            response = self.session.get(f"{self.base_url}/funds", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting funds: {e}")
            return None
    
    def get_holdings(self):
        """Get current holdings"""
        try:
            response = self.session.get(f"{self.base_url}/holdings", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting holdings: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        try:
            response = self.session.get(f"{self.base_url}/positions", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return None
    
    def place_order(self, exchange, symbol, transaction_type, quantity, price=None, order_type="MARKET"):
        """Place an order"""
        try:
            payload = {
                "exchange": exchange,
                "symbol": symbol,
                "transaction_type": transaction_type,  # BUY or SELL
                "quantity": quantity,
                "order_type": order_type  # MARKET or LIMIT
            }
            
            if order_type == "LIMIT" and price:
                payload["price"] = price
            
            response = self.session.post(f"{self.base_url}/orders", json=payload, headers=self.headers)
            response.raise_for_status()
            order_data = response.json()
            
            logger.info(f"{transaction_type} order placed for {quantity} shares of {symbol}")
            return order_data
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def get_order_status(self, order_id):
        """Check status of an order"""
        try:
            response = self.session.get(f"{self.base_url}/orders/{order_id}", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error checking order status: {e}")
            return None
    
    def get_market_data(self, exchange, symbol):
        """Get market data for a symbol"""
        try:
            params = {
                "exchange": exchange,
                "symbol": symbol
            }
            response = self.session.get(f"{self.base_url}/market-data", params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def get_historical_data(self, exchange, symbol, interval="1d", from_date=None, to_date=None):
        """Get historical price data"""
        try:
            # Calculate dates if not provided
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")
            if not from_date:
                # Default to 100 days of data
                from_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
            
            params = {
                "exchange": exchange,
                "symbol": symbol,
                "interval": interval,
                "from_date": from_date,
                "to_date": to_date
            }
            
            response = self.session.get(f"{self.base_url}/historical-data", params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            # Convert to pandas DataFrame
            if data and "candles" in data:
                df = pd.DataFrame(data["candles"], columns=["date", "open", "high", "low", "close", "volume"])
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                return df
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()


class IndianMarketTrader:
    """Main trading system for Indian markets"""
    
    def __init__(self, api, config_path=None):
        """
        Initialize the trading system
        
        Parameters:
        -----------
        api : HDFCSecuritiesAPI
            Instance of the HDFC Securities API
        config_path : str
            Path to configuration file
        """
        self.api = api
        self.running = False
        self.monitoring_thread = None
        self.positions = {}
        self.watchlist = []
        self.signals = {}
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set trading parameters
        self.max_position_size = self.config.get("max_position_size", 0.05)  # Max 5% of capital per position
        self.max_total_exposure = self.config.get("max_total_exposure", 0.60)  # Max 60% total exposure
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.03)  # 3% stop loss
        self.take_profit_pct = self.config.get("take_profit_pct", 0.08)  # 8% take profit
        self.check_interval = self.config.get("check_interval", 300)  # 5 minutes
        
        # Market data
        self.market_status = "CLOSED"  # OPEN, CLOSED
        self.market_indices = {
            "NIFTY 50": None,
            "NIFTY BANK": None,
            "NIFTY MIDCAP 100": None
        }
        
        logger.info("Indian Market Trader initialized")
    
    def _load_config(self, config_path):
        """Load configuration from file"""
        if not config_path or not os.path.exists(config_path):
            logger.info("No config file found, using default settings")
            return {
                "max_position_size": 0.05,
                "max_total_exposure": 0.60,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.08,
                "check_interval": 300,
                "exchanges": ["NSE", "BSE"],
                "watchlist": []
            }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {
                "max_position_size": 0.05,
                "max_total_exposure": 0.60,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.08,
                "check_interval": 300,
                "exchanges": ["NSE", "BSE"],
                "watchlist": []
            }
    
    def save_config(self, config_path):
        """Save configuration to file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def start(self):
        """Start the trading system"""
        if self.running:
            logger.warning("Trading system is already running")
            return False
        
        # Log in to API
        if not self.api.login():
            logger.error("Failed to log in, cannot start trading system")
            return False
        
        # Load watchlist from config
        self.watchlist = self.config.get("watchlist", [])
        
        # Set flag and start monitoring thread
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_market)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Trading system started")
        return True
    
    def stop(self):
        """Stop the trading system"""
        if not self.running:
            logger.warning("Trading system is not running")
            return False
        
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Trading system stopped")
        return True
    
    def _monitor_market(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Update market status
                self._update_market_status()
                
                # If market is open, check positions and look for new opportunities
                if self.market_status == "OPEN":
                    # Update positions
                    self._update_positions()
                    
                    # Check each position for exit conditions
                    for symbol, position in self.positions.items():
                        self._check_exit_conditions(symbol, position)
                    
                    # Scan watchlist for new opportunities
                    self._scan_for_entries()
                
                # Sleep for the check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                time.sleep(self.check_interval)
    
    def _update_market_status(self):
        """Update market status and indices"""
        # Market hours for NSE: 9:15 AM to 3:30 PM IST, Monday to Friday
        now = datetime.now()
        
        # Check if today is a weekday (0 = Monday, 4 = Friday)
        if now.weekday() > 4:
            self.market_status = "CLOSED"
            return
        
        # Check market hours (IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if market_open <= now <= market_close:
            self.market_status = "OPEN"
            
            # Update market indices
            for index_name in self.market_indices:
                try:
                    # This assumes you have a method to get index values, adjust as needed
                    data = self.api.get_market_data("NSE", index_name)
                    if data:
                        self.market_indices[index_name] = data.get("last_price")
                except Exception as e:
                    logger.error(f"Error updating index {index_name}: {e}")
        else:
            self.market_status = "CLOSED"
    
    def _update_positions(self):
        """Update current positions"""
        try:
            # Get positions from API
            positions_data = self.api.get_positions()
            if not positions_data:
                return
            
            # Process positions
            current_positions = {}
            for pos in positions_data:
                symbol = pos.get("symbol")
                exchange = pos.get("exchange")
                quantity = pos.get("quantity", 0)
                avg_price = pos.get("average_price", 0)
                
                if quantity > 0:
                    current_positions[symbol] = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "quantity": quantity,
                        "avg_price": avg_price,
                        "current_price": 0,
                        "stop_loss": avg_price * (1 - self.stop_loss_pct),
                        "take_profit": avg_price * (1 + self.take_profit_pct)
                    }
            
            # Update positions with current market prices
            for symbol, position in current_positions.items():
                try:
                    market_data = self.api.get_market_data(position["exchange"], symbol)
                    if market_data:
                        position["current_price"] = market_data.get("last_price", 0)
                except Exception as e:
                    logger.error(f"Error updating price for {symbol}: {e}")
            
            self.positions = current_positions
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _check_exit_conditions(self, symbol, position):
        """Check if a position should be exited"""
        if not position.get("current_price"):
            return
        
        current_price = position["current_price"]
        quantity = position["quantity"]
        
        # Check stop loss
        if current_price <= position["stop_loss"]:
            logger.info(f"Stop loss triggered for {symbol} at {current_price}")
            self._execute_exit(symbol, position["exchange"], quantity, "Stop Loss")
        
        # Check take profit
        elif current_price >= position["take_profit"]:
            logger.info(f"Take profit triggered for {symbol} at {current_price}")
            self._execute_exit(symbol, position["exchange"], quantity, "Take Profit")
    
    def _execute_exit(self, symbol, exchange, quantity, reason):
        """Execute an exit order"""
        try:
            order = self.api.place_order(
                exchange=exchange,
                symbol=symbol,
                transaction_type="SELL",
                quantity=quantity
            )
            
            if order:
                logger.info(f"Exit order placed for {symbol} ({reason}): {quantity} shares")
                return order
            else:
                logger.error(f"Failed to place exit order for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing exit for {symbol}: {e}")
            return None
    
    def _scan_for_entries(self):
        """Scan watchlist for entry opportunities"""
        # Get available funds
        funds_data = self.api.get_funds()
        if not funds_data:
            logger.error("Could not retrieve funds data")
            return
        
        available_cash = funds_data.get("available_cash", 0)
        
        # Calculate current exposure
        total_position_value = sum(p["current_price"] * p["quantity"] for p in self.positions.values())
        current_exposure = total_position_value / (total_position_value + available_cash) if (total_position_value + available_cash) > 0 else 0
        
        # Check if we're already at max exposure
        if current_exposure >= self.max_total_exposure:
            logger.info(f"Maximum exposure reached ({current_exposure:.2%}), not taking new positions")
            return
        
        # Analyze each stock in the watchlist
        for stock in self.watchlist:
            symbol = stock.get("symbol")
            exchange = stock.get("exchange", "NSE")
            
            # Skip if we already have a position in this stock
            if symbol in self.positions:
                continue
            
            # Run strategy analysis
            signal = self._analyze_stock(exchange, symbol)
            if signal:
                self.signals[symbol] = signal
                
                # If entry signal is generated, execute entry
                if signal.get("action") == "BUY":
                    # Calculate position size
                    max_investment = available_cash * self.max_position_size
                    price = signal.get("price", 0)
                    
                    if price > 0:
                        quantity = int(max_investment / price)
                        
                        if quantity > 0:
                            self._execute_entry(exchange, symbol, quantity, price)
    
    def _analyze_stock(self, exchange, symbol):
        """Analyze a stock for trading signals"""
        try:
            # Get historical data
            df = self.api.get_historical_data(exchange, symbol)
            if df.empty:
                return None
            
            # Calculate indicators
            # 1. Moving Averages
            df['sma20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['sma50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['ema13'] = EMAIndicator(close=df['close'], window=13).ema_indicator()
            df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
            
            # 2. RSI
            df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
            
            # 3. ATR for volatility
            df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            
            # 4. Volume indicators
            df['volume_sma20'] = SMAIndicator(close=df['volume'], window=20).sma_indicator()
            
            # Get latest data point
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Current price
            current_price = latest['close']
            
            # Check for entry conditions (Mean Reversion with Momentum Confirmation)
            signal = None
            
            # 1. Stock is in overall uptrend (above 50 SMA and 21 EMA)
            uptrend = current_price > latest['sma50'] and current_price > latest['ema21']
            
            # 2. Mean reversion setup: Recent pullback but still in uptrend
            pullback = (current_price < latest['ema13'] and 
                        current_price / df['close'].rolling(10).max() < 0.95 and  # At least 5% pullback from recent high
                        uptrend)
            
            # 3. RSI conditions (oversold in an uptrend)
            rsi_condition = 30 < latest['rsi'] < 45 and prev['rsi'] < latest['rsi']  # RSI turning up from oversold
            
            # 4. Volume confirmation (decreasing volume on pullback, not panic selling)
            volume_condition = latest['volume'] < latest['volume_sma20']
            
            # Generate buy signal if all conditions met
            if pullback and rsi_condition and volume_condition:
                # Calculate dynamic stop loss and take profit based on ATR
                atr = latest['atr']
                stop_loss = current_price - (0.5 * atr)
                take_profit = current_price + (1.5 * atr)
                
                signal = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "price": current_price,
                    "action": "BUY",
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "Mean Reversion with Momentum Confirmation"
                }
                
                logger.info(f"Buy signal generated for {symbol} at {current_price}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _execute_entry(self, exchange, symbol, quantity, price):
        """Execute an entry order"""
        try:
            order = self.api.place_order(
                exchange=exchange,
                symbol=symbol,
                transaction_type="BUY",
                quantity=quantity
            )
            
            if order:
                logger.info(f"Entry order placed for {symbol}: {quantity} shares at approximately ₹{price}")
                return order
            else:
                logger.error(f"Failed to place entry order for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing entry for {symbol}: {e}")
            return None
    
    def add_to_watchlist(self, symbol, exchange="NSE"):
        """Add a stock to the watchlist"""
        for stock in self.watchlist:
            if stock["symbol"] == symbol and stock["exchange"] == exchange:
                logger.info(f"{symbol} is already in the watchlist")
                return
        
        self.watchlist.append({
            "symbol": symbol,
            "exchange": exchange
        })
        
        # Update config
        self.config["watchlist"] = self.watchlist
        
        logger.info(f"Added {symbol} to watchlist")
    
    def remove_from_watchlist(self, symbol, exchange="NSE"):
        """Remove a stock from the watchlist"""
        self.watchlist = [s for s in self.watchlist if not (s["symbol"] == symbol and s["exchange"] == exchange)]
        
        # Update config
        self.config["watchlist"] = self.watchlist
        
        logger.info(f"Removed {symbol} from watchlist")
    
    def get_watchlist(self):
        """Get the current watchlist"""
        return self.watchlist
    
    def get_current_positions(self):
        """Get current positions with details"""
        if not self.running:
            # If system is not running, get fresh data
            positions_data = self.api.get_positions()
            if not positions_data:
                return {}
            
            current_positions = {}
            for pos in positions_data:
                symbol = pos.get("symbol")
                exchange = pos.get("exchange")
                quantity = pos.get("quantity", 0)
                avg_price = pos.get("average_price", 0)
                
                if quantity > 0:
                    market_data = self.api.get_market_data(exchange, symbol)
                    current_price = market_data.get("last_price", 0) if market_data else 0
                    
                    current_positions[symbol] = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "quantity": quantity,
                        "avg_price": avg_price,
                        "current_price": current_price,
                        "market_value": quantity * current_price,
                        "profit_loss": (current_price - avg_price) * quantity,
                        "profit_loss_pct": ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0
                    }
            
            return current_positions
        else:
            # Return positions maintained by the running system
            return self.positions
    
    def generate_report(self):
        """Generate a performance report"""
        try:
            # Get current positions
            positions = self.get_current_positions()
            
            # Get historical orders to calculate closed positions
            # This is a placeholder - you'll need to implement actual order history retrieval
            # based on your broker's API
            closed_positions = []  # Placeholder
            
            # Generate report
            report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market_status": self.market_status,
                "indices": self.market_indices,
                "open_positions": len(positions),
                "total_investment": sum(p["avg_price"] * p["quantity"] for p in positions.values()),
                "total_market_value": sum(p["market_value"] for p in positions.values() if "market_value" in p),
                "unrealized_pnl": sum(p["profit_loss"] for p in positions.values() if "profit_loss" in p),
                "positions": positions,
                "closed_positions": closed_positions
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def manual_buy(self, exchange, symbol, quantity):
        """Execute a manual buy order"""
        try:
            # Get market data
            market_data = self.api.get_market_data(exchange, symbol)
            if not market_data:
                logger.error(f"Could not get market data for {symbol}")
                return None
            
            # Place order
            order = self.api.place_order(
                exchange=exchange,
                symbol=symbol,
                transaction_type="BUY",
                quantity=quantity
            )
            
            if order:
                logger.info(f"Manual buy order placed for {symbol}: {quantity} shares")
                return order
            else:
                logger.error(f"Failed to place manual buy order for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing manual buy for {symbol}: {e}")
            return None
    
    def manual_sell(self, exchange, symbol, quantity):
        """Execute a manual sell order"""
        try:
            # Place order
            order = self.api.place_order(
                exchange=exchange,
                symbol=symbol,
                transaction_type="SELL",
                quantity=quantity
            )
            
            if order:
                logger.info(f"Manual sell order placed for {symbol}: {quantity} shares")
                return order
            else:
                logger.error(f"Failed to place manual sell order for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing manual sell for {symbol}: {e}")
            return None


class TradingSystemUI:
    """GUI for the Indian Market Trading System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Indian Market Trading System")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set up variables
        self.status_var = tk.StringVar(value="Not Started")
        self.user_id_var = tk.StringVar()
        self.password_var = tk.StringVar()
        # self.pin_var = tk.StringVar()
        self.api_key_var = tk.StringVar()
        
        # Trading system instance
        self.api = None
        self.trading_system = None
        
        # Create UI elements
        self._create_menu()
        self._create_notebook()
        self._create_footer()
        
        # Schedule periodic UI updates
        self._schedule_updates()
    
    def _create_menu(self):
        """Create the menu bar"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Login", command=self._login_dialog)
        file_menu.add_command(label="Save Config", command=self._save_config_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._exit_application)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Trading menu
        trading_menu = tk.Menu(menubar, tearoff=0)
        trading_menu.add_command(label="Start Trading System", command=self._start_trading)
        trading_menu.add_command(label="Stop Trading System", command=self._stop_trading)
        trading_menu.add_separator()
        trading_menu.add_command(label="Buy Stock", command=self._buy_dialog)
        trading_menu.add_command(label="Sell Stock", command=self._sell_dialog)
        menubar.add_cascade(label="Trading", menu=trading_menu)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Generate Report", command=self._generate_report)
        analysis_menu.add_command(label="Scan Watchlist", command=self._scan_watchlist)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Stock Screener", command=self._stock_screener)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        
        # Set the menu bar
        self.root.config(menu=menubar)
    
    def _create_notebook(self):
        """Create the main notebook with tabs"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create tabs
        self.dashboard_frame = ttk.Frame(self.notebook)
        self.positions_frame = ttk.Frame(self.notebook)
        self.watchlist_frame = ttk.Frame(self.notebook)
        self.signals_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.dashboard_frame, text="Dashboard")
        self.notebook.add(self.positions_frame, text="Positions")
        self.notebook.add(self.watchlist_frame, text="Watchlist")
        self.notebook.add(self.signals_frame, text="Signals")
        self.notebook.add(self.settings_frame, text="Settings")
        
        # Create content for each tab
        self._create_dashboard_tab()
        self._create_positions_tab()
        self._create_watchlist_tab()
        self._create_signals_tab()
        self._create_settings_tab()
    
    def _create_dashboard_tab(self):
        """Create content for the dashboard tab"""
        # Market overview frame
        overview_frame = ttk.LabelFrame(self.dashboard_frame, text="Market Overview")
        overview_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Market indices
        indices_frame = ttk.Frame(overview_frame)
        indices_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(indices_frame, text="NIFTY 50:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.nifty_var = tk.StringVar(value="N/A")
        ttk.Label(indices_frame, textvariable=self.nifty_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(indices_frame, text="NIFTY BANK:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.bank_nifty_var = tk.StringVar(value="N/A")
        ttk.Label(indices_frame, textvariable=self.bank_nifty_var).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(indices_frame, text="NIFTY MIDCAP 100:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.midcap_var = tk.StringVar(value="N/A")
        ttk.Label(indices_frame, textvariable=self.midcap_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(indices_frame, text="Market Status:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.market_status_var = tk.StringVar(value="CLOSED")
        ttk.Label(indices_frame, textvariable=self.market_status_var).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Portfolio summary frame
        summary_frame = ttk.LabelFrame(self.dashboard_frame, text="Portfolio Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Summary statistics
        stats_frame = ttk.Frame(summary_frame)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(stats_frame, text="Total Investment:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.total_investment_var = tk.StringVar(value="₹0.00")
        ttk.Label(stats_frame, textvariable=self.total_investment_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_frame, text="Current Value:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.current_value_var = tk.StringVar(value="₹0.00")
        ttk.Label(stats_frame, textvariable=self.current_value_var).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_frame, text="Today's P/L:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.today_pl_var = tk.StringVar(value="₹0.00 (0.00%)")
        ttk.Label(stats_frame, textvariable=self.today_pl_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_frame, text="Overall P/L:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.overall_pl_var = tk.StringVar(value="₹0.00 (0.00%)")
        ttk.Label(stats_frame, textvariable=self.overall_pl_var).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Chart frame
        chart_frame = ttk.LabelFrame(self.dashboard_frame, text="Portfolio Performance")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create a Figure and a Canvas
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create initial empty plots
        self.ax1 = self.fig.add_subplot(211)  # Portfolio value over time
        self.ax2 = self.fig.add_subplot(212)  # Positions breakdown
        
        # Initialize with empty data
        self._update_charts()
    
    def _create_positions_tab(self):
        """Create content for the positions tab"""
        # Control frame
        control_frame = ttk.Frame(self.positions_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Refresh", command=self._refresh_positions).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Buy", command=self._buy_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Sell", command=self._sell_dialog).pack(side=tk.LEFT, padx=5)
        
        # Positions table
        columns = ('symbol', 'exchange', 'quantity', 'avg_price', 'current_price', 
                  'market_value', 'profit_loss', 'profit_loss_pct', 'stop_loss', 'take_profit')
        
        self.positions_tree = ttk.Treeview(self.positions_frame, columns=columns, show='headings')
        
        # Define headings
        self.positions_tree.heading('symbol', text='Symbol')
        self.positions_tree.heading('exchange', text='Exchange')
        self.positions_tree.heading('quantity', text='Quantity')
        self.positions_tree.heading('avg_price', text='Avg Price')
        self.positions_tree.heading('current_price', text='Current Price')
        self.positions_tree.heading('market_value', text='Market Value')
        self.positions_tree.heading('profit_loss', text='P/L (₹)')
        self.positions_tree.heading('profit_loss_pct', text='P/L (%)')
        self.positions_tree.heading('stop_loss', text='Stop Loss')
        self.positions_tree.heading('take_profit', text='Take Profit')
        
        # Define columns
        self.positions_tree.column('symbol', width=80, anchor=tk.CENTER)
        self.positions_tree.column('exchange', width=80, anchor=tk.CENTER)
        self.positions_tree.column('quantity', width=80, anchor=tk.CENTER)
        self.positions_tree.column('avg_price', width=100, anchor=tk.CENTER)
        self.positions_tree.column('current_price', width=100, anchor=tk.CENTER)
        self.positions_tree.column('market_value', width=100, anchor=tk.CENTER)
        self.positions_tree.column('profit_loss', width=100, anchor=tk.CENTER)
        self.positions_tree.column('profit_loss_pct', width=100, anchor=tk.CENTER)
        self.positions_tree.column('stop_loss', width=100, anchor=tk.CENTER)
        self.positions_tree.column('take_profit', width=100, anchor=tk.CENTER)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.positions_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscroll=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.positions_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add right-click menu
        self.position_menu = tk.Menu(self.positions_tree, tearoff=0)
        self.position_menu.add_command(label="Sell", command=self._sell_selected_position)
        self.position_menu.add_command(label="Set Stop Loss", command=self._set_stop_loss)
        self.position_menu.add_command(label="Set Take Profit", command=self._set_take_profit)
        
        # Bind right-click to show menu
        self.positions_tree.bind("<Button-3>", self._show_position_menu)
    
    def _create_watchlist_tab(self):
        """Create content for the watchlist tab"""
        # Control frame
        control_frame = ttk.Frame(self.watchlist_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Add Stock", command=self._add_stock_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Remove Stock", command=self._remove_selected_stock).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh", command=self._refresh_watchlist).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Analyze", command=self._analyze_selected_stock).pack(side=tk.LEFT, padx=5)
        
        # Watchlist table
        columns = ('symbol', 'exchange', 'last_price', 'change', 'change_pct', 'volume', 'signal')
        
        self.watchlist_tree = ttk.Treeview(self.watchlist_frame, columns=columns, show='headings')
        
        # Define headings
        self.watchlist_tree.heading('symbol', text='Symbol')
        self.watchlist_tree.heading('exchange', text='Exchange')
        self.watchlist_tree.heading('last_price', text='Last Price')
        self.watchlist_tree.heading('change', text='Change')
        self.watchlist_tree.heading('change_pct', text='Change %')
        self.watchlist_tree.heading('volume', text='Volume')
        self.watchlist_tree.heading('signal', text='Signal')
        
        # Define columns
        self.watchlist_tree.column('symbol', width=80, anchor=tk.CENTER)
        self.watchlist_tree.column('exchange', width=80, anchor=tk.CENTER)
        self.watchlist_tree.column('last_price', width=100, anchor=tk.CENTER)
        self.watchlist_tree.column('change', width=100, anchor=tk.CENTER)
        self.watchlist_tree.column('change_pct', width=100, anchor=tk.CENTER)
        self.watchlist_tree.column('volume', width=100, anchor=tk.CENTER)
        self.watchlist_tree.column('signal', width=100, anchor=tk.CENTER)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.watchlist_frame, orient=tk.VERTICAL, command=self.watchlist_tree.yview)
        self.watchlist_tree.configure(yscroll=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.watchlist_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add right-click menu
        self.watchlist_menu = tk.Menu(self.watchlist_tree, tearoff=0)
        self.watchlist_menu.add_command(label="Buy", command=self._buy_selected_stock)
        self.watchlist_menu.add_command(label="Analyze", command=self._analyze_selected_stock)
        self.watchlist_menu.add_command(label="Remove", command=self._remove_selected_stock)
        
        # Bind right-click to show menu
        self.watchlist_tree.bind("<Button-3>", self._show_watchlist_menu)
    
    def _create_signals_tab(self):
        """Create content for the signals tab"""
        # Control frame
        control_frame = ttk.Frame(self.signals_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Refresh", command=self._refresh_signals).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Execute", command=self._execute_selected_signal).pack(side=tk.LEFT, padx=5)
        
        # Signals table
        columns = ('symbol', 'exchange', 'price', 'action', 'stop_loss', 'take_profit', 'time', 'reason')
        
        self.signals_tree = ttk.Treeview(self.signals_frame, columns=columns, show='headings')
        
        # Define headings
        self.signals_tree.heading('symbol', text='Symbol')
        self.signals_tree.heading('exchange', text='Exchange')
        self.signals_tree.heading('price', text='Price')
        self.signals_tree.heading('action', text='Action')
        self.signals_tree.heading('stop_loss', text='Stop Loss')
        self.signals_tree.heading('take_profit', text='Take Profit')
        self.signals_tree.heading('time', text='Time')
        self.signals_tree.heading('reason', text='Reason')
        
        # Define columns
        self.signals_tree.column('symbol', width=80, anchor=tk.CENTER)
        self.signals_tree.column('exchange', width=80, anchor=tk.CENTER)
        self.signals_tree.column('price', width=80, anchor=tk.CENTER)
        self.signals_tree.column('action', width=80, anchor=tk.CENTER)
        self.signals_tree.column('stop_loss', width=80, anchor=tk.CENTER)
        self.signals_tree.column('take_profit', width=80, anchor=tk.CENTER)
        self.signals_tree.column('time', width=150, anchor=tk.CENTER)
        self.signals_tree.column('reason', width=200, anchor=tk.W)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.signals_frame, orient=tk.VERTICAL, command=self.signals_tree.yview)
        self.signals_tree.configure(yscroll=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.signals_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add right-click menu
        self.signal_menu = tk.Menu(self.signals_tree, tearoff=0)
        self.signal_menu.add_command(label="Execute", command=self._execute_selected_signal)
        
        # Bind right-click to show menu
        self.signals_tree.bind("<Button-3>", self._show_signal_menu)
    
    def _create_settings_tab(self):
        """Create content for the settings tab"""
        # Login frame
        login_frame = ttk.LabelFrame(self.settings_frame, text="HDFC Securities Login")
        login_frame.pack(fill=tk.X, padx=10, pady=5)
        
        user_frame = ttk.Frame(login_frame)
        user_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(user_frame, text="User ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(user_frame, textvariable=self.user_id_var, width=30).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(user_frame, text="Password:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(user_frame, textvariable=self.password_var, width=30, show="*").grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # ttk.Label(user_frame, text="PIN:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        # ttk.Entry(user_frame, textvariable=self.pin_var, width=30, show="*").grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(user_frame, text="API Key (optional):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(user_frame, textvariable=self.api_key_var, width=30).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(user_frame, text="Login", command=self._login).grid(row=4, column=1, sticky=tk.E, padx=5, pady=5)
        
        # Trading Parameters frame
        params_frame = ttk.LabelFrame(self.settings_frame, text="Trading Parameters")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # Max position size
        ttk.Label(params_grid, text="Max Position Size (% of capital):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.position_size_var = tk.DoubleVar(value=5.0)
        ttk.Entry(params_grid, textvariable=self.position_size_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Max exposure
        ttk.Label(params_grid, text="Max Total Exposure (%):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.exposure_var = tk.DoubleVar(value=60.0)
        ttk.Entry(params_grid, textvariable=self.exposure_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Stop loss
        ttk.Label(params_grid, text="Default Stop Loss (%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.stop_loss_var = tk.DoubleVar(value=3.0)
        ttk.Entry(params_grid, textvariable=self.stop_loss_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Take profit
        ttk.Label(params_grid, text="Default Take Profit (%):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.take_profit_var = tk.DoubleVar(value=8.0)
        ttk.Entry(params_grid, textvariable=self.take_profit_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Check interval
        ttk.Label(params_grid, text="Check Interval (seconds):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.interval_var = tk.IntVar(value=300)
        ttk.Entry(params_grid, textvariable=self.interval_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Save button
        ttk.Button(params_grid, text="Save Parameters", command=self._save_parameters).grid(row=5, column=1, sticky=tk.E, padx=5, pady=5)
        
        # Additional Settings
        additional_frame = ttk.LabelFrame(self.settings_frame, text="Additional Settings")
        additional_frame.pack(fill=tk.X, padx=10, pady=5)
        
        add_settings = ttk.Frame(additional_frame)
        add_settings.pack(fill=tk.X, padx=10, pady=5)
        
        # Auto-start option
        self.autostart_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(add_settings, text="Auto-start trading system on login", variable=self.autostart_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Sound alerts option
        self.sound_alerts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(add_settings, text="Enable sound alerts for signals and executions", variable=self.sound_alerts_var).pack(anchor=tk.W, padx=5, pady=2)
    
    def _create_footer(self):
        """Create the footer with status information"""
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        # Status label
        ttk.Label(footer_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        ttk.Label(footer_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        
        # Last updated timestamp
        ttk.Label(footer_frame, text="Last Updated:").pack(side=tk.RIGHT, padx=5)
        self.last_updated_var = tk.StringVar(value="Never")
        ttk.Label(footer_frame, textvariable=self.last_updated_var).pack(side=tk.RIGHT, padx=5)
    
    def _login_dialog(self):
        """Show the login dialog"""
        # If already logged in, show message
        if self.api and self.trading_system:
            messagebox.showinfo("Login", "Already logged in")
            return
        
        # If credentials are already filled in settings tab, use those
        # if self.user_id_var.get() and self.password_var.get() and self.pin_var.get():
        #     self._login()
        #     return
        
        # Otherwise show dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("HDFC Securities Login")
        dialog.geometry("300x200")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="User ID:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        user_id = tk.StringVar()
        ttk.Entry(dialog, textvariable=user_id, width=20).grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Password:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        password = tk.StringVar()
        ttk.Entry(dialog, textvariable=password, width=20, show="*").grid(row=1, column=1, padx=10, pady=5)
        
        # ttk.Label(dialog, text="PIN:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        # pin = tk.StringVar()
        # ttk.Entry(dialog, textvariable=pin, width=20, show="*").grid(row=2, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="API Key (optional):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        api_key = tk.StringVar()
        ttk.Entry(dialog, textvariable=api_key, width=20).grid(row=3, column=1, padx=10, pady=5)
        
        def do_login():
            self.user_id_var.set(user_id.get())
            self.password_var.set(password.get())
            # self.pin_var.set(pin.get())
            self.api_key_var.set(api_key.get())
            dialog.destroy()
            self._login()
        
        ttk.Button(dialog, text="Login", command=do_login).grid(row=4, column=1, sticky=tk.E, padx=10, pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(row=4, column=0, sticky=tk.W, padx=10, pady=10)
    
    def _login(self):
        """Log in to HDFC Securities"""
        # Check credentials
        # if not self.user_id_var.get() or not self.password_var.get() or not self.pin_var.get():
        #     messagebox.showerror("Login Error", "User ID, Password and PIN are required")
        #     return
        
        try:
            # Initialize API client
            self.api = HDFCSecuritiesAPI(
                user_id=self.user_id_var.get(),
                password=self.password_var.get(),
                # pin=self.pin_var.get(),
                api_key=self.api_key_var.get() if self.api_key_var.get() else None
            )
            
            # Try to login
            if self.api.login():
                messagebox.showinfo("Login", "Successfully logged in to HDFC Securities")
                self.status_var.set("Logged in")
                
                # Initialize trading system
                self.trading_system = IndianMarketTrader(self.api)
                
                # Update UI with user data
                self._refresh_all()
                
                # Auto-start if enabled
                if self.autostart_var.get():
                    self._start_trading()
            else:
                messagebox.showerror("Login Error", "Failed to log in. Please check your credentials.")
                self.api = None
                
        except Exception as e:
            messagebox.showerror("Login Error", f"An error occurred: {str(e)}")
            self.api = None
    
    def _save_config_dialog(self):
        """Show dialog to save configuration"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        # Ask for file path
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            # Save config
            if self.trading_system.save_config(filepath):
                messagebox.showinfo("Save Config", f"Configuration saved to {filepath}")
            else:
                messagebox.showerror("Save Config", "Failed to save configuration")
    
    def _save_parameters(self):
        """Save trading parameters"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        try:
            # Update trading system parameters
            self.trading_system.max_position_size = self.position_size_var.get() / 100.0  # Convert from percentage
            self.trading_system.max_total_exposure = self.exposure_var.get() / 100.0  # Convert from percentage
            self.trading_system.stop_loss_pct = self.stop_loss_var.get() / 100.0  # Convert from percentage
            self.trading_system.take_profit_pct = self.take_profit_var.get() / 100.0  # Convert from percentage
            self.trading_system.check_interval = self.interval_var.get()
            
            # Update config
            self.trading_system.config["max_position_size"] = self.trading_system.max_position_size
            self.trading_system.config["max_total_exposure"] = self.trading_system.max_total_exposure
            self.trading_system.config["stop_loss_pct"] = self.trading_system.stop_loss_pct
            self.trading_system.config["take_profit_pct"] = self.trading_system.take_profit_pct
            self.trading_system.config["check_interval"] = self.trading_system.check_interval
            
            messagebox.showinfo("Parameters", "Trading parameters saved")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save parameters: {str(e)}")
    
    def _start_trading(self):
        """Start the trading system"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        if self.trading_system.running:
            messagebox.showinfo("Trading", "Trading system is already running")
            return
        
        try:
            if self.trading_system.start():
                messagebox.showinfo("Trading", "Trading system started")
                self.status_var.set("Trading system running")
                self._refresh_all()
            else:
                messagebox.showerror("Trading", "Failed to start trading system")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start trading system: {str(e)}")
    
    def _stop_trading(self):
        """Stop the trading system"""
        if not self.trading_system:
            messagebox.showerror("Error", "Trading system not initialized")
            return
        
        if not self.trading_system.running:
            messagebox.showinfo("Trading", "Trading system is not running")
            return
        
        try:
            if self.trading_system.stop():
                messagebox.showinfo("Trading", "Trading system stopped")
                self.status_var.set("Trading system stopped")
            else:
                messagebox.showerror("Trading", "Failed to stop trading system")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop trading system: {str(e)}")
    
    
    def _buy_dialog(self):
        """Show dialog to buy a stock"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Buy Stock")
        dialog.geometry("350x250")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Stock details
        ttk.Label(dialog, text="Exchange:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        exchange_var = tk.StringVar(value="NSE")
        ttk.Combobox(dialog, textvariable=exchange_var, values=["NSE", "BSE"], state="readonly").grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Symbol:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        symbol_var = tk.StringVar()
        symbol_entry = ttk.Entry(dialog, textvariable=symbol_var, width=20)
        symbol_entry.grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Quantity:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        quantity_var = tk.IntVar(value=1)
        ttk.Entry(dialog, textvariable=quantity_var, width=20).grid(row=2, column=1, padx=10, pady=5)
        
        # Current price display
        ttk.Label(dialog, text="Current Price:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        price_var = tk.StringVar(value="N/A")
        ttk.Label(dialog, textvariable=price_var).grid(row=3, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Get price button
        def get_price():
            if not symbol_var.get():
                messagebox.showerror("Error", "Please enter a symbol")
                return
            
            try:
                market_data = self.api.get_market_data(exchange_var.get(), symbol_var.get())
                if market_data and "last_price" in market_data:
                    price_var.set(f"₹{market_data['last_price']:.2f}")
                else:
                    price_var.set("N/A")
            except Exception as e:
                price_var.set("Error")
                messagebox.showerror("Error", f"Failed to get price: {str(e)}")
        
        ttk.Button(dialog, text="Get Price", command=get_price).grid(row=4, column=0, padx=10, pady=5)
        
        # Buy button
        def do_buy():
            if not symbol_var.get() or quantity_var.get() <= 0:
                messagebox.showerror("Error", "Please enter a valid symbol and quantity")
                return
            
            try:
                order = self.trading_system.manual_buy(
                    exchange=exchange_var.get(),
                    symbol=symbol_var.get(),
                    quantity=quantity_var.get()
                )
                
                if order:
                    messagebox.showinfo("Buy", f"Buy order placed for {quantity_var.get()} shares of {symbol_var.get()}")
                    dialog.destroy()
                    self._refresh_positions()
                else:
                    messagebox.showerror("Error", "Failed to place buy order")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to buy stock: {str(e)}")
        
        ttk.Button(dialog, text="Buy", command=do_buy).grid(row=4, column=1, sticky=tk.E, padx=10, pady=5)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(row=5, column=1, sticky=tk.E, padx=10, pady=5)
    
    def _sell_dialog(self):
        """Show dialog to sell a stock"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        # Get current positions
        positions = self.trading_system.get_current_positions()
        if not positions:
            messagebox.showinfo("Positions", "No positions to sell")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Sell Stock")
        dialog.geometry("350x250")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Position selection
        ttk.Label(dialog, text="Symbol:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        symbols = list(positions.keys())
        symbol_var = tk.StringVar(value=symbols[0] if symbols else "")
        symbol_combo = ttk.Combobox(dialog, textvariable=symbol_var, values=symbols, state="readonly")
        symbol_combo.grid(row=0, column=1, padx=10, pady=5)
        
        # Quantity
        ttk.Label(dialog, text="Quantity:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        quantity_var = tk.IntVar(value=1)
        quantity_entry = ttk.Entry(dialog, textvariable=quantity_var, width=20)
        quantity_entry.grid(row=1, column=1, padx=10, pady=5)
        
        # Current holdings
        ttk.Label(dialog, text="Current Holdings:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        holdings_var = tk.StringVar(value=f"{positions[symbols[0]]['quantity'] if symbols else 0}")
        ttk.Label(dialog, textvariable=holdings_var).grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Current price
        ttk.Label(dialog, text="Current Price:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        price_var = tk.StringVar(value=f"₹{positions[symbols[0]]['current_price'] if symbols else 0:.2f}")
        ttk.Label(dialog, textvariable=price_var).grid(row=3, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Update displayed information when symbol changes
        def update_info(*args):
            selected_symbol = symbol_var.get()
            if selected_symbol in positions:
                position = positions[selected_symbol]
                holdings_var.set(f"{position['quantity']}")
                price_var.set(f"₹{position['current_price']:.2f}")
                quantity_var.set(position['quantity'])  # Default to selling all
        
        symbol_var.trace('w', update_info)
        
        # Sell button
        def do_sell():
            selected_symbol = symbol_var.get()
            if not selected_symbol:
                messagebox.showerror("Error", "Please select a symbol")
                return
            
            if quantity_var.get() <= 0:
                messagebox.showerror("Error", "Please enter a valid quantity")
                return
            
            if quantity_var.get() > positions[selected_symbol]['quantity']:
                messagebox.showerror("Error", "Cannot sell more than you own")
                return
            
            try:
                order = self.trading_system.manual_sell(
                    exchange=positions[selected_symbol]['exchange'],
                    symbol=selected_symbol,
                    quantity=quantity_var.get()
                )
                
                if order:
                    messagebox.showinfo("Sell", f"Sell order placed for {quantity_var.get()} shares of {selected_symbol}")
                    dialog.destroy()
                    self._refresh_positions()
                else:
                    messagebox.showerror("Error", "Failed to place sell order")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to sell stock: {str(e)}")
        
        ttk.Button(dialog, text="Sell", command=do_sell).grid(row=4, column=1, sticky=tk.E, padx=10, pady=5)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(row=5, column=1, sticky=tk.E, padx=10, pady=5)
    
    def _add_stock_dialog(self):
        """Show dialog to add a stock to watchlist"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Stock to Watchlist")
        dialog.geometry("300x150")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Exchange:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        exchange_var = tk.StringVar(value="NSE")
        ttk.Combobox(dialog, textvariable=exchange_var, values=["NSE", "BSE"], state="readonly").grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Symbol:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        symbol_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=symbol_var, width=20).grid(row=1, column=1, padx=10, pady=5)
        
        # Add button
        def add_stock():
            if not symbol_var.get():
                messagebox.showerror("Error", "Please enter a symbol")
                return
            
            try:
                # Verify the symbol exists
                market_data = self.api.get_market_data(exchange_var.get(), symbol_var.get())
                if not market_data:
                    messagebox.showerror("Error", f"Symbol {symbol_var.get()} not found")
                    return
                
                # Add to watchlist
                self.trading_system.add_to_watchlist(symbol_var.get(), exchange_var.get())
                messagebox.showinfo("Watchlist", f"Added {symbol_var.get()} to watchlist")
                dialog.destroy()
                self._refresh_watchlist()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add stock: {str(e)}")
        
        ttk.Button(dialog, text="Add", command=add_stock).grid(row=2, column=1, sticky=tk.E, padx=10, pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)
    
    def _remove_selected_stock(self):
        """Remove selected stock from watchlist"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        selected = self.watchlist_tree.selection()
        if not selected:
            messagebox.showinfo("Watchlist", "No stock selected")
            return
        
        item = self.watchlist_tree.item(selected[0])
        symbol = item['values'][0]
        exchange = item['values'][1]
        
        # Confirm removal
        if messagebox.askyesno("Confirm", f"Remove {symbol} from watchlist?"):
            self.trading_system.remove_from_watchlist(symbol, exchange)
            self._refresh_watchlist()
    
    def _refresh_all(self):
        """Refresh all data"""
        self._refresh_positions()
        self._refresh_watchlist()
        self._refresh_signals()
        self._update_dashboard()
    
    def _refresh_positions(self):
        """Refresh positions table"""
        if not self.trading_system:
            return
        
        # Clear current items
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        # Get current positions
        positions = self.trading_system.get_current_positions()
        
        # Add to table
        for symbol, pos in positions.items():
            try:
                pl_color = "green" if pos.get('profit_loss', 0) >= 0 else "red"
                self.positions_tree.insert('', tk.END, values=(
                    symbol,
                    pos.get('exchange', 'NSE'),
                    pos.get('quantity', 0),
                    f"₹{pos.get('avg_price', 0):.2f}",
                    f"₹{pos.get('current_price', 0):.2f}",
                    f"₹{pos.get('market_value', 0):.2f}",
                    f"₹{pos.get('profit_loss', 0):.2f}",
                    f"{pos.get('profit_loss_pct', 0):.2f}%",
                    f"₹{pos.get('stop_loss', 0):.2f}" if 'stop_loss' in pos else "N/A",
                    f"₹{pos.get('take_profit', 0):.2f}" if 'take_profit' in pos else "N/A"
                ), tags=(pl_color,))
            except Exception as e:
                print(f"Error adding position {symbol} to table: {e}")
        
        # Configure tag colors
        self.positions_tree.tag_configure("green", foreground="green")
        self.positions_tree.tag_configure("red", foreground="red")
        
        # Update last updated time
        self.last_updated_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def _refresh_watchlist(self):
        """Refresh watchlist table"""
        if not self.trading_system:
            return
        
        # Clear current items
        for item in self.watchlist_tree.get_children():
            self.watchlist_tree.delete(item)
        
        # Get watchlist
        watchlist = self.trading_system.get_watchlist()
        
        # Add each stock to table
        for stock in watchlist:
            try:
                symbol = stock['symbol']
                exchange = stock['exchange']
                
                # Get market data
                market_data = self.api.get_market_data(exchange, symbol)
                
                if market_data:
                    last_price = market_data.get('last_price', 0)
                    prev_close = market_data.get('prev_close', 0)
                    change = last_price - prev_close
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                    volume = market_data.get('volume', 0)
                    
                    # Check for signal
                    signal = "None"
                    if symbol in self.trading_system.signals:
                        sig = self.trading_system.signals[symbol]
                        if sig['action'] == 'BUY':
                            signal = "BUY"
                    
                    # Add to tree with appropriate color tag
                    tag = "up" if change >= 0 else "down"
                    self.watchlist_tree.insert('', tk.END, values=(
                        symbol,
                        exchange,
                        f"₹{last_price:.2f}",
                        f"₹{change:.2f}",
                        f"{change_pct:.2f}%",
                        f"{volume:,}",
                        signal
                    ), tags=(tag,))
            except Exception as e:
                print(f"Error adding {symbol} to watchlist table: {e}")
        
        # Configure tag colors
        self.watchlist_tree.tag_configure("up", foreground="green")
        self.watchlist_tree.tag_configure("down", foreground="red")
        
        # Update last updated time
        self.last_updated_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def _refresh_signals(self):
        """Refresh signals table"""
        if not self.trading_system:
            return
        
        # Clear current items
        for item in self.signals_tree.get_children():
            self.signals_tree.delete(item)
        
        # Get signals
        signals = self.trading_system.signals
        
        # Add to table
        for symbol, signal in signals.items():
            try:
                action = signal.get('action', '')
                action_tag = "buy" if action == "BUY" else "sell" if action == "SELL" else "neutral"
                
                self.signals_tree.insert('', tk.END, values=(
                    signal.get('symbol', ''),
                    signal.get('exchange', ''),
                    f"₹{signal.get('price', 0):.2f}",
                    action,
                    f"₹{signal.get('stop_loss', 0):.2f}",
                    f"₹{signal.get('take_profit', 0):.2f}",
                    signal.get('time', ''),
                    signal.get('reason', '')
                ), tags=(action_tag,))
            except Exception as e:
                print(f"Error adding signal for {symbol} to table: {e}")
        
        # Configure tag colors
        self.signals_tree.tag_configure("buy", foreground="green")
        self.signals_tree.tag_configure("sell", foreground="red")
        self.signals_tree.tag_configure("neutral", foreground="blue")
        
        # Update last updated time
        self.last_updated_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def _update_dashboard(self):
        """Update dashboard information"""
        if not self.trading_system:
            return
        
        try:
            # Update market indices
            for index, var in [
                ("NIFTY 50", self.nifty_var),
                ("NIFTY BANK", self.bank_nifty_var),
                ("NIFTY MIDCAP 100", self.midcap_var)
            ]:
                if index in self.trading_system.market_indices:
                    value = self.trading_system.market_indices[index]
                    if value:
                        var.set(f"₹{value:.2f}")
            
            # Update market status
            self.market_status_var.set(self.trading_system.market_status)
            
            # Generate report for portfolio data
            report = self.trading_system.generate_report()
            if report:
                # Update portfolio summary
                self.total_investment_var.set(f"₹{report.get('total_investment', 0):,.2f}")
                self.current_value_var.set(f"₹{report.get('total_market_value', 0):,.2f}")
                
                # Calculate P/L
                unrealized_pl = report.get('unrealized_pnl', 0)
                total_investment = report.get('total_investment', 0)
                pl_pct = (unrealized_pl / total_investment) * 100 if total_investment > 0 else 0
                
                self.overall_pl_var.set(f"₹{unrealized_pl:,.2f} ({pl_pct:.2f}%)")
                
                # Update charts
                self._update_charts(report)
            
        except Exception as e:
            print(f"Error updating dashboard: {e}")
    
    def _update_charts(self, report=None):
        """Update dashboard charts"""
        # Clear existing plots
        self.ax1.clear()
        self.ax2.clear()
        
        if report and 'positions' in report and report['positions']:
            positions = report['positions']
            
            # Prepare data for portfolio breakdown chart
            symbols = []
            values = []
            colors = []
            
            for symbol, pos in positions.items():
                symbols.append(symbol)
                values.append(pos.get('market_value', 0))
                pl = pos.get('profit_loss', 0)
                colors.append('green' if pl >= 0 else 'red')
            
            # Create pie chart for portfolio breakdown
            if values:
                self.ax2.pie(values, labels=symbols, autopct='%1.1f%%', colors=colors)
                self.ax2.set_title('Portfolio Composition')
            else:
                self.ax2.text(0.5, 0.5, 'No positions', horizontalalignment='center',
                             verticalalignment='center', transform=self.ax2.transAxes)
            
            # Create bar chart for P/L by position
            pl_values = [pos.get('profit_loss_pct', 0) for pos in positions.values()]
            pl_colors = ['green' if pl >= 0 else 'red' for pl in pl_values]
            
            if symbols and pl_values:
                self.ax1.bar(symbols, pl_values, color=pl_colors)
                self.ax1.set_title('Position P/L (%)')
                self.ax1.set_ylabel('P/L %')
                self.ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                # Rotate x-axis labels for better readability
                self.ax1.set_xticklabels(symbols, rotation=45, ha='right')
                
                # Set y-axis limits
                max_abs_pl = max([abs(pl) for pl in pl_values])
                self.ax1.set_ylim(-max_abs_pl*1.1, max_abs_pl*1.1)
            else:
                self.ax1.text(0.5, 0.5, 'No P/L data', horizontalalignment='center',
                             verticalalignment='center', transform=self.ax1.transAxes)
        else:
            # No data available
            self.ax1.text(0.5, 0.5, 'No positions data available', horizontalalignment='center',
                         verticalalignment='center', transform=self.ax1.transAxes)
            self.ax2.text(0.5, 0.5, 'No positions data available', horizontalalignment='center',
                         verticalalignment='center', transform=self.ax2.transAxes)
        
        # Adjust layout and redraw
        self.fig.tight_layout()
        self.chart_canvas.draw()
    
    def _generate_report(self):
        """Generate and display a performance report"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        try:
            report = self.trading_system.generate_report()
            if not report:
                messagebox.showerror("Error", "Failed to generate report")
                return
            
            # Create report window
            report_window = tk.Toplevel(self.root)
            report_window.title("Performance Report")
            report_window.geometry("800x600")
            report_window.transient(self.root)
            
            # Report text
            report_text = tk.Text(report_window, wrap=tk.WORD)
            report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(report_text, orient=tk.VERTICAL, command=report_text.yview)
            report_text.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Format and insert report
            report_text.insert(tk.END, f"=== PERFORMANCE REPORT ===\n")
            report_text.insert(tk.END, f"Generated: {report.get('timestamp', '')}\n\n")
            
            report_text.insert(tk.END, f"Market Status: {report.get('market_status', '')}\n\n")
            
            report_text.insert(tk.END, f"Market Indices:\n")
            for index, value in report.get('indices', {}).items():
                if value:
                    report_text.insert(tk.END, f"  {index}: ₹{value:.2f}\n")
            
            report_text.insert(tk.END, f"\nPortfolio Summary:\n")
            report_text.insert(tk.END, f"  Open Positions: {report.get('open_positions', 0)}\n")
            report_text.insert(tk.END, f"  Total Investment: ₹{report.get('total_investment', 0):,.2f}\n")
            report_text.insert(tk.END, f"  Current Market Value: ₹{report.get('total_market_value', 0):,.2f}\n")
            
            unrealized_pnl = report.get('unrealized_pnl', 0)
            report_text.insert(tk.END, f"  Unrealized P/L: ₹{unrealized_pnl:,.2f}\n")
            
            if report.get('total_investment', 0) > 0:
                pnl_pct = (unrealized_pnl / report['total_investment']) * 100
                report_text.insert(tk.END, f"  Unrealized P/L (%): {pnl_pct:.2f}%\n")
            
            report_text.insert(tk.END, f"\nPositions:\n")
            for symbol, position in report.get('positions', {}).items():
                report_text.insert(tk.END, f"  {symbol}:\n")
                report_text.insert(tk.END, f"    Exchange: {position.get('exchange', '')}\n")
                report_text.insert(tk.END, f"    Quantity: {position.get('quantity', 0)}\n")
                report_text.insert(tk.END, f"    Avg Price: ₹{position.get('avg_price', 0):.2f}\n")
                report_text.insert(tk.END, f"    Current Price: ₹{position.get('current_price', 0):.2f}\n")
                report_text.insert(tk.END, f"    Market Value: ₹{position.get('market_value', 0):.2f}\n")
                report_text.insert(tk.END, f"    P/L: ₹{position.get('profit_loss', 0):.2f} ({position.get('profit_loss_pct', 0):.2f}%)\n")
                
                if 'stop_loss' in position:
                    report_text.insert(tk.END, f"    Stop Loss: ₹{position.get('stop_loss', 0):.2f}\n")
                if 'take_profit' in position:
                    report_text.insert(tk.END, f"    Take Profit: ₹{position.get('take_profit', 0):.2f}\n")
                
                report_text.insert(tk.END, f"\n")
            
            # Make text read-only
            report_text.config(state=tk.DISABLED)
            
            # Export button
            def export_report():
                from tkinter import filedialog
                filepath = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                )
                
                if filepath:
                    with open(filepath, 'w') as f:
                        f.write(report_text.get(1.0, tk.END))
                    messagebox.showinfo("Export", f"Report exported to {filepath}")
            
            export_button = ttk.Button(report_window, text="Export Report", command=export_report)
            export_button.pack(side=tk.RIGHT, padx=10, pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def _scan_watchlist(self):
        """Manually scan watchlist for trading signals"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        try:
            # Get watchlist
            watchlist = self.trading_system.get_watchlist()
            if not watchlist:
                messagebox.showinfo("Watchlist", "Watchlist is empty")
                return
            
            # Show progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Scanning Watchlist")
            progress_dialog.geometry("300x100")
            progress_dialog.resizable(False, False)
            progress_dialog.transient(self.root)
            progress_dialog.grab_set()
            
            ttk.Label(progress_dialog, text="Scanning watchlist for trading signals...").pack(padx=10, pady=5)
            
            progress_var = tk.DoubleVar()
            progress = ttk.Progressbar(progress_dialog, variable=progress_var, maximum=len(watchlist))
            progress.pack(fill=tk.X, padx=10, pady=5)
            
            status_var = tk.StringVar(value="Starting scan...")
            ttk.Label(progress_dialog, textvariable=status_var).pack(padx=10, pady=5)
            
            # Function to perform scan in a separate thread
            def do_scan():
                signals_found = 0
                
                for i, stock in enumerate(watchlist):
                    symbol = stock['symbol']
                    exchange = stock['exchange']
                    
                    # Update progress
                    progress_var.set(i + 1)
                    status_var.set(f"Analyzing {symbol}...")
                    progress_dialog.update()
                    
                    # Analyze stock
                    signal = self.trading_system._analyze_stock(exchange, symbol)
                    if signal and signal.get('action') == 'BUY':
                        self.trading_system.signals[symbol] = signal
                        signals_found += 1
                
                # Update signals tab
                self._refresh_signals()
                
                # Close progress dialog
                progress_dialog.destroy()
                
                # Show results
                messagebox.showinfo("Scan Results", f"Scan complete. Found {signals_found} trading signals.")
            
            # Start scan in a separate thread
            scan_thread = threading.Thread(target=do_scan)
            scan_thread.daemon = True
            scan_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan watchlist: {str(e)}")
    
    def _stock_screener(self):
        """Open stock screener window"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        # Create screener window
        screener_window = tk.Toplevel(self.root)
        screener_window.title("Stock Screener")
        screener_window.geometry("800x600")
        screener_window.resizable(True, True)
        screener_window.transient(self.root)
        
        # Create notebook for different screens
        notebook = ttk.Notebook(screener_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        momentum_frame = ttk.Frame(notebook)
        reversal_frame = ttk.Frame(notebook)
        fundamental_frame = ttk.Frame(notebook)
        
        notebook.add(momentum_frame, text="Momentum Screen")
        notebook.add(reversal_frame, text="Mean Reversion Screen")
        notebook.add(fundamental_frame, text="Fundamental Screen")
        
        # Momentum Screen
        momentum_params = ttk.LabelFrame(momentum_frame, text="Parameters")
        momentum_params.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(momentum_params, text="Min RSI:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        min_rsi_var = tk.IntVar(value=60)
        ttk.Entry(momentum_params, textvariable=min_rsi_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(momentum_params, text="Min Volume Increase (%):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        vol_increase_var = tk.IntVar(value=20)
        ttk.Entry(momentum_params, textvariable=vol_increase_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(momentum_params, text="Min Price Increase (%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        price_increase_var = tk.DoubleVar(value=1.5)
        ttk.Entry(momentum_params, textvariable=price_increase_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Exchanges to scan
        ttk.Label(momentum_params, text="Exchange:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        exchange_var = tk.StringVar(value="NSE")
        ttk.Combobox(momentum_params, textvariable=exchange_var, values=["NSE", "BSE"], state="readonly").grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Index to scan
        ttk.Label(momentum_params, text="Scan Index:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        index_var = tk.StringVar(value="NIFTY 500")
        ttk.Combobox(momentum_params, textvariable=index_var, values=["NIFTY 50", "NIFTY 100", "NIFTY 500"], state="readonly").grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Scan button
        def run_momentum_scan():
            messagebox.showinfo("Screener", "Momentum scan not implemented in this demo")
            # In a full implementation, this would scan stocks based on parameters
        
        ttk.Button(momentum_params, text="Run Scan", command=run_momentum_scan).grid(row=2, column=3, sticky=tk.E, padx=5, pady=5)
        
        # Results table
        ttk.Label(momentum_frame, text="Results:").pack(anchor=tk.W, padx=10, pady=5)
        
        # Create treeview for results
        columns = ('symbol', 'price', 'change_pct', 'volume', 'rsi', 'strength')
        
        momentum_tree = ttk.Treeview(momentum_frame, columns=columns, show='headings')
        
        momentum_tree.heading('symbol', text='Symbol')
        momentum_tree.heading('price', text='Price')
        momentum_tree.heading('change_pct', text='Change %')
        momentum_tree.heading('volume', text='Volume')
        momentum_tree.heading('rsi', text='RSI')
        momentum_tree.heading('strength', text='Strength')
        
        momentum_tree.column('symbol', width=80, anchor=tk.CENTER)
        momentum_tree.column('price', width=80, anchor=tk.CENTER)
        momentum_tree.column('change_pct', width=80, anchor=tk.CENTER)
        momentum_tree.column('volume', width=100, anchor=tk.CENTER)
        momentum_tree.column('rsi', width=80, anchor=tk.CENTER)
        momentum_tree.column('strength', width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(momentum_frame, orient=tk.VERTICAL, command=momentum_tree.yview)
        momentum_tree.configure(yscroll=scrollbar.set)
        
        momentum_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add dummy data for demonstration
        momentum_tree.insert('', tk.END, values=('RELIANCE', '₹2,450.00', '+2.3%', '1,245,678', '74.5', 'Strong'))
        momentum_tree.insert('', tk.END, values=('HDFCBANK', '₹1,620.50', '+1.8%', '987,123', '68.2', 'Moderate'))
        momentum_tree.insert('', tk.END, values=('TCS', '₹3,550.75', '+1.5%', '654,321', '65.7', 'Moderate'))
        
        # Similar structure for other screens (Mean Reversion and Fundamental)
        # For brevity, I'm only showing the momentum screen in detail
        
        ttk.Label(reversal_frame, text="Mean Reversion Screen - Not implemented in demo").pack(padx=10, pady=20)
        ttk.Label(fundamental_frame, text="Fundamental Screen - Not implemented in demo").pack(padx=10, pady=20)
    
    def _sell_selected_position(self):
        """Sell the position selected in the positions tree"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        selected = self.positions_tree.selection()
        if not selected:
            messagebox.showinfo("Positions", "No position selected")
            return
        
        # Get selected position
        item = self.positions_tree.item(selected[0])
        symbol = item['values'][0]
        exchange = item['values'][1]
        quantity = int(item['values'][2])
        
        # Confirm sell
        if messagebox.askyesno("Confirm", f"Sell {quantity} shares of {symbol}?"):
            try:
                order = self.trading_system.manual_sell(exchange, symbol, quantity)
                if order:
                    messagebox.showinfo("Sell", f"Sell order placed for {quantity} shares of {symbol}")
                    self._refresh_positions()
                else:
                    messagebox.showerror("Error", "Failed to place sell order")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to sell position: {str(e)}")
    
    def _set_stop_loss(self):
        """Set stop loss for selected position"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        selected = self.positions_tree.selection()
        if not selected:
            messagebox.showinfo("Positions", "No position selected")
            return
        
        # Get selected position
        item = self.positions_tree.item(selected[0])
        symbol = item['values'][0]
        current_price = float(item['values'][4].replace('₹', ''))
        
        # Get current stop loss
        current_stop = None
        if item['values'][8] != "N/A":
            current_stop = float(item['values'][8].replace('₹', ''))
        
        # Calculate default stop loss percentage
        default_pct = 3.0
        if current_stop:
            default_pct = round((1 - current_stop / current_price) * 100, 2)
        
        # Ask for stop loss percentage
        from tkinter import simpledialog
        stop_pct = simpledialog.askfloat(
            "Stop Loss",
            f"Enter stop loss percentage for {symbol}:",
            initialvalue=default_pct,
            minvalue=0.1,
            maxvalue=20.0
        )
        
        if stop_pct is None:
            return
        
        # Calculate stop loss price
        stop_price = current_price * (1 - stop_pct / 100)
        
        # Update position
        try:
            positions = self.trading_system.positions
            if symbol in positions:
                positions[symbol]['stop_loss'] = stop_price
                messagebox.showinfo("Stop Loss", f"Stop loss for {symbol} set to ₹{stop_price:.2f} ({stop_pct:.2f}%)")
                self._refresh_positions()
            else:
                messagebox.showerror("Error", f"Position {symbol} not found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set stop loss: {str(e)}")
    
    def _set_take_profit(self):
        """Set take profit for selected position"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        selected = self.positions_tree.selection()
        if not selected:
            messagebox.showinfo("Positions", "No position selected")
            return
        
        # Get selected position
        item = self.positions_tree.item(selected[0])
        symbol = item['values'][0]
        current_price = float(item['values'][4].replace('₹', ''))
        
        # Get current take profit
        current_tp = None
        if item['values'][9] != "N/A":
            current_tp = float(item['values'][9].replace('₹', ''))
        
        # Calculate default take profit percentage
        default_pct = 8.0
        if current_tp:
            default_pct = round((current_tp / current_price - 1) * 100, 2)
        
        # Ask for take profit percentage
        from tkinter import simpledialog
        tp_pct = simpledialog.askfloat(
            "Take Profit",
            f"Enter take profit percentage for {symbol}:",
            initialvalue=default_pct,
            minvalue=0.1,
            maxvalue=50.0
        )
        
        if tp_pct is None:
            return
        
        # Calculate take profit price
        tp_price = current_price * (1 + tp_pct / 100)
        
        # Update position
        try:
            positions = self.trading_system.positions
            if symbol in positions:
                positions[symbol]['take_profit'] = tp_price
                messagebox.showinfo("Take Profit", f"Take profit for {symbol} set to ₹{tp_price:.2f} ({tp_pct:.2f}%)")
                self._refresh_positions()
            else:
                messagebox.showerror("Error", f"Position {symbol} not found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set take profit: {str(e)}")
    
    def _buy_selected_stock(self):
        """Buy the stock selected in the watchlist"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        selected = self.watchlist_tree.selection()
        if not selected:
            messagebox.showinfo("Watchlist", "No stock selected")
            return
        
        # Get selected stock
        item = self.watchlist_tree.item(selected[0])
        symbol = item['values'][0]
        exchange = item['values'][1]
        
        # Show buy dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Buy {symbol}")
        dialog.geometry("300x150")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Quantity:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        quantity_var = tk.IntVar(value=1)
        ttk.Entry(dialog, textvariable=quantity_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        current_price = float(item['values'][2].replace('₹', ''))
        ttk.Label(dialog, text=f"Current Price: ₹{current_price:.2f}").grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)
        
        total_amount = tk.StringVar(value=f"₹{current_price:.2f}")
        ttk.Label(dialog, text="Total Amount:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        ttk.Label(dialog, textvariable=total_amount).grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Update total amount when quantity changes
        def update_total(*args):
            total = quantity_var.get() * current_price
            total_amount.set(f"₹{total:.2f}")
        
        quantity_var.trace('w', update_total)
        
        # Buy function
        def do_buy():
            if quantity_var.get() <= 0:
                messagebox.showerror("Error", "Please enter a valid quantity")
                return
            
            try:
                order = self.trading_system.manual_buy(exchange, symbol, quantity_var.get())
                if order:
                    messagebox.showinfo("Buy", f"Buy order placed for {quantity_var.get()} shares of {symbol}")
                    dialog.destroy()
                    self._refresh_positions()
                else:
                    messagebox.showerror("Error", "Failed to place buy order")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to buy stock: {str(e)}")
        
        ttk.Button(dialog, text="Buy", command=do_buy).grid(row=3, column=1, sticky=tk.E, padx=10, pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(row=3, column=0, sticky=tk.W, padx=10, pady=10)
    
    def _analyze_selected_stock(self):
        """Analyze the stock selected in the watchlist"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        selected = self.watchlist_tree.selection()
        if not selected:
            messagebox.showinfo("Watchlist", "No stock selected")
            return
        
        # Get selected stock
        item = self.watchlist_tree.item(selected[0])
        symbol = item['values'][0]
        exchange = item['values'][1]
        
        try:
            # Show analysis window
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title(f"Analysis: {symbol}")
            analysis_window.geometry("800x600")
            analysis_window.transient(self.root)
            
            # Create notebook for different analyses
            notebook = ttk.Notebook(analysis_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create tabs
            technical_frame = ttk.Frame(notebook)
            chart_frame = ttk.Frame(notebook)
            
            notebook.add(technical_frame, text="Technical Analysis")
            notebook.add(chart_frame, text="Price Chart")
            
            # Show loading message
            ttk.Label(technical_frame, text="Analyzing... Please wait").pack(padx=10, pady=20)
            ttk.Label(chart_frame, text="Loading chart... Please wait").pack(padx=10, pady=20)
            
            # Force UI update
            analysis_window.update()
            
            # Get historical data for analysis
            df = self.api.get_historical_data(exchange, symbol)
            if df.empty:
                messagebox.showerror("Error", f"No historical data available for {symbol}")
                analysis_window.destroy()
                return
            
            # Calculate technical indicators
            df['sma20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['sma50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['ema13'] = ta.trend.EMAIndicator(close=df['close'], window=13).ema_indicator()
            df['ema21'] = ta.trend.EMAIndicator(close=df['close'], window=21).ema_indicator()
            df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            
            # Clear loading message
            for widget in technical_frame.winfo_children():
                widget.destroy()
            for widget in chart_frame.winfo_children():
                widget.destroy()
            
            # Technical Analysis tab
            ttk.Label(technical_frame, text=f"Technical Analysis for {symbol}", font=('Arial', 14, 'bold')).pack(anchor=tk.W, padx=10, pady=5)
            
            # Get current price and indicators
            current_price = df['close'].iloc[-1]
            sma20 = df['sma20'].iloc[-1]
            sma50 = df['sma50'].iloc[-1]
            ema13 = df['ema13'].iloc[-1]
            ema21 = df['ema21'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # Current price and moving averages
            price_frame = ttk.LabelFrame(technical_frame, text="Price and Moving Averages")
            price_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(price_frame, text=f"Current Price: ₹{current_price:.2f}").grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
            ttk.Label(price_frame, text=f"20-day SMA: ₹{sma20:.2f} ({(current_price/sma20-1)*100:.2f}%)").grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
            ttk.Label(price_frame, text=f"50-day SMA: ₹{sma50:.2f} ({(current_price/sma50-1)*100:.2f}%)").grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
            ttk.Label(price_frame, text=f"13-day EMA: ₹{ema13:.2f} ({(current_price/ema13-1)*100:.2f}%)").grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
            ttk.Label(price_frame, text=f"21-day EMA: ₹{ema21:.2f} ({(current_price/ema21-1)*100:.2f}%)").grid(row=2, column=0, sticky=tk.W, padx=10, pady=2)
            
            # Oscillators
            oscillator_frame = ttk.LabelFrame(technical_frame, text="Oscillators")
            oscillator_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(oscillator_frame, text=f"RSI (14): {rsi:.2f}").grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
            ttk.Label(oscillator_frame, text=f"ATR (14): ₹{atr:.2f}").grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Signals
            signal_frame = ttk.LabelFrame(technical_frame, text="Trading Signals")
            signal_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Trend signals
            trend = "Uptrend" if current_price > sma50 and sma20 > sma50 else "Downtrend" if current_price < sma50 and sma20 < sma50 else "Sideways"
            trend_color = "green" if trend == "Uptrend" else "red" if trend == "Downtrend" else "black"
            
            ttk.Label(signal_frame, text=f"Trend: ").grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
            trend_label = ttk.Label(signal_frame, text=trend)
            trend_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
            
            # RSI signals
            rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            rsi_color = "red" if rsi_signal == "Overbought" else "green" if rsi_signal == "Oversold" else "black"
            
            ttk.Label(signal_frame, text=f"RSI Signal: ").grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
            rsi_label = ttk.Label(signal_frame, text=rsi_signal)
            rsi_label.grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
            
            # MA crossover
            ma_cross = "Bullish" if sma20 > sma50 and df['sma20'].iloc[-2] <= df['sma50'].iloc[-2] else "Bearish" if sma20 < sma50 and df['sma20'].iloc[-2] >= df['sma50'].iloc[-2] else "None"
            ma_color = "green" if ma_cross == "Bullish" else "red" if ma_cross == "Bearish" else "black"
            
            ttk.Label(signal_frame, text=f"MA Crossover: ").grid(row=2, column=0, sticky=tk.W, padx=10, pady=2)
            ma_label = ttk.Label(signal_frame, text=ma_cross)
            ma_label.grid(row=2, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Overall signal
            # Calculate a simple signal based on the above indicators
            if (trend == "Uptrend" and rsi < 70) or ma_cross == "Bullish":
                overall = "BUY"
                color = "green"
            elif (trend == "Downtrend" and rsi > 30) or ma_cross == "Bearish":
                overall = "SELL"
                color = "red"
            else:
                overall = "HOLD"
                color = "blue"
            
            ttk.Label(signal_frame, text=f"Overall Signal: ").grid(row=3, column=0, sticky=tk.W, padx=10, pady=2)
            overall_label = ttk.Label(signal_frame, text=overall)
            overall_label.grid(row=3, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Strategy recommendations
            strategy_frame = ttk.LabelFrame(technical_frame, text="Strategy Recommendations")
            strategy_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Entry recommendation
            entry_text = tk.Text(strategy_frame, wrap=tk.WORD, height=4, width=80)
            entry_text.pack(fill=tk.X, padx=10, pady=5)
            
            if overall == "BUY":
                entry_text.insert(tk.END, f"ENTRY STRATEGY: Consider buying {symbol} with a target price of ₹{current_price * 1.08:.2f} ({8}%) and stop loss at ₹{current_price * 0.97:.2f} ({3}%). Entry is favorable as the stock is in an uptrend with positive momentum indicators.")
            elif overall == "SELL":
                entry_text.insert(tk.END, f"EXIT STRATEGY: Consider selling {symbol} if already held. The stock is showing bearish signals with a downtrend and negative momentum indicators. If you want to short, consider a target of ₹{current_price * 0.92:.2f} ({8}% down) with a stop loss at ₹{current_price * 1.03:.2f} ({3}% up).")
            else:
                entry_text.insert(tk.END, f"NEUTRAL STRATEGY: {symbol} is currently in a neutral zone. Consider waiting for clearer signals before entering. If already holding the position, maintain stop loss at ₹{current_price * 0.97:.2f} ({3}%).")
            
            entry_text.config(state=tk.DISABLED)
            
            # Chart tab
            fig = plt.Figure(figsize=(10, 8), dpi=100)
            canvas = FigureCanvasTkAgg(fig, chart_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Price chart with moving averages
            ax1 = fig.add_subplot(211)
            ax1.plot(df.index[-60:], df['close'][-60:], label='Close Price')
            ax1.plot(df.index[-60:], df['sma20'][-60:], label='SMA 20')
            ax1.plot(df.index[-60:], df['sma50'][-60:], label='SMA 50')
            ax1.set_title(f'{symbol} Price Chart (Last 60 days)')
            ax1.set_ylabel('Price (₹)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # RSI subplot
            ax2 = fig.add_subplot(212, sharex=ax1)
            ax2.plot(df.index[-60:], df['rsi'][-60:], color='purple', label='RSI')
            ax2.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            ax2.axhline(y=30, color='g', linestyle='-', alpha=0.3)
            ax2.set_title('RSI Indicator')
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            fig.tight_layout()
            canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze stock: {str(e)}")
    
    def _execute_selected_signal(self):
        """Execute the signal selected in the signals tree"""
        if not self.trading_system:
            messagebox.showerror("Error", "Please log in first")
            return
        
        selected = self.signals_tree.selection()
        if not selected:
            messagebox.showinfo("Signals", "No signal selected")
            return
        
        # Get selected signal
        item = self.signals_tree.item(selected[0])
        symbol = item['values'][0]
        exchange = item['values'][1]
        action = item['values'][3]
        
        if action != "BUY":
            messagebox.showinfo("Signal", f"Only BUY signals can be executed manually")
            return
        
        # Confirm execution
        if messagebox.askyesno("Confirm", f"Execute {action} signal for {symbol}?"):
            try:
                # Show quantity dialog
                dialog = tk.Toplevel(self.root)
                dialog.title(f"Execute {action} Signal")
                dialog.geometry("300x150")
                dialog.resizable(False, False)
                dialog.transient(self.root)
                dialog.grab_set()
                
                ttk.Label(dialog, text="Quantity:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
                quantity_var = tk.IntVar(value=1)
                ttk.Entry(dialog, textvariable=quantity_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
                
                def execute_signal():
                    if quantity_var.get() <= 0:
                        messagebox.showerror("Error", "Please enter a valid quantity")
                        return
                    
                    if action == "BUY":
                        order = self.trading_system.manual_buy(exchange, symbol, quantity_var.get())
                    else:
                        order = None
                    
                    if order:
                        messagebox.showinfo("Signal", f"{action} signal executed for {symbol}")
                        dialog.destroy()
                        self._refresh_positions()
                        
                        # Remove signal after execution
                        if symbol in self.trading_system.signals:
                            del self.trading_system.signals[symbol]
                        self._refresh_signals()
                    else:
                        messagebox.showerror("Error", f"Failed to execute {action} signal")
                
                ttk.Button(dialog, text="Execute", command=execute_signal).grid(row=1, column=1, sticky=tk.E, padx=10, pady=10)
                ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to execute signal: {str(e)}")
    
    def _show_position_menu(self, event):
        """Show context menu for positions"""
        item = self.positions_tree.identify_row(event.y)
        if item:
            self.positions_tree.selection_set(item)
            self.position_menu.post(event.x_root, event.y_root)
    
    def _show_watchlist_menu(self, event):
        """Show context menu for watchlist"""
        item = self.watchlist_tree.identify_row(event.y)
        if item:
            self.watchlist_tree.selection_set(item)
            self.watchlist_menu.post(event.x_root, event.y_root)
    
    def _show_signal_menu(self, event):
        """Show context menu for signals"""
        item = self.signals_tree.identify_row(event.y)
        if item:
            self.signals_tree.selection_set(item)
            self.signal_menu.post(event.x_root, event.y_root)
    
    def _schedule_updates(self):
        """Schedule periodic UI updates"""
        def update():
            if self.trading_system and self.trading_system.running:
                self._refresh_all()
            elif self.api:
                # Even if the trading system is not running, update the UI if logged in
                self._refresh_positions()
                self._refresh_watchlist()
            
            # Schedule the next update
            self.root.after(30000, update)  # Update every 30 seconds
        
        # Start the update cycle
        self.root.after(30000, update)
    
    def _exit_application(self):
        """Exit the application"""
        if self.trading_system and self.trading_system.running:
            if not messagebox.askyesno("Exit", "Trading system is running. Are you sure you want to exit?"):
                return
            
            # Stop the trading system
            self.trading_system.stop()
        
        self.root.destroy()


def main():
    """Main function to start the application"""
    root = tk.Tk()
    app = TradingSystemUI(root)
    root.protocol("WM_DELETE_WINDOW", app._exit_application)
    root.mainloop()


if __name__ == "__main__":
    main()
    