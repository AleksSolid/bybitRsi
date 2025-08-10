import numpy as np
import time
import logging
import requests
from datetime import datetime
from pybit.unified_trading import HTTP
from config import API_KEY, SECRET_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_botV2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.client = HTTP(api_key=API_KEY, api_secret=SECRET_KEY)
        self.symbol = "XRPUSDT"
        self.category = "linear"
        self.trade_qty = "10"
        self.rsi_window = 14
        self.klines_limit = 200
        self.rsi_buy_threshold = 35
        self.rsi_sell_threshold = 65
        self.stop_loss_pct = 2.0
        self.take_profit_pct = 2.0
        self.trailing_stop_pct = 0.5
        self.max_retries = 3
        self.retry_delay = 5
        self.position_side = None
        self.entry_price = None
        self.highest_price = None
        self.lowest_price = None
        self.trailing_activated = False
        self.position_qty = 0
        self.telegram_token = TELEGRAM_TOKEN
        self.telegram_chat_id = TELEGRAM_CHAT_ID
        
        # Для постепенного закрытия позиции
        self.profit_targets = [
            {'percent': 0.5, 'tp_percent': 0.5, 'close_ratio': 0.5, 'move_sl': True},
            {'percent': 0.8, 'tp_percent': 0.8, 'close_ratio': 0.3, 'move_sl': False},
            {'percent': 1.0, 'tp_percent': 1.0, 'close_ratio': 0.2, 'move_sl': False}
        ]
        self.achieved_targets = []

    def send_telegram_notification(self, message):
        """Send notification to Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not set. Notification not sent.")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info("Telegram notification sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False

    def calculate_rsi(self, prices):
        """Calculate RSI with numpy for better performance"""
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gain[:self.rsi_window])
        avg_loss = np.mean(loss[:self.rsi_window])
        
        # Avoid division by zero
        if avg_loss == 0:
            return np.full(len(prices), 100.0)
            
        rs = avg_gain / avg_loss
        rsi = np.zeros(len(prices))
        rsi[self.rsi_window] = 100.0 - (100.0 / (1.0 + rs))
        
        # Calculate subsequent RSI values
        for i in range(self.rsi_window + 1, len(prices)):
            delta = deltas[i-1]
            gain_val = delta if delta > 0 else 0.0
            loss_val = -delta if delta < 0 else 0.0
            
            avg_gain = (avg_gain * (self.rsi_window - 1) + gain_val) / self.rsi_window
            avg_loss = (avg_loss * (self.rsi_window - 1) + loss_val) / self.rsi_window
            
            # Handle case where avg_loss becomes zero
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
                
        return rsi

    def sleep_to_next_candle(self):
        """Sleep until the next 15-minute candle starts"""
        now = datetime.now()
        current_minute = now.minute
        minutes_past = current_minute % 15
        minutes_to_wait = 15 - minutes_past if minutes_past > 0 else 0
        
        # Calculate seconds to sleep
        seconds_to_wait = (minutes_to_wait * 60) - now.second
        if seconds_to_wait <= 0:
            seconds_to_wait = 900
            
        logger.info(f'Sleeping for {seconds_to_wait} seconds until next candle')
        time.sleep(seconds_to_wait)

    def has_open_position(self):
        """Check if there's an open position with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.get_positions(
                    category=self.category,
                    symbol=self.symbol
                )
                
                if not isinstance(response, dict) or 'result' not in response:
                    logger.error(f"Invalid API response on attempt {attempt + 1}: {response}")
                    continue
                    
                positions = response['result'].get('list', [])
                
                if isinstance(positions, list):
                    for pos in positions:
                        if float(pos['size']) > 0:
                            self.position_side = pos['side'].lower()
                            self.entry_price = float(pos['avgPrice'])
                            self.position_qty = float(pos['size'])
                            return True
                    # Reset tracking variables
                    self.position_side = None
                    self.entry_price = None
                    self.highest_price = None
                    self.lowest_price = None
                    self.trailing_activated = False
                    self.position_qty = 0
                    self.achieved_targets = []  # Reset achieved targets
                    return False
                
                logger.error(f"Unexpected positions format on attempt {attempt + 1}: {positions}")
                
            except Exception as e:
                logger.error(f"Position check error on attempt {attempt + 1}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        return False

    def update_trailing_stop(self, current_price):
        """Update trailing stop levels based on current price"""
        if not self.entry_price:
            return False
            
        activation_threshold = 0.005 * self.entry_price
        
        if self.position_side == 'buy':
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
                
            if current_price > self.entry_price + activation_threshold:
                self.trailing_activated = True
                
            if self.trailing_activated:
                if current_price <= self.highest_price * (1 - self.trailing_stop_pct/100):
                    logger.info(f"Trailing stop triggered at {current_price:.2f} ({(current_price - self.entry_price)/self.entry_price*100:.2f}%)")
                    return 'sell'
                
        elif self.position_side == 'sell':
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
                
            if current_price < self.entry_price - activation_threshold:
                self.trailing_activated = True
                
            if self.trailing_activated:
                if current_price >= self.lowest_price * (1 + self.trailing_stop_pct/100):
                    logger.info(f"Trailing stop triggered at {current_price:.2f} ({(self.entry_price - current_price)/self.entry_price*100:.2f}%)")
                    return 'buy'
        
        return False

    def check_sl_tp(self, current_price):
        """Check if stop loss, take profit or trailing stop should be triggered"""
        if not self.entry_price or not self.position_side:
            return False
            
        trailing_action = self.update_trailing_stop(current_price)
        if trailing_action:
            return trailing_action
            
        price_diff_pct = (current_price - self.entry_price) / self.entry_price * 100
        if self.position_side == 'buy':
            if price_diff_pct <= -self.stop_loss_pct:
                logger.info(f"Stop loss triggered at {current_price:.2f} ({price_diff_pct:.2f}%)")
                return 'sell'
            elif price_diff_pct >= self.take_profit_pct:
                logger.info(f"Take profit triggered at {current_price:.2f} ({price_diff_pct:.2f}%)")
                return 'sell'
        elif self.position_side == 'sell':
            if price_diff_pct >= self.stop_loss_pct:
                logger.info(f"Stop loss triggered at {current_price:.2f} ({price_diff_pct:.2f}%)")
                return 'buy'
            elif price_diff_pct <= -self.take_profit_pct:
                logger.info(f"Take profit triggered at {current_price:.2f} ({price_diff_pct:.2f}%)")
                return 'buy'
        return False

    def get_klines_data(self):
        """Fetch klines data with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.get_kline(
                    category=self.category,
                    symbol=self.symbol,
                    interval="15",
                    limit=self.klines_limit
                )
                klines = sorted(response['result']['list'], key=lambda x: int(x[0]))
                return klines
            except Exception as e:
                logger.error(f"Error fetching klines on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return None

    def get_current_price(self):
        """Get current market price"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.get_tickers(
                    category=self.category,
                    symbol=self.symbol
                )
                return float(response['result']['list'][0]['lastPrice'])
            except Exception as e:
                logger.error(f"Error getting current price on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return None

    def place_trade_order(self, side, qty=None):
        """Place a trade order with retry logic"""
        if qty is None:
            qty = self.trade_qty
        else:
            # Convert to string if number is provided
            if isinstance(qty, (float, int)):
                qty = str(qty)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting {side} order for {qty} (attempt {attempt + 1})")
                response = self.client.place_order(
                    category=self.category,
                    symbol=self.symbol,
                    side=side,
                    orderType="Market",
                    qty=qty
                )
                logger.info(f"Successfully executed {side} order. Response: {response}")
                return True
            except Exception as e:
                logger.error(f"Error placing {side} order on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return False

    def check_profit_targets(self, current_price, last_rsi):
        """Check and execute profit-taking targets"""
        if not self.position_side or not self.entry_price or self.position_qty <= 0:
            return False
        
        # Calculate current profit percentage
        if self.position_side == 'buy':
            profit_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:  # sell
            profit_pct = (self.entry_price - current_price) / self.entry_price * 100
        
        # Check each profit target
        for target in self.profit_targets:
            target_percent = target['percent'] * self.take_profit_pct
            
            # Skip if target already achieved
            if target['percent'] in self.achieved_targets:
                continue
                
            # Check if target reached
            if profit_pct >= target_percent:
                try:
                    # Calculate quantity to close
                    close_qty = self.position_qty * target['close_ratio']
                    
                    # For the last target, close entire remaining position
                    if target['percent'] == 1.0:
                        close_qty = self.position_qty
                    
                    # Determine close side (opposite to position)
                    close_side = 'Sell' if self.position_side == 'buy' else 'Buy'
                    
                    # Execute partial close
                    if self.place_trade_order(close_side, qty=close_qty):
                        # Update position quantity
                        self.position_qty -= close_qty
                        self.achieved_targets.append(target['percent'])
                        
                        # Move stop loss to break-even if required
                        if target['move_sl']:
                            self.stop_loss_pct = 0
                            logger.info(f"Stop loss moved to break-even at {self.entry_price:.4f}")
                        
                        # Prepare notification
                        action_type = "TAKE PROFIT" if target['percent'] < 1.0 else "FULL CLOSE"
                        message = (
                            f"🎯 *{action_type}*\n\n"
                            f"*Target:* {target['percent']*100:.0f}% of TP\n"
                            f"*Action:* {close_side}\n"
                            f"*Symbol:* {self.symbol}\n"
                            f"*Quantity:* {close_qty:.2f}\n"
                            f"*Price:* {current_price:.6f}\n"
                            f"*Profit:* {profit_pct:.2f}%\n"
                            f"*Remaining Qty:* {self.position_qty:.2f}\n"
                            f"*RSI:* {last_rsi:.2f}\n"
                            f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        self.send_telegram_notification(message)
                        
                        # Reset if position fully closed
                        if self.position_qty <= 0:
                            self.position_side = None
                            self.entry_price = None
                            self.highest_price = None
                            self.lowest_price = None
                            self.trailing_activated = False
                            self.achieved_targets = []
                            logger.info("Position fully closed by profit targets")
                        return True
                except Exception as e:
                    logger.error(f"Error executing profit target {target['percent']}: {str(e)}")
        return False

    def run(self):
        """Main trading loop"""
        logger.info("Starting trading bot")
        
        # Send startup notification
        startup_msg = (
            "🤖 *Trading Bot Started*\n\n"
            f"*Symbol:* {self.symbol}\n"
            f"*Strategy:* RSI ({self.rsi_buy_threshold}/{self.rsi_sell_threshold})\n"
            f"*Quantity:* {self.trade_qty}\n"
            f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_telegram_notification(startup_msg)
        
        # Test API connection
        try:
            logger.info("Testing API connection...")
            test_response = self.client.get_tickers(category=self.category, symbol=self.symbol)
            logger.info("API connection OK!")
        except Exception as e:
            logger.error(f"API connection FAILED: {e}")
            error_msg = f"❌ *API Connection Failed*\n\nError: {str(e)}"
            self.send_telegram_notification(error_msg)
            return

        while True:
            self.sleep_to_next_candle()
            
            try:
                # Get current market data
                current_price = self.get_current_price()
                if not current_price:
                    logger.error("Failed to get current price after retries")
                    continue
                
                klines = self.get_klines_data()
                if not klines:
                    logger.error("Failed to get klines data after retries")
                    continue
                
                # Extract close prices
                close_prices = np.array([float(k[4]) for k in klines], dtype=float)
                logger.info(f"Fetched {len(close_prices)} klines. Price range: {np.min(close_prices):.4f}-{np.max(close_prices):.4f}")
                
                # Calculate RSI
                rsi_values = self.calculate_rsi(close_prices)
                last_rsi = rsi_values[-1]
                logger.info(f'Last RSI: {last_rsi:.2f}, Current Price: {current_price:.4f}')
                
                # Check for open position
                if self.has_open_position():
                    logger.info(f"Open position: {self.position_side} at {self.entry_price:.4f}, Size: {self.position_qty}")
                    
                    # 1. Проверяем цели по прибыли
                    self.check_profit_targets(current_price, last_rsi)
                    
                    # Если позиция осталась открытой после проверки целей
                    if self.position_side:
                        # 2. Проверяем SL/TP/Trailing stop
                        action = self.check_sl_tp(current_price)
                        if action:
                            logger.info(f"Closing position with {action} order")
                            if self.place_trade_order(action.capitalize(), qty=self.position_qty):
                                # Save closing details for notification
                                close_qty = self.position_qty
                                close_price = current_price
                                
                                # Reset position variables
                                self.position_side = None
                                self.entry_price = None
                                self.highest_price = None
                                self.lowest_price = None
                                self.trailing_activated = False
                                self.position_qty = 0
                                self.achieved_targets = []
                                
                                # Send closing notification
                                message = (
                                    f"🔻 *POSITION CLOSED*\n\n"
                                    f"*Action:* {action.upper()}\n"
                                    f"*Symbol:* {self.symbol}\n"
                                    f"*Quantity:* {close_qty}\n"
                                    f"*Exit Price:* {close_price:.6f}\n"
                                    f"*RSI:* {last_rsi:.2f}\n"
                                    f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                )
                                self.send_telegram_notification(message)
                        else:
                            # 3. Проверяем возможность усреднения
                            if (self.position_side == 'buy' and last_rsi <= self.rsi_buy_threshold) or \
                               (self.position_side == 'sell' and last_rsi >= self.rsi_sell_threshold):
                                logger.info(f"Averaging {self.position_side} position")
                                if self.place_trade_order(self.position_side.capitalize()):
                                    # Calculate new average price
                                    new_qty = self.position_qty + float(self.trade_qty)
                                    new_avg_price = (self.position_qty * self.entry_price + 
                                                    float(self.trade_qty) * current_price) / new_qty
                                    
                                    # Update position tracking
                                    self.position_qty = new_qty
                                    self.entry_price = new_avg_price
                                    
                                    # Reset profit targets
                                    self.achieved_targets = []
                                    
                                    # Update trailing stop levels
                                    if self.position_side == 'buy':
                                        self.highest_price = max(self.highest_price, current_price) \
                                            if self.highest_price is not None else current_price
                                    else:
                                        self.lowest_price = min(self.lowest_price, current_price) \
                                            if self.lowest_price is not None else current_price
                                    
                                    # Send averaging notification
                                    message = (
                                        f"📊 *POSITION AVERAGED*\n\n"
                                        f"*Action:* {self.position_side.upper()} Averaging\n"
                                        f"*Symbol:* {self.symbol}\n"
                                        f"*Added Quantity:* {self.trade_qty}\n"
                                        f"*New Average Price:* {new_avg_price:.6f}\n"
                                        f"*Total Quantity:* {new_qty:.2f}\n"
                                        f"*Current RSI:* {last_rsi:.2f}\n"
                                        f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                    )
                                    self.send_telegram_notification(message)
                            else:
                                logger.info("No averaging signal detected")
                else:
                    # No open position - check new signals
                    logger.info("No open position - checking for signals")
                    if last_rsi <= self.rsi_buy_threshold:
                        logger.info('BUY SIGNAL detected')
                        if self.place_trade_order("Buy"):
                            self.position_side = 'buy'
                            self.entry_price = current_price
                            self.position_qty = float(self.trade_qty)
                            self.highest_price = current_price
                            self.achieved_targets = []  # Reset profit targets
                            
                            # Send opening notification
                            message = (
                                f"🚀 *NEW POSITION ENTRY*\n\n"
                                f"*Action:* BUY\n"
                                f"*Symbol:* {self.symbol}\n"
                                f"*Quantity:* {self.trade_qty}\n"
                                f"*Entry Price:* {current_price:.6f}\n"
                                f"*RSI:* {last_rsi:.2f}\n"
                                f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            )
                            self.send_telegram_notification(message)
                    elif last_rsi >= self.rsi_sell_threshold:
                        logger.info('SELL SIGNAL detected')
                        if self.place_trade_order("Sell"):
                            self.position_side = 'sell'
                            self.entry_price = current_price
                            self.position_qty = float(self.trade_qty)
                            self.lowest_price = current_price
                            self.achieved_targets = []  # Reset profit targets
                            
                            # Send opening notification
                            message = (
                                f"🚀 *NEW POSITION ENTRY*\n\n"
                                f"*Action:* SELL\n"
                                f"*Symbol:* {self.symbol}\n"
                                f"*Quantity:* {self.trade_qty}\n"
                                f"*Entry Price:* {current_price:.6f}\n"
                                f"*RSI:* {last_rsi:.2f}\n"
                                f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            )
                            self.send_telegram_notification(message)
                    else:
                        logger.info("No trading signal detected")
                    
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                error_msg = (
                    "⚠️ *Trading Bot Error*\n\n"
                    f"*Error:* {str(e)}\n"
                    f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.send_telegram_notification(error_msg)
                time.sleep(900)

if __name__ == '__main__':
    bot = TradingBot()
    bot.run()