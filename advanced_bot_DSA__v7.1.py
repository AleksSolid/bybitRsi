import time
import logging
import threading
import signal
import requests
import numpy as np
from datetime import datetime
from pybit.unified_trading import HTTP
from config import API_KEY, SECRET_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

# Optional: RSI identical to TradingView (pandas+ta); falls back to numpy if not installed
try:
    import pandas as pd
    import ta
    HAS_TA = True
except Exception:
    HAS_TA = False

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_GPT_V2.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        # If your pybit version supports timeout parameter, you can pass timeout=10
        self.client = HTTP(api_key=API_KEY, api_secret=SECRET_KEY)

        # Instrument/strategy
        self.symbol = "XRPUSDT"
        self.category = "linear"
        self.trade_qty = "10"
        self.rsi_window = 21
        self.klines_limit = 200
        self.rsi_buy_threshold = 35
        self.rsi_sell_threshold = 65

        # Risk (percent values)
        self.base_stop_loss_pct = 2.0
        self.current_stop_loss_pct = self.base_stop_loss_pct
        self.take_profit_pct = 3.0
        self.trailing_stop_pct = 1.0

        # Retry policy
        self.max_retries = 3
        self.retry_delay = 5

        # Position state
        self.position_side = None          # 'buy' | 'sell'
        self.entry_price = None
        self.position_qty = 0.0
        self.highest_price = None
        self.lowest_price = None
        self.trailing_activated = False
        self.breakeven_active = False

        # Exchange TP/SL bookkeeping
        self.exchange_tp_price = None
        self.exchange_sl_price = None

        # DCA
        self.max_dca_steps = 2
        self.dca_count = 0
        self.max_position_qty = None  # optional cap

        # Telegram
        self.telegram_token = TELEGRAM_TOKEN
        self.telegram_chat_id = TELEGRAM_CHAT_ID
        self.allowed_chat_id = None
        self.allowed_chat_username = None
        self._notify_chat_id = None
        if self.telegram_chat_id:
            s = str(self.telegram_chat_id).strip()
            if (s.startswith("-") and s[1:].isdigit()) or s.isdigit():
                self.allowed_chat_id = int(s)
            else:
                self.allowed_chat_username = s.lstrip("@").lower()

        # Partial ladder (kept optional; main TP/SL are on-exchange)
        self.profit_targets = [
            {"percent": 0.5, "tp_percent": 0.5, "close_ratio": 0.5, "move_sl": True},
            {"percent": 0.8, "tp_percent": 0.8, "close_ratio": 0.3, "move_sl": False},
            {"percent": 1.0, "tp_percent": 1.0, "close_ratio": 0.2, "move_sl": False},
        ]
        self.achieved_targets = []

        # Emergency
        self.enable_intracandle_guard = True
        self.emergency_sl_pct = 2.0

        # Concurrency
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self._state_lock = threading.Lock()
        self.trading_thread = None

        # Telegram updates
        self._tg_offset = None
        self._tg_thread = None

        # Cooldown after series
        self.consecutive_sl = 0
        self.consecutive_losses = 0
        self.cooldown_until_ts = 0.0
        self.cooldown_hours = 6

    # ---------- Utils ----------
    @staticmethod
    def _format_qty(qty: float) -> str:
        s = f"{qty:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def _sleep(self, seconds: float):
        self.stop_event.wait(timeout=max(0, seconds))

    def in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until_ts

    def _enter_cooldown(self, reason: str):
        self.cooldown_until_ts = time.time() + self.cooldown_hours * 3600
        self.send_telegram_notification(
            f"🧊 Cooldown {self.cooldown_hours}h started ({reason}). No entries/DCA during cooldown.",
            markdown=False
        )
        logger.info(f"Cooldown started for {self.cooldown_hours}h due to {reason}")

    def _update_counters_on_close(self, reason: str, close_price: float):
        if not self.entry_price or not self.position_side:
            pnl_pct = 0.0
        else:
            if self.position_side == "buy":
                pnl_pct = (close_price - self.entry_price) / self.entry_price * 100.0
            else:
                pnl_pct = (self.entry_price - close_price) / self.entry_price * 100.0

        if reason in ("SL", "EMERGENCY_SL"):
            self.consecutive_sl += 1
        else:
            self.consecutive_sl = 0

        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        logger.info(f"Close reason={reason}, pnl={pnl_pct:.2f}% (SLx{self.consecutive_sl}, Lossx{self.consecutive_losses})")

        if self.consecutive_sl >= 3:
            self.consecutive_sl = 0
            self._enter_cooldown("3 consecutive SL")
        elif self.consecutive_losses >= 3:
            self.consecutive_losses = 0
            self._enter_cooldown("3 consecutive losing trades")

    # ---------- Telegram ----------
    def send_telegram_notification(self, message: str, markdown: bool = True):
        if not self.telegram_token:
            return False
        target_chat = self._notify_chat_id or self.allowed_chat_id or self.telegram_chat_id
        if not target_chat:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": target_chat,
                "text": message,
                "disable_web_page_preview": True,
                **({"parse_mode": "Markdown"} if markdown else {}),
            }
            r = requests.post(url, json=payload, timeout=10)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def _telegram_command_worker(self):
        logger.info("Telegram command listener started")
        try:
            requests.get(f"https://api.telegram.org/bot{self.telegram_token}/deleteWebhook", timeout=10)
        except Exception:
            pass
        base_url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
        while True:
            try:
                params = {"timeout": 25}
                if self._tg_offset is not None:
                    params["offset"] = self._tg_offset + 1
                resp = requests.get(base_url, params=params, timeout=30)
                if resp.status_code != 200:
                    time.sleep(2); continue
                data = resp.json()
                if not data.get("ok", False):
                    time.sleep(2); continue

                for upd in data.get("result", []):
                    self._tg_offset = upd["update_id"]
                    msg = (upd.get("message") or upd.get("edited_message")
                           or upd.get("channel_post") or upd.get("edited_channel_post"))
                    if not msg:
                        continue

                    chat = msg.get("chat", {})
                    chat_id = chat.get("id")
                    chat_username = (chat.get("username") or "").lower()
                    text = (msg.get("text") or msg.get("caption") or "")
                    entities = (msg.get("entities") or msg.get("caption_entities") or [])

                    if self.allowed_chat_id is not None or self.allowed_chat_username is not None:
                        allowed = ((self.allowed_chat_id is not None and chat_id == self.allowed_chat_id)
                                   or (self.allowed_chat_username is not None and chat_username == self.allowed_chat_username))
                        if not allowed:
                            continue
                    else:
                        self.allowed_chat_id = chat_id
                        self.allowed_chat_username = chat_username
                        self._notify_chat_id = chat_id
                        self.send_telegram_notification(f"🔗 Bound to chat {chat_id}")

                    self._notify_chat_id = chat_id

                    cmd = ""
                    for ent in entities:
                        if ent.get("type") == "bot_command":
                            start = ent.get("offset", 0); length = ent.get("length", 0)
                            cmd = text[start:start+length]
                            break
                    if not cmd:
                        parts = text.strip().split()
                        cmd = parts[0] if parts else ""
                    cmd = cmd.lstrip("/").split("@")[0].lower()

                    if cmd == "ping":
                        self.send_telegram_notification("pong")
                    elif cmd == "stop":
                        ok = self.stop_trading()
                        self.send_telegram_notification("⏹ Stopping..." if ok else "ℹ Already stopped")
                    elif cmd == "start":
                        self.pause_event.clear()
                        ok = self.start_trading()
                        self.send_telegram_notification("▶️ Trading started" if ok else "ℹ Already running")
                    elif cmd == "pause":
                        self.pause_event.set()
                        self.send_telegram_notification("⏸ Trading paused (entries/DCA disabled)")
                    elif cmd == "resume":
                        self.pause_event.clear()
                        self.send_telegram_notification("▶️ Trading resumed")
                    elif cmd == "status":
                        running = self.is_trading()
                        paused = self.pause_event.is_set()
                        cd = self.in_cooldown()
                        left = max(0, int(self.cooldown_until_ts - time.time())) if cd else 0
                        left_h = f"{left//3600}h{(left%3600)//60}m" if cd else "0"
                        pos = f"{self.position_side or '-'} qty={self.position_qty:.6f} entry={self.entry_price if self.entry_price else '-'}"
                        self.send_telegram_notification(
                            f"ℹ Status: {'running' if running else 'stopped'} | paused: {'yes' if paused else 'no'} | "
                            f"cooldown: {'yes' if cd else 'no'} ({left_h})\nPosition: {pos}",
                            markdown=False
                        )
                    elif cmd == "config":
                        reply = self._handle_config_command(text)
                        self.send_telegram_notification(reply, markdown=False)
                    elif cmd in ("size", "qty"):
                        tokens = (text or "").strip().split()
                        qty_arg = tokens[1] if len(tokens) >= 2 else None
                        if not qty_arg:
                            self.send_telegram_notification("Usage: /size 10  (или /qty 12.5)", markdown=False)
                        else:
                            try:
                                new_qty_num = float(qty_arg)
                                if new_qty_num <= 0:
                                    raise ValueError("qty must be > 0")
                                new_qty_str = self._format_qty(new_qty_num)
                                with self._state_lock:
                                    self.trade_qty = new_qty_str
                                self.send_telegram_notification(
                                    f"✅ trade_qty set to {new_qty_str}\n(применится к новым входам и DCA; текущая позиция не изменится)",
                                    markdown=False
                                )
                            except Exception:
                                self.send_telegram_notification(f"❌ invalid size: {qty_arg}. Example: /size 10", markdown=False)
            except Exception as e:
                logger.error(f"Telegram listener error: {e}")
                time.sleep(3)

    def start_telegram_listener(self):
        if not self.telegram_token:
            logger.warning("Telegram token not set; listener disabled")
            return
        if self._tg_thread and self._tg_thread.is_alive():
            return
        self._tg_thread = threading.Thread(target=self._telegram_command_worker, daemon=True)
        self._tg_thread.start()

    def _handle_config_command(self, full_text: str) -> str:
        try:
            parts = full_text.strip().split()
            if len(parts) == 1:
                return self._config_snapshot()
            kv_pairs = parts[1:]
            mapping = {
                "symbol": ("symbol", str),
                "trade_qty": ("trade_qty", str),
                "rsi_window": ("rsi_window", int),
                "rsi_buy_threshold": ("rsi_buy_threshold", float),
                "rsi_sell_threshold": ("rsi_sell_threshold", float),
                "base_stop_loss_pct": ("base_stop_loss_pct", float),
                "take_profit_pct": ("take_profit_pct", float),
                "trailing_stop_pct": ("trailing_stop_pct", float),
                "max_dca_steps": ("max_dca_steps", int),
                "emergency_sl_pct": ("emergency_sl_pct", float),
            }
            applied = []
            with self._state_lock:
                for pair in kv_pairs:
                    if "=" not in pair:
                        continue
                    k, v = pair.split("=", 1)
                    k = k.strip().lower(); v = v.strip()
                    if k not in mapping:
                        continue
                    attr, caster = mapping[k]
                    cast_val = caster(v)
                    setattr(self, attr, cast_val)
                    applied.append(f"{k}={cast_val}")
            if not applied:
                return f"⚙️ No valid keys found. Allowed: {', '.join(mapping.keys())}"
            return "✅ Applied:\n" + "\n".join(applied) + "\n\n" + self._config_snapshot()
        except Exception as e:
            return f"❌ Config error: {e}"

    def _config_snapshot(self) -> str:
        return (
            "⚙️ Current config:\n"
            f"symbol={self.symbol}\n"
            f"trade_qty={self.trade_qty}\n"
            f"rsi_window={self.rsi_window}\n"
            f"rsi_buy_threshold={self.rsi_buy_threshold}\n"
            f"rsi_sell_threshold={self.rsi_sell_threshold}\n"
            f"base_stop_loss_pct={self.base_stop_loss_pct}\n"
            f"take_profit_pct={self.take_profit_pct}\n"
            f"trailing_stop_pct={self.trailing_stop_pct}\n"
            f"max_dca_steps={self.max_dca_steps}\n"
            f"emergency_sl_pct={self.emergency_sl_pct}"
        )

    # ---------- Indicators ----------
    def calculate_rsi(self, prices):
        if HAS_TA:
            s = pd.Series(prices, dtype="float64")
            rsi = ta.momentum.RSIIndicator(close=s, window=self.rsi_window, fillna=False).rsi()
            return rsi.to_numpy(dtype="float64")
        # fallback (numpy)
        prices = np.asarray(prices, dtype=float)
        n = self.rsi_window
        length = len(prices)
        rsi = np.full(length, np.nan)
        if length <= n:
            return rsi
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:n]); avg_loss = np.mean(losses[:n])
        if avg_loss == 0 and avg_gain == 0:
            rsi[n] = 50.0
        elif avg_loss == 0:
            rsi[n] = 100.0
        else:
            rs = avg_gain / avg_loss; rsi[n] = 100.0 - (100.0 / (1.0 + rs))
        for i in range(n + 1, length):
            d = deltas[i - 1]
            gain_val = d if d > 0 else 0.0
            loss_val = -d if d < 0 else 0.0
            avg_gain = (avg_gain * (n - 1) + gain_val) / n
            avg_loss = (avg_loss * (n - 1) + loss_val) / n
            if avg_loss == 0 and avg_gain == 0:
                rsi[i] = 50.0
            elif avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss; rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    # ---------- Exchange helpers ----------
    def get_klines_data(self, interval="15"):
        for attempt in range(self.max_retries):
            if self.stop_event.is_set():
                return None
            try:
                response = self.client.get_kline(
                    category=self.category, symbol=self.symbol, interval=interval, limit=self.klines_limit
                )
                if isinstance(response, dict) and "result" in response:
                    lst = response["result"].get("list") or []
                    if isinstance(lst, list) and len(lst) > 0:
                        return sorted(lst, key=lambda x: int(x[0]))
                logger.error(f"Empty/invalid klines (interval={interval}) on attempt {attempt + 1}: {response}")
            except Exception as e:
                logger.error(f"Error fetching klines (interval={interval}) on attempt {attempt + 1}: {e}")
            if attempt < self.max_retries - 1:
                self._sleep(self.retry_delay)
        return None

    def get_current_price(self):
        for attempt in range(self.max_retries):
            if self.stop_event.is_set():
                return None
            try:
                resp = self.client.get_tickers(category=self.category, symbol=self.symbol)
                if isinstance(resp, dict) and "result" in resp:
                    lst = resp["result"].get("list") or []
                    if lst:
                        return float(lst[0]["lastPrice"])
                logger.error(f"Empty/invalid ticker on attempt {attempt + 1}: {resp}")
            except Exception as e:
                logger.error(f"Error getting current price on attempt {attempt + 1}: {e}")
            if attempt < self.max_retries - 1:
                self._sleep(self.retry_delay)
        return None

    def place_trade_order(self, side: str, qty=None, reduce_only: bool = False):
        if qty is None:
            qty_str = str(self.trade_qty)
        else:
            qty_str = self._format_qty(float(qty)) if isinstance(qty, (float, int)) else str(qty)
        for attempt in range(self.max_retries):
            if self.stop_event.is_set():
                return False
            try:
                logger.info(f"Attempting {side} order qty={qty_str} (attempt {attempt + 1})")
                _ = self.client.place_order(
                    category=self.category,
                    symbol=self.symbol,
                    side=side,
                    orderType="Market",
                    qty=qty_str,
                    reduceOnly=reduce_only,
                )
                return True
            except Exception as e:
                logger.error(f"Error placing {side} order on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    self._sleep(self.retry_delay)
        return False

    def has_open_position(self):
        """Sync local state (size, side, avgPrice, TP/SL) with exchange."""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.get_positions(category=self.category, symbol=self.symbol)
                if not (isinstance(resp, dict) and "result" in resp):
                    logger.error(f"Invalid positions response: {resp}")
                    continue
                lst = resp["result"].get("list") or []
                found = None
                for pos in lst:
                    size = float(pos.get("size") or 0.0)
                    if size > 0:
                        found = pos
                        break
                if not found:
                    self._reset_position_state()
                    return False

                side = (found.get("side") or "").lower()
                avg = float(found.get("avgPrice") or 0.0)
                size = float(found.get("size") or 0.0)
                tp_price = float(found.get("takeProfit") or 0.0) if found.get("takeProfit") else None
                sl_price = float(found.get("stopLoss") or 0.0) if found.get("stopLoss") else None

                if self.position_side != side:
                    self.highest_price = None
                    self.lowest_price = None
                    self.trailing_activated = False

                self.position_side = side
                self.entry_price = avg
                self.position_qty = size
                self.exchange_tp_price = tp_price
                self.exchange_sl_price = sl_price
                return True
            except Exception as e:
                logger.error(f"Position check error on attempt {attempt + 1}: {e}")
            if attempt < self.max_retries - 1:
                self._sleep(self.retry_delay)
        return False

    def _reset_position_state(self):
        self.position_side = None
        self.entry_price = None
        self.position_qty = 0.0
        self.highest_price = None
        self.lowest_price = None
        self.trailing_activated = False
        self.breakeven_active = False
        self.current_stop_loss_pct = self.base_stop_loss_pct
        self.exchange_tp_price = None
        self.exchange_sl_price = None
        self.achieved_targets = []
        self.dca_count = 0

    # ---------- TP/SL on Exchange ----------
    def _compute_tpsl_prices(self):
        if not self.entry_price or not self.position_side:
            return None, None
        entry = self.entry_price
        tp_pct = float(self.take_profit_pct) / 100.0
        sl_pct = float(self.current_stop_loss_pct) / 100.0
        if self.position_side == "buy":
            tp_price = entry * (1 + tp_pct)
            sl_price = entry * (1 - sl_pct)
        else:
            tp_price = entry * (1 - tp_pct)
            sl_price = entry * (1 + sl_pct)
        return round(tp_price, 6), round(sl_price, 6)

    def _apply_tpsl_exchange(self, tp_price=None, sl_price=None):
        if not self.position_side:
            return False
        params = {
            "category": self.category,
            "symbol": self.symbol,
            "tpTriggerBy": "LastPrice",
            "slTriggerBy": "LastPrice",
        }
        if tp_price is not None:
            params["takeProfit"] = str(tp_price)
        if sl_price is not None:
            params["stopLoss"] = str(sl_price)
        try:
            resp = self.client.set_trading_stop(**params)
            ok = isinstance(resp, dict) and resp.get("retCode") == 0
            if ok:
                if tp_price is not None:
                    self.exchange_tp_price = tp_price
                if sl_price is not None:
                    self.exchange_sl_price = sl_price
                tp_str = f"{self.exchange_tp_price:.6f}" if self.exchange_tp_price is not None else "-"
                sl_str = f"{self.exchange_sl_price:.6f}" if self.exchange_sl_price is not None else "-"
                msg = f"Exchange TP/SL updated: TP={tp_str}, SL={sl_str}"
                logger.info(msg)
                self.send_telegram_notification(msg, markdown=False)
            else:
                logger.error(f"set_trading_stop failed: {resp}")
            return ok
        except Exception as e:
            logger.error(f"set_trading_stop error: {e}")
            return False

    def _ensure_tpsl_after_sync(self):
        if not self.position_side or not self.entry_price:
            return
        tp, sl = self._compute_tpsl_prices()
        need_tp = (self.exchange_tp_price is None or self.exchange_tp_price <= 0)
        need_sl = (self.exchange_sl_price is None or self.exchange_sl_price <= 0)
        if need_tp or need_sl:
            self._apply_tpsl_exchange(tp_price=tp if need_tp else None,
                                      sl_price=sl if need_sl else None)

    def _update_trailing_stop_exchange(self, current_price: float):
        if not (self.position_side and self.entry_price):
            return
        act_threshold = 0.01 * self.entry_price  # activate trailing after 1% favorable move
        sl_pct = self.trailing_stop_pct / 100.0

        if self.position_side == "buy":
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
            if current_price > self.entry_price + act_threshold:
                self.trailing_activated = True
            if self.trailing_activated:
                target_sl = round(self.highest_price * (1 - sl_pct), 6)
                if not self.exchange_sl_price or target_sl > float(self.exchange_sl_price):
                    self._apply_tpsl_exchange(sl_price=target_sl)
        else:
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
            if current_price < self.entry_price - act_threshold:
                self.trailing_activated = True
            if self.trailing_activated:
                target_sl = round(self.lowest_price * (1 + sl_pct), 6)
                if not self.exchange_sl_price or target_sl < float(self.exchange_sl_price):
                    self._apply_tpsl_exchange(sl_price=target_sl)

    # ---------- Main loop ----------
    def _trading_loop(self):
        logger.info("Starting GPT trading loop (24/7, exchange TP/SL active)")
        self.send_telegram_notification(
            "🤖 *Trading GPT Started (24/7)*\n\n"
            f"*Symbol:* {self.symbol}\n"
            f"*Strategy:* RSI ({self.rsi_buy_threshold}/{self.rsi_sell_threshold})\n"
            f"*Quantity:* {self.trade_qty}\n"
            f"*SL/TP:* On-Exchange via set_trading_stop\n"
            f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            _ = self.client.get_tickers(category=self.category, symbol=self.symbol)
        except Exception as e:
            logger.error(f"API connection FAILED: {e}")
            self.send_telegram_notification(f"❌ *API Connection Failed*\n\nError: {str(e)}")
            return

        while not self.stop_event.is_set():
            self.sleep_to_next_candle()
            if self.stop_event.is_set():
                break
            try:
                current_price = self.get_current_price()
                if self.stop_event.is_set():
                    break
                if not current_price:
                    logger.error("Failed to get current price after retries")
                    continue

                had_position = self.position_side is not None and self.position_qty > 0
                if self.has_open_position():
                    self._ensure_tpsl_after_sync()
                else:
                    if had_position:
                        self._update_counters_on_close("SYNC_CLOSE", current_price)
                    # no position -> consider entries below
                klines_15 = self.get_klines_data(interval="15")
                if self.stop_event.is_set():
                    break
                if not klines_15:
                    logger.error("Failed to get M15 klines data")
                    continue
                close_15 = np.array([float(k[4]) for k in klines_15], dtype=float)
                rsi_15_arr = self.calculate_rsi(close_15)
                if np.isnan(rsi_15_arr[-1]):
                    valid = rsi_15_arr[~np.isnan(rsi_15_arr)]
                    if valid.size == 0:
                        logger.warning("RSI M15 not available yet; skipping this candle")
                        continue
                    rsi_15 = valid[-1]
                else:
                    rsi_15 = rsi_15_arr[-1]
                logger.info(f"M15: last RSI: {rsi_15:.2f}, price range: {np.min(close_15):.4f}-{np.max(close_15):.4f}")

                # If have position: update trailing SL on exchange
                if self.position_side:
                    self._update_trailing_stop_exchange(current_price)

                # Entries (24/7), respecting pause & cooldown
                if not self.position_side:
                    if not self.pause_event.is_set() and not self.in_cooldown():
                        if rsi_15 <= self.rsi_buy_threshold:
                            if self.place_trade_order("Buy", qty=self.trade_qty, reduce_only=False):
                                if self.has_open_position():
                                    tp, sl = self._compute_tpsl_prices()
                                    self._apply_tpsl_exchange(tp_price=tp, sl_price=sl)
                                    self.highest_price = current_price
                                    self.send_telegram_notification(
                                        "🚀 *NEW POSITION ENTRY*\n\n"
                                        f"*Action:* BUY\n*Qty:* {self.trade_qty}\n*Entry:* {self.entry_price:.6f}\n"
                                        f"*TP/SL:* {tp:.6f} / {sl:.6f}\n*RSI(M15):* {rsi_15:.2f}"
                                    )
                        elif rsi_15 >= self.rsi_sell_threshold:
                            if self.place_trade_order("Sell", qty=self.trade_qty, reduce_only=False):
                                if self.has_open_position():
                                    tp, sl = self._compute_tpsl_prices()
                                    self._apply_tpsl_exchange(tp_price=tp, sl_price=sl)
                                    self.lowest_price = current_price
                                    self.send_telegram_notification(
                                        "🚀 *NEW POSITION ENTRY*\n\n"
                                        f"*Action:* SELL\n*Qty:* {self.trade_qty}\n*Entry:* {self.entry_price:.6f}\n"
                                        f"*TP/SL:* {tp:.6f} / {sl:.6f}\n*RSI(M15):* {rsi_15:.2f}"
                                    )
                else:
                    # DCA (respect pause/cooldown)
                    if (not self.pause_event.is_set() and not self.in_cooldown()
                        and self.dca_count < self.max_dca_steps):
                        can_avg = (
                            (self.position_side == "buy" and rsi_15 <= self.rsi_buy_threshold) or
                            (self.position_side == "sell" and rsi_15 >= self.rsi_sell_threshold)
                        )
                        if can_avg:
                            planned_new_qty = self.position_qty + float(self.trade_qty)
                            if self.max_position_qty is not None and planned_new_qty > self.max_position_qty:
                                logger.info("Skip averaging: would exceed max_position_qty")
                            else:
                                if self.place_trade_order(self.position_side.capitalize(), qty=self.trade_qty, reduce_only=False):
                                    if self.has_open_position():
                                        tp, sl = self._compute_tpsl_prices()
                                        self._apply_tpsl_exchange(tp_price=tp, sl_price=sl)
                                        self.dca_count += 1
                                        self.send_telegram_notification(
                                            "📊 *POSITION AVERAGED*\n\n"
                                            f"*Side:* {self.position_side.upper()}\n*Added:* {self.trade_qty}\n"
                                            f"*New Avg:* {self.entry_price:.6f}\n*New Qty:* {self.position_qty:.6f}\n"
                                            f"*TP/SL:* {tp:.6f} / {sl:.6f}\n*RSI(M15):* {rsi_15:.2f}"
                                        )
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                self.send_telegram_notification(
                    "⚠️ *Trading Bot GPT Error*\n\n"
                    f"*Error:* {str(e)}\n"
                    f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self._sleep(5)

        logger.info("Trading GPT loop stopped")

    def sleep_to_next_candle(self):
        now = datetime.now()
        minutes_past = now.minute % 15
        minutes_to_wait = 15 - minutes_past if minutes_past > 0 else 0
        seconds_to_wait = (minutes_to_wait * 60) - now.second
        if seconds_to_wait <= 0:
            seconds_to_wait = 900
        logger.info(f"Sleeping for {seconds_to_wait} seconds until next candle")
        end = time.monotonic() + seconds_to_wait
        while not self.stop_event.is_set():
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            self._sleep(min(1.0, max(0.05, remaining)))

    # ---------- Start/Stop ----------
    def start_trading(self):
        with self._state_lock:
            if self.trading_thread and self.trading_thread.is_alive():
                logger.info("Trading already running")
                return False
            self.stop_event.clear()
            self.pause_event.clear()
            self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.trading_thread.start()
            return True

    def _wait_trading_stop(self):
        t = self.trading_thread
        if t:
            t.join(timeout=30)
            if t.is_alive():
                logger.warning("Trading thread did not stop within timeout")
        with self._state_lock:
            self.trading_thread = None

    def stop_trading(self):
        with self._state_lock:
            if not (self.trading_thread and self.trading_thread.is_alive()):
                logger.info("Trading not running")
                return False
            self.stop_event.set()
            threading.Thread(target=self._wait_trading_stop, daemon=True).start()
            return True

    def is_trading(self):
        t = self.trading_thread
        return bool(t and t.is_alive())


# ---------- Entrypoint ----------
def _handle_sig(bot: TradingBot, *_):
    try:
        bot.stop_trading()
    finally:
        logger.info("Process GPT stopping by signal")
        bot.send_telegram_notification("⏹ Bot process stopped by signal")


if __name__ == "__main__":
    bot = TradingBot()
    bot.start_telegram_listener()
    bot.start_trading()

    signal.signal(signal.SIGINT, lambda *a: _handle_sig(bot, *a))
    try:
        signal.signal(signal.SIGTERM, lambda *a: _handle_sig(bot, *a))
    except Exception:
        pass

    try:
        bot.stop_event.wait()
    except KeyboardInterrupt:
        _handle_sig(bot)