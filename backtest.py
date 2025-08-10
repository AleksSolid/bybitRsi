import argparse
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    rsi_window: int = 14
    rsi_buy_threshold: float = 35.0
    rsi_sell_threshold: float = 65.0
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 2.0
    trailing_stop_pct: float = 0.5
    trailing_activation_frac_of_entry: float = 0.005
    trade_qty: float = 10.0
    taker_fee_rate: float = 0.0006


@dataclass
class TradeLog:
    timestamp: pd.Timestamp
    action: str
    side: str
    qty: float
    price: float
    rsi: float
    reason: str
    realized_pnl: float


@dataclass
class BacktestResult:
    total_trades: int
    total_entries: int
    total_exits: int
    wins: int
    losses: int
    win_rate: float
    total_realized_pnl: float
    total_fees: float
    net_pnl: float
    max_drawdown: float
    equity_curve: pd.Series
    trades: List[TradeLog] = field(default_factory=list)


def calculate_rsi(prices: np.ndarray, window: int) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if prices.size < window + 1:
        return np.array([])

    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gain[:window])
    avg_loss = np.mean(loss[:window])

    if avg_loss == 0:
        return np.full(len(prices), 100.0)

    rs = avg_gain / avg_loss
    rsi = np.zeros(len(prices))
    rsi[window] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(window + 1, len(prices)):
        delta = deltas[i - 1]
        gain_val = delta if delta > 0 else 0.0
        loss_val = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (window - 1) + gain_val) / window
        avg_loss = (avg_loss * (window - 1) + loss_val) / window
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    ts_cols = [c for c in df.columns if c.lower() in ("timestamp", "time", "date", "datetime")]
    if ts_cols:
        col = ts_cols[0]
        s = df[col]
        if np.issubdtype(s.dtype, np.number):
            df.index = pd.to_datetime(s, unit="ms" if s.max() > 10**12 else "s")
        else:
            df.index = pd.to_datetime(s)
    else:
        df.index = pd.RangeIndex(start=0, stop=len(df))
    return df


def backtest(df: pd.DataFrame, params: StrategyParams) -> BacktestResult:
    df = df.copy()
    df = ensure_datetime_index(df)
    if 'close' not in df.columns:
        raise ValueError("Input data must contain a 'close' column")

    closes = df['close'].astype(float).values
    rsi = calculate_rsi(closes, params.rsi_window)
    if rsi.size == 0:
        raise ValueError("Not enough data to compute RSI")

    df['rsi'] = pd.Series(rsi, index=df.index)

    pos_side: Optional[str] = None  # 'buy' or 'sell'
    entry_price: Optional[float] = None
    position_qty: float = 0.0
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None
    trailing_activated: bool = False
    achieved_targets: List[float] = []

    profit_targets = [
        {'percent': 0.5, 'close_ratio': 0.5, 'move_sl': True},
        {'percent': 0.8, 'close_ratio': 0.3, 'move_sl': False},
        {'percent': 1.0, 'close_ratio': 0.2, 'move_sl': False},
    ]

    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    equity_curve: List[float] = []

    trades: List[TradeLog] = []

    def mark_entry(ts: pd.Timestamp, side: str, price: float, rsi_val: float, reason: str):
        nonlocal pos_side, entry_price, position_qty, highest_price, lowest_price, trailing_activated, achieved_targets, fees_paid
        pos_side = side
        entry_price = price
        position_qty = position_qty + params.trade_qty
        highest_price = price if side == 'buy' else None
        lowest_price = price if side == 'sell' else None
        trailing_activated = False
        achieved_targets = []
        fee = params.taker_fee_rate * price * params.trade_qty
        fees_paid += fee
        trades.append(TradeLog(ts, 'ENTRY', side, params.trade_qty, price, rsi_val, reason, realized_pnl=0.0))

    def mark_exit(ts: pd.Timestamp, action: str, qty: float, price: float, rsi_val: float, reason: str):
        nonlocal pos_side, entry_price, position_qty, highest_price, lowest_price, trailing_activated, achieved_targets, realized_pnl, fees_paid
        if position_qty <= 0 or entry_price is None or pos_side is None:
            return
        qty_to_close = min(qty, position_qty)
        if pos_side == 'buy':
            pnl = (price - entry_price) * qty_to_close
        else:
            pnl = (entry_price - price) * qty_to_close
        fee = params.taker_fee_rate * price * qty_to_close
        realized_pnl += pnl - fee
        fees_paid += fee
        position_qty -= qty_to_close
        trades.append(TradeLog(ts, action, 'Sell' if pos_side == 'buy' else 'Buy', qty_to_close, price, rsi_val, reason, realized_pnl=pnl - fee))
        if position_qty <= 0:
            pos_side = None
            entry_price = None
            highest_price = None
            lowest_price = None
            trailing_activated = False
            achieved_targets = []

    for ts, row in df.iterrows():
        price = float(row['close'])
        rsi_val = float(row['rsi']) if not math.isnan(row['rsi']) else None
        if rsi_val is None:
            equity_curve.append(realized_pnl)
            continue

        if pos_side == 'buy':
            if highest_price is None or price > highest_price:
                highest_price = price
        elif pos_side == 'sell':
            if lowest_price is None or price < lowest_price:
                lowest_price = price

        if pos_side and entry_price is not None and position_qty > 0:
            if not trailing_activated:
                if pos_side == 'buy' and price > entry_price * (1 + params.trailing_activation_frac_of_entry):
                    trailing_activated = True
                if pos_side == 'sell' and price < entry_price * (1 - params.trailing_activation_frac_of_entry):
                    trailing_activated = True

            if trailing_activated:
                if pos_side == 'buy' and highest_price and price <= highest_price * (1 - params.trailing_stop_pct / 100):
                    mark_exit(ts, 'EXIT', position_qty, price, rsi_val, 'TRAILING_STOP')
                elif pos_side == 'sell' and lowest_price and price >= lowest_price * (1 + params.trailing_stop_pct / 100):
                    mark_exit(ts, 'EXIT', position_qty, price, rsi_val, 'TRAILING_STOP')

        if pos_side and entry_price is not None and position_qty > 0:
            profit_pct = (price - entry_price) / entry_price * 100 if pos_side == 'buy' else (entry_price - price) / entry_price * 100
            for target in profit_targets:
                target_percent = target['percent'] * params.take_profit_pct
                if target['percent'] in achieved_targets:
                    continue
                if profit_pct >= target_percent:
                    qty_to_close = position_qty if target['percent'] == 1.0 else position_qty * target['close_ratio']
                    mark_exit(ts, 'PARTIAL', qty_to_close, price, rsi_val, f'TARGET_{int(target["percent"]*100)}')
                    achieved_targets.append(target['percent'])
                    if target['move_sl'] and entry_price is not None:
                        params.stop_loss_pct = 0.0
            if pos_side and entry_price is not None and position_qty > 0:
                price_diff_pct = (price - entry_price) / entry_price * 100
                if pos_side == 'buy':
                    if price_diff_pct <= -params.stop_loss_pct:
                        mark_exit(ts, 'EXIT', position_qty, price, rsi_val, 'STOP_LOSS')
                    elif price_diff_pct >= params.take_profit_pct:
                        mark_exit(ts, 'EXIT', position_qty, price, rsi_val, 'TAKE_PROFIT')
                elif pos_side == 'sell':
                    if price_diff_pct >= params.stop_loss_pct:
                        mark_exit(ts, 'EXIT', position_qty, price, rsi_val, 'STOP_LOSS')
                    elif price_diff_pct <= -params.take_profit_pct:
                        mark_exit(ts, 'EXIT', position_qty, price, rsi_val, 'TAKE_PROFIT')

        if pos_side is None:
            if rsi_val <= params.rsi_buy_threshold:
                mark_entry(ts, 'buy', price, rsi_val, 'RSI_BUY')
            elif rsi_val >= params.rsi_sell_threshold:
                mark_entry(ts, 'sell', price, rsi_val, 'RSI_SELL')
        else:
            if pos_side == 'buy' and rsi_val <= params.rsi_buy_threshold:
                mark_entry(ts, 'buy', price, rsi_val, 'AVERAGING_BUY')
            elif pos_side == 'sell' and rsi_val >= params.rsi_sell_threshold:
                mark_entry(ts, 'sell', price, rsi_val, 'AVERAGING_SELL')

        equity_curve.append(realized_pnl)

    equity = pd.Series(equity_curve, index=df.index)
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    exits = [t for t in trades if t.action in ("EXIT", "PARTIAL")]
    full_exits = [t for t in trades if t.action == "EXIT"]
    entries = [t for t in trades if t.action == "ENTRY"]

    trade_outcomes: List[float] = []
    running_pos = 0.0
    for t in trades:
        if t.action == 'ENTRY':
            running_pos += t.qty
        elif t.action in ('EXIT', 'PARTIAL'):
            trade_outcomes.append(t.realized_pnl)
            running_pos -= t.qty

    wins = sum(1 for x in trade_outcomes if x > 0)
    losses = sum(1 for x in trade_outcomes if x <= 0)
    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0.0

    total_realized = realized_pnl
    result = BacktestResult(
        total_trades=len(trades),
        total_entries=len(entries),
        total_exits=len(full_exits),
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_realized_pnl=total_realized,
        total_fees=fees_paid,
        net_pnl=total_realized,
        max_drawdown=float(max_dd),
        equity_curve=equity,
        trades=trades,
    )
    return result


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    if 'close' not in cols_lower:
        raise ValueError("CSV must include a 'close' column")
    if 'close' != cols_lower['close']:
        df.rename(columns={cols_lower['close']: 'close'}, inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Backtest RSI + MA bot rules on CSV klines")
    parser.add_argument('--csv', required=True, help='Path to CSV with candles (must include close column)')
    parser.add_argument('--rsi-window', type=int, default=14)
    parser.add_argument('--rsi-buy', type=float, default=35.0)
    parser.add_argument('--rsi-sell', type=float, default=65.0)
    parser.add_argument('--tp', type=float, default=2.0, help='Take profit percent')
    parser.add_argument('--sl', type=float, default=2.0, help='Stop loss percent')
    parser.add_argument('--trail', type=float, default=0.5, help='Trailing stop percent')
    parser.add_argument('--trail-activate', type=float, default=0.5, help='Activation threshold percent of entry (e.g., 0.5)')
    parser.add_argument('--qty', type=float, default=10.0, help='Trade quantity in base units')
    parser.add_argument('--fee', type=float, default=0.0006, help='Taker fee rate per trade')
    parser.add_argument('--trades-out', default='', help='Optional path to write trades CSV')

    args = parser.parse_args()
    df = load_csv(args.csv)

    params = StrategyParams(
        rsi_window=args.rsi_window,
        rsi_buy_threshold=args.rsi_buy,
        rsi_sell_threshold=args.rsi_sell,
        take_profit_pct=args.tp,
        stop_loss_pct=args.sl,
        trailing_stop_pct=args.trail,
        trailing_activation_frac_of_entry=args.trail_activate / 100.0,
        trade_qty=args.qty,
        taker_fee_rate=args.fee,
    )

    result = backtest(df, params)

    print("=== Backtest Summary ===")
    print(f"Total trades: {result.total_trades}")
    print(f"Entries: {result.total_entries} | Full exits: {result.total_exits}")
    print(f"Wins: {result.wins} | Losses: {result.losses} | Win rate: {result.win_rate:.2f}%")
    print(f"Fees paid: {result.total_fees:.2f} USDT")
    print(f"Total realized PnL: {result.total_realized_pnl:.2f} USDT")
    print(f"Net PnL: {result.net_pnl:.2f} USDT")
    print(f"Max drawdown: {result.max_drawdown:.2f} USDT")

    if args.trades_out:
        trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
        trades_df.to_csv(args.trades_out, index=False)
        print(f"Trades written to {args.trades_out}")


if __name__ == '__main__':
    main()