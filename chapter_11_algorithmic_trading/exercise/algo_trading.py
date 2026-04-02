import numpy as np

print("=== Algorithmic Trading: Moving Average Crossover ===\n")

np.random.seed(42)
days = 252
returns = np.random.normal(loc=0.0005, scale=0.02, size=days)
prices = 100 * np.cumprod(1 + returns)

print(f"Simulated {days} trading days")
print(f"Starting price: ${prices[0]:.2f}")
print(f"Ending price:   ${prices[-1]:.2f}")

short_window = 10
long_window = 50

short_ma = np.convolve(prices, np.ones(short_window)/short_window, mode='valid')
long_ma = np.convolve(prices, np.ones(long_window)/long_window, mode='valid')

min_len = min(len(short_ma), len(long_ma))
short_ma = short_ma[-min_len:]
long_ma = long_ma[-min_len:]
aligned_prices = prices[-min_len:]

signals = np.where(short_ma > long_ma, 1, -1)

daily_returns = np.diff(aligned_prices) / aligned_prices[:-1]
strategy_returns = signals[:-1] * daily_returns

cumulative_market = np.cumprod(1 + daily_returns) - 1
cumulative_strategy = np.cumprod(1 + strategy_returns) - 1

buy_signals = np.sum(np.diff(signals) > 0)
sell_signals = np.sum(np.diff(signals) < 0)

print(f"\n📊 Backtest Results:")
print(f"  Buy signals generated:  {buy_signals}")
print(f"  Sell signals generated: {sell_signals}")
print(f"  Market return (buy & hold): {cumulative_market[-1]*100:+.2f}%")
print(f"  Strategy return:            {cumulative_strategy[-1]*100:+.2f}%")

if cumulative_strategy[-1] > cumulative_market[-1]:
    print("\n✅ Strategy BEAT the market!")
else:
    print("\n📉 Strategy underperformed the market this time.")
    print("   (This is expected — no strategy wins every time!)")
