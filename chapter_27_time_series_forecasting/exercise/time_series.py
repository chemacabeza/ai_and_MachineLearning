import numpy as np

def generate_time_series(n=300, seed=42):
    """Generate a synthetic time series with trend, seasonality, and noise."""
    np.random.seed(seed)
    t = np.arange(n)
    trend = 0.02 * t
    seasonality = 2 * np.sin(2 * np.pi * t / 50)
    noise = np.random.randn(n) * 0.5
    return trend + seasonality + noise

def difference(series, order=1):
    """Apply differencing to remove trend."""
    diff = series.copy()
    for _ in range(order):
        diff = np.diff(diff)
    return diff

def create_sliding_windows(series, window_size):
    """Convert time series to supervised learning format."""
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def fit_ar(X, y):
    """Fit autoregressive model via least squares."""
    X_bias = np.column_stack([np.ones(len(X)), X])
    w = np.linalg.lstsq(X_bias, y, rcond=None)[0]
    return w

def predict_ar(X, w):
    """Predict using fitted AR model."""
    X_bias = np.column_stack([np.ones(len(X)), X])
    return X_bias @ w

def forecast_multi_step(series, w, window_size, steps):
    """Generate multi-step forecasts by feeding predictions back."""
    history = list(series[-window_size:])
    forecasts = []
    for _ in range(steps):
        x = np.array(history[-window_size:]).reshape(1, -1)
        x_bias = np.column_stack([np.ones(1), x])
        pred = (x_bias @ w)[0]
        forecasts.append(pred)
        history.append(pred)
    return np.array(forecasts)

def main():
    series = generate_time_series(n=300)

    print("=" * 60)
    print("TIME SERIES FORECASTING FROM SCRATCH")
    print("=" * 60)
    print(f"Series length: {len(series)}")
    print(f"Mean: {series.mean():.2f}, Std: {series.std():.2f}")
    print("-" * 60)

    # Part 1: Effect of differencing
    print("\n--- Part 1: Stationarity via Differencing ---")
    for d in [0, 1]:
        s = difference(series, order=d) if d > 0 else series
        print(f"  d={d}: Mean={s.mean():7.3f}, Std={s.std():5.3f}, "
              f"Range=[{s.min():.2f}, {s.max():.2f}]")

    # Part 2: Sliding window size comparison
    print("\n--- Part 2: Window Size vs Forecast Accuracy ---")
    print(f"  {'Window':>6} {'Train MSE':>10} {'Test MSE':>10}  Visual")
    print("  " + "-" * 45)

    diff_series = difference(series, order=1)
    train_size = int(len(diff_series) * 0.8)

    for window in [3, 5, 10, 20, 50]:
        X, y = create_sliding_windows(diff_series, window)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        if len(X_train) < window + 1:
            continue

        w = fit_ar(X_train, y_train)
        train_pred = predict_ar(X_train, w)
        test_pred = predict_ar(X_test, w)

        train_mse = np.mean((y_train - train_pred)**2)
        test_mse = np.mean((y_test - test_pred)**2)

        bar = "█" * max(1, int((1 - min(test_mse, 1)) * 30))
        print(f"  {window:6d} {train_mse:10.4f} {test_mse:10.4f}  {bar}")

    # Part 3: Multi-step forecasting
    print("\n--- Part 3: Multi-Step Forecasting (window=10) ---")
    window = 10
    X, y = create_sliding_windows(diff_series, window)
    w = fit_ar(X, y)

    forecasts = forecast_multi_step(diff_series, w, window, steps=20)
    last_value = series[-1]
    reconstructed = [last_value]
    for f in forecasts:
        reconstructed.append(reconstructed[-1] + f)
    reconstructed = np.array(reconstructed[1:])

    print(f"  Last observed value: {series[-1]:.3f}")
    print(f"  Forecasted values:")
    for i in range(0, 20, 4):
        print(f"    t+{i+1:2d}: {reconstructed[i]:7.3f}")

    # Part 4: Autocorrelation analysis
    print("\n--- Part 4: Autocorrelation Analysis ---")
    centered = diff_series - diff_series.mean()
    var = np.var(centered)
    print(f"  {'Lag':>4} {'ACF':>8}  Visual")
    print("  " + "-" * 35)
    for lag in [1, 2, 5, 10, 25, 50]:
        if lag < len(centered):
            acf = np.mean(centered[lag:] * centered[:-lag]) / var
            bar_len = int(abs(acf) * 30)
            direction = "+" if acf > 0 else "-"
            bar = direction * bar_len
            print(f"  {lag:4d} {acf:8.3f}  {bar}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("1. Differencing removes trends → stationarity.")
    print("2. Window size controls how much 'memory' the model uses.")
    print("3. Multi-step forecasts accumulate error over time.")
    print("4. Autocorrelation reveals the series' temporal structure.")
    print("=" * 60)

if __name__ == "__main__":
    main()
