import numpy as np

def true_function(x):
    """The true underlying function (unknown to the model)."""
    return np.sin(2 * x) + 0.5 * x

def generate_data(n=30, noise=0.5, seed=None):
    """Generate noisy observations of the true function."""
    if seed is not None:
        np.random.seed(seed)
    X = np.sort(np.random.uniform(-3, 3, n))
    y = true_function(X) + np.random.randn(n) * noise
    return X, y

def polynomial_features(X, degree):
    """Create polynomial feature matrix."""
    return np.column_stack([X**d for d in range(degree + 1)])

def ridge_regression(X, y, lam=0.0):
    """Closed-form Ridge Regression: w = (X'X + λI)^{-1} X'y"""
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0  # Don't regularize the bias term
    w = np.linalg.solve(X.T @ X + lam * I, X.T @ y)
    return w

def main():
    np.random.seed(42)

    print("=" * 60)
    print("BIAS-VARIANCE TRADEOFF & REGULARIZATION")
    print("=" * 60)

    # Part 1: Bias-Variance with polynomial degree
    print("\n--- Part 1: Polynomial Complexity vs Error ---")
    print(f"{'Degree':>6} {'Train MSE':>10} {'Test MSE':>10} {'Gap':>8}  Visual")
    print("-" * 55)

    X_train, y_train = generate_data(n=30, seed=42)
    X_test, y_test = generate_data(n=100, seed=99)

    for degree in [1, 2, 3, 5, 10, 15]:
        Phi_train = polynomial_features(X_train, degree)
        Phi_test = polynomial_features(X_test, degree)

        w = ridge_regression(Phi_train, y_train, lam=0.0)

        train_pred = Phi_train @ w
        test_pred = Phi_test @ w

        train_mse = np.mean((y_train - train_pred)**2)
        test_mse = np.mean((y_test - test_pred)**2)
        gap = test_mse - train_mse

        bar = "█" * min(int(test_mse * 5), 30)
        status = "← sweet spot" if degree == 3 else ("OVERFIT" if gap > 2 else "")
        print(f"{degree:6d} {train_mse:10.4f} {test_mse:10.4f} {gap:8.4f}  {bar} {status}")

    # Part 2: Regularization effect
    print("\n--- Part 2: Ridge Regularization (degree=10) ---")
    print(f"{'Lambda':>10} {'Train MSE':>10} {'Test MSE':>10} {'Nonzero':>8}")
    print("-" * 45)

    Phi_train = polynomial_features(X_train, 10)
    Phi_test = polynomial_features(X_test, 10)

    for lam in [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        w = ridge_regression(Phi_train, y_train, lam=lam)
        train_mse = np.mean((y_train - Phi_train @ w)**2)
        test_mse = np.mean((y_test - Phi_test @ w)**2)
        nonzero = np.sum(np.abs(w) > 0.01)

        marker = " ← best" if 0.05 < lam < 0.5 else ""
        print(f"{lam:10.3f} {train_mse:10.4f} {test_mse:10.4f} {nonzero:8d}{marker}")

    # Part 3: L1 vs L2 simulation
    print("\n--- Part 3: L1 (Lasso) vs L2 (Ridge) Effect ---")
    print("Simulating weight shrinkage on a 20-feature problem:")

    np.random.seed(42)
    n, p = 50, 20
    X = np.random.randn(n, p)
    true_w = np.zeros(p)
    true_w[:5] = [3, -2, 1.5, -1, 0.5]  # Only 5 features matter
    y = X @ true_w + np.random.randn(n) * 0.5

    # Ridge
    w_ridge = ridge_regression(X, y, lam=1.0)
    # Approximate L1 via iteratively reweighted L2
    w_lasso = w_ridge.copy()
    for _ in range(50):
        D = np.diag(1.0 / (np.abs(w_lasso) + 1e-6))
        w_lasso = np.linalg.solve(X.T @ X + 1.0 * D, X.T @ y)

    print(f"\n  {'Feature':>8} {'True':>6} {'Ridge':>8} {'Lasso':>8}")
    print("  " + "-" * 35)
    for i in range(p):
        marker = " *" if abs(true_w[i]) > 0 else ""
        print(f"  w[{i:2d}]   {true_w[i]:6.2f} {w_ridge[i]:8.4f} {w_lasso[i]:8.4f}{marker}")

    ridge_zeros = np.sum(np.abs(w_ridge) < 0.01)
    lasso_zeros = np.sum(np.abs(w_lasso) < 0.01)
    print(f"\n  Near-zero weights: Ridge={ridge_zeros}, Lasso={lasso_zeros}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("1. Too simple = high bias. Too complex = high variance.")
    print("2. Ridge shrinks weights; Lasso eliminates them (sparsity).")
    print("3. Cross-validation finds the optimal complexity level.")
    print("=" * 60)

if __name__ == "__main__":
    main()
