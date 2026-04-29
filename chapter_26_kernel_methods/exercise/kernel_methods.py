import numpy as np

def rbf_kernel(X1, X2, gamma):
    """Compute RBF kernel matrix between X1 and X2."""
    sq_dists = np.sum(X1**2, axis=1, keepdims=True) + \
               np.sum(X2**2, axis=1, keepdims=True).T - \
               2 * X1 @ X2.T
    return np.exp(-gamma * sq_dists)

def linear_kernel(X1, X2):
    """Compute linear kernel matrix."""
    return X1 @ X2.T

def polynomial_kernel(X1, X2, degree=3, c=1):
    """Compute polynomial kernel matrix."""
    return (X1 @ X2.T + c) ** degree

def kernel_ridge_classifier(X_train, y_train, X_test, kernel_fn, lam=0.1):
    """Kernel Ridge Regression for classification."""
    K_train = kernel_fn(X_train, X_train)
    n = K_train.shape[0]
    alpha = np.linalg.solve(K_train + lam * np.eye(n), y_train)
    K_test = kernel_fn(X_test, X_train)
    predictions = K_test @ alpha
    return np.sign(predictions)

def generate_circles(n=200, noise=0.15, seed=42):
    """Generate two concentric circles (non-linearly separable)."""
    np.random.seed(seed)
    n_per_class = n // 2

    # Inner circle
    theta1 = np.random.uniform(0, 2 * np.pi, n_per_class)
    r1 = 0.5 + np.random.randn(n_per_class) * noise
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])

    # Outer circle
    theta2 = np.random.uniform(0, 2 * np.pi, n_per_class)
    r2 = 1.5 + np.random.randn(n_per_class) * noise
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])

    X = np.vstack([X1, X2])
    y = np.array([1] * n_per_class + [-1] * n_per_class, dtype=float)
    return X, y

def main():
    X, y = generate_circles(n=200, seed=42)
    split = 150
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print("=" * 60)
    print("KERNEL METHODS FROM SCRATCH")
    print("=" * 60)
    print(f"Dataset: Two concentric circles (non-linearly separable)")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print("-" * 60)

    # Linear kernel (expected to fail)
    pred = kernel_ridge_classifier(X_train, y_train, X_test,
                                   lambda X1, X2: linear_kernel(X1, X2))
    acc = np.mean(pred == y_test)
    bar = "█" * int(acc * 40)
    print(f"\nLinear Kernel:      {acc:.2%}  {bar}")

    # Polynomial kernels
    for d in [2, 3, 5]:
        pred = kernel_ridge_classifier(X_train, y_train, X_test,
                                       lambda X1, X2, d=d: polynomial_kernel(X1, X2, degree=d))
        acc = np.mean(pred == y_test)
        bar = "█" * int(acc * 40)
        print(f"Polynomial (d={d}):   {acc:.2%}  {bar}")

    # RBF kernel with varying gamma
    print(f"\nRBF Kernel (varying γ):")
    print(f"  {'γ':>8} {'Accuracy':>10}  {'Visual':>30}")
    print("  " + "-" * 50)
    for gamma in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        pred = kernel_ridge_classifier(X_train, y_train, X_test,
                                       lambda X1, X2, g=gamma: rbf_kernel(X1, X2, g))
        acc = np.mean(pred == y_test)
        bar = "█" * int(acc * 40)
        note = ""
        if gamma < 0.05:
            note = " (underfit)"
        elif gamma > 20:
            note = " (overfit)"
        elif acc > 0.9:
            note = " ← sweet spot"
        print(f"  {gamma:8.2f} {acc:10.2%}  {bar}{note}")

    # Kernel matrix visualization
    print(f"\nKernel Matrix Properties (RBF, γ=1.0):")
    K = rbf_kernel(X_train[:10], X_train[:10], gamma=1.0)
    print(f"  Shape: {K.shape}")
    print(f"  Diagonal (self-similarity): {K[0,0]:.4f} (always 1.0 for RBF)")
    print(f"  Min off-diagonal: {K[np.triu_indices(10, k=1)].min():.4f}")
    print(f"  Max off-diagonal: {K[np.triu_indices(10, k=1)].max():.4f}")
    print(f"  Symmetric: {np.allclose(K, K.T)}")

    eigenvalues = np.linalg.eigvalsh(K)
    print(f"  Positive semi-definite: {np.all(eigenvalues >= -1e-10)}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("1. Linear kernels fail on non-linear data (concentric circles).")
    print("2. RBF kernel achieves high accuracy via implicit mapping.")
    print("3. γ controls complexity: too small = underfit, too large = overfit.")
    print("4. The kernel trick avoids explicit high-dimensional computation.")
    print("=" * 60)

if __name__ == "__main__":
    main()
