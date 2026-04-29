import numpy as np

def pca_from_scratch(X, n_components):
    """Perform PCA using eigendecomposition of the covariance matrix."""
    # Step 1: Center the data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort by eigenvalue (descending)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Step 5: Select top-k components
    components = eigenvectors[:, :n_components]

    # Step 6: Project data onto new axes
    X_projected = X_centered @ components

    return X_projected, eigenvalues, components

def main():
    np.random.seed(42)

    # Generate synthetic 10D data with only 3 real dimensions of variation
    n_samples = 200
    n_features = 10
    n_informative = 3

    # Create 3 informative features
    Z = np.random.randn(n_samples, n_informative) * [5, 3, 1]

    # Embed into 10D via a random linear transformation
    W = np.random.randn(n_informative, n_features)
    X = Z @ W + np.random.randn(n_samples, n_features) * 0.5  # small noise

    print("=" * 60)
    print("PCA FROM SCRATCH")
    print("=" * 60)
    print(f"Original data: {n_samples} samples × {n_features} features")
    print(f"True informative dimensions: {n_informative}")
    print("-" * 60)

    # Run PCA
    X_reduced, eigenvalues, components = pca_from_scratch(X, n_components=n_features)

    # Explained variance ratios
    total_var = eigenvalues.sum()
    explained_ratios = eigenvalues / total_var
    cumulative = np.cumsum(explained_ratios)

    print("\nEigenvalue Spectrum (Scree Plot):")
    print("-" * 45)
    for i, (ev, ratio, cum) in enumerate(zip(eigenvalues, explained_ratios, cumulative)):
        bar = "█" * int(ratio * 80)
        marker = " ← 95% threshold" if i > 0 and cumulative[i-1] < 0.95 <= cum else ""
        print(f"  PC{i+1:2d}: {ratio:6.1%} (cumulative: {cum:6.1%}) {bar}{marker}")

    # Determine optimal components for 95% variance
    n_optimal = np.argmax(cumulative >= 0.95) + 1
    print(f"\nOptimal components for 95% variance: {n_optimal}")
    print(f"Compression ratio: {n_features}D → {n_optimal}D ({(1 - n_optimal/n_features):.0%} reduction)")

    # Show reconstruction error
    for k in [1, 2, 3, 5, n_features]:
        X_proj, _, comps = pca_from_scratch(X, n_components=k)
        X_reconstructed = X_proj @ comps.T + X.mean(axis=0)
        error = np.mean((X - X_reconstructed) ** 2)
        print(f"  k={k:2d} components → MSE: {error:.4f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print(f"10 features, but {n_optimal} principal components capture 95%+ variance.")
    print("PCA reveals the true dimensionality hidden in the data.")
    print("=" * 60)

if __name__ == "__main__":
    main()
