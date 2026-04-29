import numpy as np

def generate_data(n=200, seed=42):
    """Generate a non-linearly separable 2D dataset."""
    np.random.seed(seed)
    X = np.random.randn(n, 2)
    # Circle boundary: points inside radius 1 are class +1
    y = np.where(X[:, 0]**2 + X[:, 1]**2 < 1.2, 1, -1)
    # Add noise by flipping 5% of labels
    flip = np.random.choice(n, size=n // 20, replace=False)
    y[flip] *= -1
    return X, y

class DecisionStump:
    """A weak classifier: single threshold on one feature."""
    def __init__(self):
        self.feature = 0
        self.threshold = 0
        self.polarity = 1

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    predictions[polarity * X[:, feature] < polarity * threshold] = -1
                    error = np.sum(weights[predictions != y])
                    if error < best_error:
                        best_error = error
                        self.feature = feature
                        self.threshold = threshold
                        self.polarity = polarity
        return best_error

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        predictions[self.polarity * X[:, self.feature] < self.polarity * self.threshold] = -1
        return predictions

def adaboost(X, y, T=10):
    """AdaBoost with decision stumps."""
    n_samples = X.shape[0]
    weights = np.ones(n_samples) / n_samples
    stumps = []
    alphas = []

    for t in range(T):
        stump = DecisionStump()
        error = stump.fit(X, y, weights)
        error = max(error, 1e-10)  # prevent division by zero

        # Classifier weight: better stumps get higher alpha
        alpha = 0.5 * np.log((1 - error) / error)

        predictions = stump.predict(X)

        # Update weights: increase weight of misclassified samples
        weights *= np.exp(-alpha * y * predictions)
        weights /= weights.sum()

        stumps.append(stump)
        alphas.append(alpha)

    return stumps, alphas

def predict_adaboost(X, stumps, alphas):
    """Combined prediction from all stumps."""
    n_samples = X.shape[0]
    final_pred = np.zeros(n_samples)
    for stump, alpha in zip(stumps, alphas):
        final_pred += alpha * stump.predict(X)
    return np.sign(final_pred)

def main():
    X, y = generate_data()
    split = 150
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print("=" * 60)
    print("ADABOOST FROM SCRATCH")
    print("=" * 60)

    # Single stump baseline
    stump = DecisionStump()
    weights = np.ones(len(y_train)) / len(y_train)
    stump.fit(X_train, y_train, weights)
    single_acc = np.mean(stump.predict(X_test) == y_test)
    print(f"\nSingle Decision Stump Accuracy: {single_acc:.2%}")

    # AdaBoost with increasing rounds
    for T in [1, 5, 10, 25, 50]:
        stumps, alphas = adaboost(X_train, y_train, T=T)
        train_pred = predict_adaboost(X_train, stumps, alphas)
        test_pred = predict_adaboost(X_test, stumps, alphas)

        train_acc = np.mean(train_pred == y_train)
        test_acc = np.mean(test_pred == y_test)

        bar = "█" * int(test_acc * 40)
        print(f"  T={T:3d} stumps → Train: {train_acc:.2%}  Test: {test_acc:.2%}  {bar}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("A single stump barely beats random guessing.")
    print("AdaBoost combines many stumps into a strong classifier.")
    print("Each new stump focuses on what previous ones got wrong.")
    print("=" * 60)

if __name__ == "__main__":
    main()
