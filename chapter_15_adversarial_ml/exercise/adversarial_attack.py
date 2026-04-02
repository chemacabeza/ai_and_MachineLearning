import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

print("=== Adversarial ML: FGSM-Style Attack Simulation ===\n")
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
model = LogisticRegression(max_iter=200).fit(X, y)
original_acc = model.score(X, y)
print(f"Original model accuracy: {original_acc*100:.1f}%\n")

epsilons = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
print(f"{'Epsilon':<10} {'Accuracy':>10} {'Flipped':>10}")
print("-" * 32)
for eps in epsilons:
    np.random.seed(42)
    coefficients = model.coef_[0]
    perturbation = eps * np.sign(coefficients)
    X_adv = X.copy()
    for i in range(len(X_adv)):
        if model.predict([X_adv[i]])[0] == y[i]:
            X_adv[i] += perturbation
    adv_acc = model.score(X_adv, y)
    flipped = int((original_acc - adv_acc) * len(X))
    print(f"{eps:<10.2f} {adv_acc*100:>9.1f}% {flipped:>9d}")

print(f"\n⚠️  With ε=0.5, the model's accuracy drops dramatically!")
print("   The perturbation is small but targeted at the decision boundary.")
print("\n✅ This demonstrates why adversarial robustness matters for AI safety.")
