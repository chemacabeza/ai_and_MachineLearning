import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=== Active Learning: Uncertainty Sampling vs Random ===\n")
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)
X_test, y_test = X[800:], y[800:]
X_pool, y_pool = X[:800], y[:800]

def run_experiment(strategy, seed=42):
    rng = np.random.RandomState(seed)
    labeled_idx = list(rng.choice(len(X_pool), size=10, replace=False))
    unlabeled_idx = [i for i in range(len(X_pool)) if i not in labeled_idx]
    accuracies = []
    for round_num in range(20):
        model = LogisticRegression(max_iter=200)
        model.fit(X_pool[labeled_idx], y_pool[labeled_idx])
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies.append(acc)
        if not unlabeled_idx:
            break
        if strategy == "uncertainty":
            probs = model.predict_proba(X_pool[unlabeled_idx])
            uncertainty = 1 - np.max(probs, axis=1)
            top_5 = np.argsort(uncertainty)[-5:]
            query_idx = [unlabeled_idx[i] for i in top_5]
        else:
            query_idx = [unlabeled_idx[i] for i in rng.choice(len(unlabeled_idx), 5, replace=False)]
        labeled_idx.extend(query_idx)
        for idx in query_idx:
            unlabeled_idx.remove(idx)
    return accuracies

active_acc = run_experiment("uncertainty")
random_acc = run_experiment("random")
print(f"{'Round':<8} {'Active':>10} {'Random':>10}")
print("-" * 30)
for i in range(len(active_acc)):
    print(f"{i+1:<8} {active_acc[i]*100:>9.1f}% {random_acc[i]*100:>9.1f}%")
print(f"\n✅ Active Learning reaches {active_acc[-1]*100:.1f}% accuracy")
print(f"   Random sampling reaches {random_acc[-1]*100:.1f}% accuracy")
