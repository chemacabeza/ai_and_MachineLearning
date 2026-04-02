import numpy as np
from sklearn.datasets import make_moons
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score

print("=== Semi-Supervised Learning: Label Propagation ===\n")

X, y_true = make_moons(n_samples=500, noise=0.1, random_state=42)
print(f"Total data points: {len(X)}")

rng = np.random.RandomState(42)
labeled_mask = rng.rand(len(X)) < 0.10

y_partial = y_true.copy()
y_partial[~labeled_mask] = -1

labeled_count = (y_partial != -1).sum()
unlabeled_count = (y_partial == -1).sum()
print(f"Labeled samples: {labeled_count} ({100*labeled_count/len(X):.0f}%)")
print(f"Unlabeled samples: {unlabeled_count} ({100*unlabeled_count/len(X):.0f}%)")

print("\nTraining Label Propagation model...")
model = LabelPropagation(kernel='rbf', gamma=20)
model.fit(X, y_partial)

y_predicted = model.predict(X)
accuracy_all = accuracy_score(y_true, y_predicted)
accuracy_unlabeled = accuracy_score(y_true[~labeled_mask], y_predicted[~labeled_mask])

print(f"\n📊 Results:")
print(f"  Overall accuracy:    {accuracy_all*100:.1f}%")
print(f"  Accuracy on unlabeled points: {accuracy_unlabeled*100:.1f}%")
print(f"\n✅ With only {labeled_count} labeled points, the model correctly")
print(f"   classified {accuracy_unlabeled*100:.1f}% of the {unlabeled_count} unlabeled points!")
