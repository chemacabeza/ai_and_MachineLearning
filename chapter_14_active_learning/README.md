<div align="center">
  <img src="cover.png" alt="Active Learning Cover" width="800"/>
</div>

# Chapter 14: Active Learning

**🎯 The Big Goal:** Understand how a machine learning model can intelligently choose which data points to ask a human to label — minimizing labeling cost while maximizing model accuracy.

## Core Concepts

Labeling data is the most expensive part of any ML project. A medical AI might need 100,000 X-rays labeled by radiologists at $10 each. **Active Learning** flips the script: instead of labeling everything, the model itself picks the most informative samples to label.

### The Active Learning Loop

1. **Train** on the small set of currently labeled data.
2. **Predict** on the unlabeled pool.
3. **Query** — select the most "uncertain" predictions and ask the human to label them.
4. **Add** the newly labeled data to the training set.
5. **Repeat** until the model reaches target accuracy.

### Query Strategies

How does the model decide which sample is most informative?

- **Uncertainty Sampling:** Pick the sample the model is least certain about. If the model predicts 50%/50% cat vs. dog, that sample is maximally informative.
- **Query-by-Committee:** Train multiple models. Pick the sample they disagree on most.
- **Expected Model Change:** Pick the sample that would change the model's parameters the most if labeled.

### Why It Works

Not all data points are equally valuable. A model that already knows how to classify easy examples gains nothing from seeing more easy examples. But one confusing example near the decision boundary can shift the boundary dramatically.

---

## 🤔 Reflection Questions

<details>
<summary>💡 View Answer: How much labeling effort can Active Learning save in practice?</summary>

Research consistently shows Active Learning can achieve the same accuracy as random sampling while using **50–80% fewer labels**. In production systems, this translates to enormous cost savings. For example, a model that needs 10,000 randomly labeled examples might need only 2,000–5,000 actively selected examples to reach the same performance.
</details>

<details>
<summary>💡 View Answer: What is the "cold start" problem in Active Learning?</summary>

At the very beginning, the model has no labeled data at all, so it cannot make meaningful uncertainty estimates. The initial queries are essentially random. Solutions include starting with a small random seed set, using diversity-based sampling for the first batch, or employing pre-trained embeddings to find a representative initial set.
</details>

---

## 🐳 Hands-On Exercise: Uncertainty-Based Active Learning

This exercise demonstrates active learning by training a classifier that strategically selects which samples to label, compared against random selection.

### Step 1: Build the Docker Environment
```bash
cd exercise
docker build -t ch14-active-learning .
```

### Step 2: Run
```bash
docker run --rm ch14-active-learning
```

### Source Code

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=== Active Learning: Uncertainty Sampling vs Random ===\n")

# Generate a classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_redundant=2, random_state=42)
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
    labels = 10 + i * 5
    print(f"{i+1:<8} {active_acc[i]*100:>9.1f}% {random_acc[i]*100:>9.1f}%")

print(f"\n✅ Active Learning reaches {active_acc[-1]*100:.1f}% accuracy")
print(f"   Random sampling reaches {random_acc[-1]*100:.1f}% accuracy")
print(f"   Both used {10 + 20*5} labeled samples (from 800 available)")
```

### Dockerfile

```dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN pip install numpy scikit-learn
COPY active_learning.py /app/
CMD ["python", "active_learning.py"]
```
