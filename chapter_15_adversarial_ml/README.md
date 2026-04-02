<div align="center">
  <img src="cover.png" alt="Adversarial ML Cover" width="800"/>
</div>

# Chapter 15: Adversarial Machine Learning

**🎯 The Big Goal:** Understand how tiny, invisible perturbations to input data can fool state-of-the-art neural networks — and learn the fundamental attack and defense strategies that define AI security.

## Core Concepts

A neural network that classifies images with 99% accuracy can be tricked into making absurd mistakes by adding imperceptible noise. This is **Adversarial Machine Learning** — the study of how ML models can be attacked and defended.

### The FGSM Attack

The **Fast Gradient Sign Method** (Goodfellow et al., 2015) is the simplest and most famous adversarial attack:

1. Take a correctly classified image (e.g., a panda classified at 98% confidence).
2. Compute the gradient of the loss with respect to the input pixels (not the weights).
3. Add a tiny amount of noise in the direction that maximizes the loss.
4. The result looks identical to humans but fools the model completely.

Formula: `x_adv = x + ε × sign(∇x Loss(model(x), y_true))`

Where `ε` is a tiny number (e.g., 0.01) controlling how much noise to add.

### Types of Attacks

| Attack Type | Description |
|-------------|-------------|
| **White-box** | Attacker has full access to the model (weights, architecture) |
| **Black-box** | Attacker can only query the model and observe outputs |
| **Targeted** | Force the model to predict a specific wrong class |
| **Untargeted** | Just make the model predict anything wrong |

### Defenses

- **Adversarial Training:** Include adversarial examples in the training set so the model learns to handle them.
- **Input Preprocessing:** Smooth, compress, or denoise inputs before feeding them to the model.
- **Gradient Masking:** Make gradients harder to compute (but attackers can work around this).

---

## 🤔 Reflection Questions

<details>
<summary>💡 View Answer: Why are adversarial attacks a real-world threat?</summary>

Adversarial attacks can compromise safety-critical systems. A self-driving car's vision system could misclassify a stop sign with a sticker on it. A facial recognition system could be fooled by adversarial glasses. Spam filters can be evaded by carefully crafted text. Any AI system that makes decisions based on external input is potentially vulnerable.
</details>

<details>
<summary>💡 View Answer: Why is adversarial training not a complete solution?</summary>

Adversarial training hardens the model against specific attack methods used during training, but new attack methods can still succeed. It's an arms race — every defense inspires a new attack. Additionally, adversarial training significantly increases training time and can reduce accuracy on clean (non-adversarial) inputs, creating a robustness-accuracy tradeoff.
</details>

---

## 🐳 Hands-On Exercise: FGSM Attack Simulation

This exercise demonstrates how tiny noise can flip a classifier's prediction, simulating the core concept of adversarial attacks using scikit-learn.

### Step 1: Build the Docker Environment
```bash
cd exercise
docker build -t ch15-adversarial-ml .
```

### Step 2: Run
```bash
docker run --rm ch15-adversarial-ml
```

### Source Code

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

print("=== Adversarial ML: FGSM-Style Attack Simulation ===\n")

X, y = make_classification(n_samples=200, n_features=5, random_state=42)
model = LogisticRegression(max_iter=200).fit(X, y)

original_acc = model.score(X, y)
print(f"Original model accuracy: {original_acc*100:.1f}%\n")

# Simulate gradient-based perturbation (FGSM concept)
epsilons = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
print(f"{'Epsilon':<10} {'Accuracy':>10} {'Flipped':>10}")
print("-" * 32)

for eps in epsilons:
    np.random.seed(42)
    # Approximate gradient direction: perturb in direction that increases loss
    coefficients = model.coef_[0]
    perturbation = eps * np.sign(coefficients)
    
    X_adv = X.copy()
    for i in range(len(X_adv)):
        if model.predict([X_adv[i]])[0] == y[i]:
            X_adv[i] += perturbation  # Push toward wrong class
    
    adv_acc = model.score(X_adv, y)
    flipped = int((original_acc - adv_acc) * len(X))
    print(f"{eps:<10.2f} {adv_acc*100:>9.1f}% {flipped:>9d}")

print(f"\n⚠️  With ε=0.5, the model's accuracy drops dramatically!")
print("   The perturbation is small but targeted at the decision boundary.")
print("\n✅ This demonstrates why adversarial robustness matters for AI safety.")
```

### Dockerfile

```dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN pip install numpy scikit-learn
COPY adversarial_attack.py /app/
CMD ["python", "adversarial_attack.py"]
```
