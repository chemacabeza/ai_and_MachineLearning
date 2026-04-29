<div align="center">
  <img src="cover.png" alt="Metric Learning Cover" width="800"/>
</div>

# Chapter 19: Metric Learning

**🎯 The Big Goal:** Learn how to train neural networks to measure similarity — producing embeddings where similar items are close together and different items are far apart, enabling face recognition, recommendation systems, and few-shot learning.

## Core Concepts

Traditional classification assigns each input to one of N fixed classes. But what if you need to recognize a new person's face without retraining the model? Or find products similar to one a customer liked? **Metric Learning** solves this by learning a distance function.

### The Embedding Space

Instead of outputting class labels, a metric learning model outputs a vector (embedding) that captures the "essence" of the input. In this learned space:
- Two photos of the same person → embeddings close together
- Photos of different people → embeddings far apart

### Triplet Loss

The most popular training strategy uses "triplets" of examples:
- **Anchor (A):** A reference example (e.g., photo of Alice)
- **Positive (P):** Another example of the same class (another photo of Alice)
- **Negative (N):** An example of a different class (photo of Bob)

The loss function: `L = max(0, d(A,P) - d(A,N) + margin)`

This pushes the model to make d(A,P) small (same person = close) and d(A,N) large (different person = far), with a safety margin.

---

## 🤔 Reflection Questions

<details>
<summary>💡 View Answer: How does face recognition work using metric learning?</summary>

During enrollment, the system takes a few photos of each person and stores their embeddings. During recognition, it computes the embedding of the new face and finds the nearest stored embedding using cosine similarity or Euclidean distance. If the distance is below a threshold, the face is identified. This approach scales to millions of identities without retraining — you just add new embeddings to the database.
</details>

<details>
<summary>💡 View Answer: What is the difference between contrastive loss and triplet loss?</summary>

**Contrastive loss** works with pairs: pull similar pairs together, push dissimilar pairs apart. **Triplet loss** works with triplets (anchor, positive, negative) and directly enforces that the positive is closer than the negative. Triplet loss is generally more effective because it considers relative distances rather than absolute distances, but it requires careful "hard negative mining" — finding the most challenging negative examples for effective training.
</details>

---

## 🐳 Hands-On Exercise: Triplet Loss Embedding

### Step 1: Build
```bash
cd exercise
docker build -t ch19-metric .
```

### Step 2: Run
```bash
docker run --rm ch19-metric
```

### Dockerfile
```dockerfile
FROM python:3.9-alpine
WORKDIR /app
RUN pip install numpy
COPY metric_learning.py /app/
CMD ["python", "metric_learning.py"]
```

### Source Code

```python
import numpy as np
print("=== Metric Learning: Triplet Loss Embedding ===\n")
np.random.seed(42)
class_data = {
    "Cat":  np.random.randn(5, 8) + np.array([2, 0, 0, 0, 0, 0, 0, 0]),
    "Dog":  np.random.randn(5, 8) + np.array([0, 2, 0, 0, 0, 0, 0, 0]),
    "Bird": np.random.randn(5, 8) + np.array([0, 0, 2, 0, 0, 0, 0, 0]),
}
W = np.random.randn(8, 2) * 0.3
lr = 0.01
margin = 1.0
print("Training with Triplet Loss (100 iterations)...")
for epoch in range(100):
    total_loss = 0
    for cls_name in class_data:
        others = [k for k in class_data if k != cls_name]
        for i in range(len(class_data[cls_name])):
            anchor = class_data[cls_name][i]
            pos_idx = (i + 1) % len(class_data[cls_name])
            positive = class_data[cls_name][pos_idx]
            neg_cls = np.random.choice(others)
            negative = class_data[neg_cls][np.random.randint(len(class_data[neg_cls]))]
            a_emb = anchor @ W
            p_emb = positive @ W
            n_emb = negative @ W
            d_pos = np.sum((a_emb - p_emb)**2)
            d_neg = np.sum((a_emb - n_emb)**2)
            loss = max(0, d_pos - d_neg + margin)
            total_loss += loss
            if loss > 0:
                grad_a = 2*(a_emb - p_emb) - 2*(a_emb - n_emb)
                W -= lr * np.outer(anchor, grad_a)
    if (epoch+1) % 25 == 0:
        print(f"  Epoch {epoch+1}: Loss = {total_loss:.3f}")
print("\n📊 Learned 2D Embeddings:")
for cls_name in class_data:
    embeddings = class_data[cls_name] @ W
    centroid = embeddings.mean(axis=0)
    print(f"  {cls_name:5s} centroid: ({centroid[0]:+.2f}, {centroid[1]:+.2f})")
print("\n📏 Inter-class distances:")
classes = list(class_data.keys())
for i in range(len(classes)):
    for j in range(i+1, len(classes)):
        c1 = (class_data[classes[i]] @ W).mean(axis=0)
        c2 = (class_data[classes[j]] @ W).mean(axis=0)
        dist = np.linalg.norm(c1 - c2)
        print(f"  {classes[i]:5s} ↔ {classes[j]:5s}: {dist:.3f}")
print("\n✅ Similar items are close, different items are far apart!")
```
