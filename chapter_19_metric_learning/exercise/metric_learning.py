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
