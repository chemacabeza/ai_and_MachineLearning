import numpy as np

print("=== Graph Neural Networks: Message Passing ===\n")

node_names = ["Alice", "Bob", "Carol", "Dave", "Eve"]

A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
])

H = np.array([
    [0.2, 0.9],
    [0.5, 0.3],
    [0.8, 0.7],
    [0.3, 0.1],
    [0.6, 0.5],
])

print("Initial Node Features:")
for i, name in enumerate(node_names):
    print(f"  {name}: {H[i]}")

W = np.random.RandomState(42).randn(2, 2) * 0.5

for round_num in range(2):
    A_hat = A + np.eye(len(A))
    D = np.diag(1.0 / A_hat.sum(axis=1))
    H = np.tanh(D @ A_hat @ H @ W)
    print(f"\nAfter Round {round_num + 1}:")
    for i, name in enumerate(node_names):
        print(f"  {name}: [{H[i][0]:+.4f}, {H[i][1]:+.4f}]")

print("\n📊 Node Similarity (cosine) after message passing:")
for i in range(len(node_names)):
    for j in range(i+1, len(node_names)):
        cos_sim = np.dot(H[i], H[j]) / (np.linalg.norm(H[i]) * np.linalg.norm(H[j]))
        marker = "🔗" if A[i][j] == 1 else "  "
        print(f"  {marker} {node_names[i]:6s} ↔ {node_names[j]:6s}: {cos_sim:.4f}")

print("\n✅ Connected nodes (🔗) should have higher similarity!")
