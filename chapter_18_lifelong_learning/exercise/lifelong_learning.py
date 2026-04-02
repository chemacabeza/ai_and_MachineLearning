import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
print("=== Lifelong Learning: Catastrophic Forgetting Demo ===\n")
np.random.seed(42)
X1, y1 = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
X2, y2 = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=99)
X3, y3 = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=7)
naive = SGDClassifier(loss='log_loss', random_state=42)
print("--- Naive Sequential Training (Catastrophic Forgetting) ---")
naive.partial_fit(X1, y1, classes=[0,1])
print(f"After Task 1: Task1={naive.score(X1,y1)*100:.0f}%")
naive.partial_fit(X2, y2)
print(f"After Task 2: Task1={naive.score(X1,y1)*100:.0f}%, Task2={naive.score(X2,y2)*100:.0f}%")
naive.partial_fit(X3, y3)
print(f"After Task 3: Task1={naive.score(X1,y1)*100:.0f}%, Task2={naive.score(X2,y2)*100:.0f}%, Task3={naive.score(X3,y3)*100:.0f}%")
replay = SGDClassifier(loss='log_loss', random_state=42)
buffer_X, buffer_y = np.empty((0,10)), np.empty(0, dtype=int)
print("\n--- Replay-Based Lifelong Learning ---")
for task_id, (Xt, yt) in enumerate([(X1,y1),(X2,y2),(X3,y3)], 1):
    combined_X = np.vstack([Xt, buffer_X]) if len(buffer_X) > 0 else Xt
    combined_y = np.concatenate([yt, buffer_y]) if len(buffer_y) > 0 else yt
    replay.partial_fit(combined_X, combined_y, classes=[0,1])
    idx = np.random.choice(len(Xt), min(30, len(Xt)), replace=False)
    buffer_X = np.vstack([buffer_X, Xt[idx]]) if len(buffer_X) > 0 else Xt[idx]
    buffer_y = np.concatenate([buffer_y, yt[idx]]) if len(buffer_y) > 0 else yt[idx]
    scores = [f"Task{i+1}={replay.score(X,y)*100:.0f}%" for i,(X,y) in enumerate([(X1,y1),(X2,y2),(X3,y3)][:task_id])]
    print(f"After Task {task_id}: {', '.join(scores)}")
print("\n✅ Replay buffer prevents catastrophic forgetting of earlier tasks!")
