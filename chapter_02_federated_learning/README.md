<div align="center">
  <img src="cover.png" alt="Federated Learning Basics Cover" width="800"/>
</div>

# Chapter 2: Federated Learning Basics

**🎯 The Big Goal:** Understand how models can be trained collaboratively across many decentralized devices without ever moving or sharing the raw, private user data.

## Core Concepts

Federated Learning (FL) is a machine learning setting where many clients (e.g., mobile devices, hospital servers) collaboratively train a model under the orchestration of a central server. Crucially, the training data remains decentralized and private.

### The Standard Federated Averaging Algorithm
Instead of sending your personal data to a centralized Cloud to train an AI, what if the Cloud sent the AI to your phone instead?
1. **Broadcast**: A central server sends the current global model to multiple participating nodes.
2. **Local Training**: Each node uses its own local (private) data to train the model for a few iterations.
3. **Aggregation**: Each node sends only the **learned weights** (not the data) back to the server. The server computes the average of these weights and updates the global model.

### Why is this revolutionary?
Because learning can happen across millions of diverse endpoints securely. Healthcare institutions can train medical diagnostic AIs collaboratively without violating patient privacy laws (like HIPAA). 

---

## 🤔 Reflection Questions

<details>
<summary>💡 View Answer: In Federated Learning, what exactly is sent across the network?</summary>

Only the **model parameters** (weights and biases) or **gradients** are transmitted across the network. Raw user data never leaves the local device.
</details>

<details>
<summary>💡 View Answer: What happens if a client's device loses power during training?</summary>

Federated Learning is designed to be highly resilient to node dropouts. If a client goes offline, the central server simply ignores it and aggregates the updates from the remaining active and responding clients.
</details>

---

## Hands-On Exercise: Federated Averaging Simulation

In this exercise, you will run a simple simulation of the `FedAvg` algorithm. We will simulate a global server and 3 local clients. Each client trains a simple NumPy model on different local data, and the server averages their learned weights. 

### Step 1: Build the Docker Environment
Navigate to the `exercise` folder and run:
```bash
cd exercise
docker build -t ch2-federated-learning .
```

### Step 2: Run the Simulation
```bash
docker run --rm ch2-federated-learning
```

You will see exactly how a continuous cycle of sending global weights, local training, and global averaging slowly brings all models into consensus while preserving data isolation!


### Source Code

```python
import numpy as np

# A tiny linear model y = mx + c
class SimpleLinearModel:
    def __init__(self):
        self.weights = np.random.randn(1)
        self.bias = np.random.randn(1)
    
    def predict(self, X):
        return self.weights * X + self.bias

    def train(self, X, Y, epochs=100, lr=0.005):
        for _ in range(epochs):
            predictions = self.predict(X)
            errors = predictions - Y
            
            # Gradients
            dw = np.mean(errors * X)
            db = np.mean(errors)
            
            # Update
            self.weights -= lr * dw
            self.bias -= lr * db

# Three isolated clients with completely different, local (private) data trying to learn y = 2x + 1
client_data = [
    (np.array([1, 2, 3]), np.array([3, 5, 7])),     # Client 1 Data
    (np.array([4, 5, 6]), np.array([9, 11, 13])),   # Client 2 Data
    (np.array([7, 8, 9]), np.array([15, 17, 19]))   # Client 3 Data
]

global_model = SimpleLinearModel()

print(f"Initial Global Model: Weights={global_model.weights[0]:.2f}, Bias={global_model.bias[0]:.2f}")

# Train for 5 Federated Rounds
for round_num in range(1, 6):
    print(f"\n--- Federated Round {round_num} ---")
    
    client_weights = []
    client_biases = []

    for idx, (X, Y) in enumerate(client_data):
        # 1. Download Global Model to local Client
        local_model = SimpleLinearModel()
        local_model.weights = np.copy(global_model.weights)
        local_model.bias = np.copy(global_model.bias)
        
        # 2. Train Locally
        local_model.train(X, Y, epochs=10)
        print(f" Client {idx + 1} Local Update -> Weights: {local_model.weights[0]:.2f}, Bias: {local_model.bias[0]:.2f}")
        
        # 3. Store the updates to be sent back (Wait, Data is NOT sent!)
        client_weights.append(local_model.weights)
        client_biases.append(local_model.bias)
    
    # Federated Averaging (FedAvg): Aggregation Step at Server
    global_model.weights = np.mean(client_weights, axis=0)
    global_model.bias = np.mean(client_biases, axis=0)
    
    print(f"Aggregated Global Model -> Weights: {global_model.weights[0]:.2f}, Bias: {global_model.bias[0]:.2f}")

print("\n(Notice how the global model approaches the true function y = 2x + 1 collaboratively!)")
```
