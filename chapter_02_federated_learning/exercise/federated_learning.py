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
