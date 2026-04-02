import numpy as np
import random
from collections import deque

print("=== Deep Learning for Games: Coin Collector DQN ===\n")

class CoinGrid:
    def __init__(self, size=5):
        self.size = size
        self.reset()
    def reset(self):
        self.agent = [0, 0]
        self.coin = [self.size-1, self.size-1]
        return self._get_state()
    def _get_state(self):
        return tuple(self.agent + self.coin)
    def step(self, action):
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        dr, dc = moves[action]
        self.agent[0] = max(0, min(self.size-1, self.agent[0] + dr))
        self.agent[1] = max(0, min(self.size-1, self.agent[1] + dc))
        if self.agent == self.coin:
            return self._get_state(), 10.0, True
        return self._get_state(), -1.0, False

class SimpleQNet:
    def __init__(self, state_size, action_size, hidden=32):
        self.W1 = np.random.randn(state_size, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, action_size) * 0.1
        self.b2 = np.zeros(action_size)
        self.lr = 0.01
    def predict(self, state):
        state = np.array(state, dtype=float)
        self.h = np.maximum(0, state @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2
    def train(self, state, target_q):
        pred_q = self.predict(state)
        error = pred_q - target_q
        dW2 = np.outer(self.h, error)
        db2 = error
        dh = error @ self.W2.T * (self.h > 0)
        state = np.array(state, dtype=float)
        dW1 = np.outer(state, dh)
        db1 = dh
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

env = CoinGrid(size=5)
qnet = SimpleQNet(state_size=4, action_size=4)
replay_buffer = deque(maxlen=5000)
epsilon = 1.0
gamma = 0.95
episodes = 500
wins = 0
recent_rewards = deque(maxlen=50)

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    for step in range(50):
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            q_vals = qnet.predict(state)
            action = np.argmax(q_vals)
        next_state, reward, done = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state
        if len(replay_buffer) > 32:
            s, a, r, ns, d = random.choice(replay_buffer)
            target = qnet.predict(s).copy()
            if d:
                target[a] = r
            else:
                target[a] = r + gamma * np.max(qnet.predict(ns))
            qnet.train(s, target)
        if done:
            wins += 1
            break
    recent_rewards.append(total_reward)
    epsilon = max(0.01, epsilon * 0.995)
    if (ep + 1) % 100 == 0:
        avg_reward = np.mean(recent_rewards)
        print(f"Episode {ep+1:4d} | Avg Reward: {avg_reward:6.1f} | Epsilon: {epsilon:.3f} | Wins: {wins}")

print(f"\n✅ Agent won {wins}/{episodes} episodes!")
print(f"   Final average reward: {np.mean(recent_rewards):.1f}")
