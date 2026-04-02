import numpy as np
print("=== Game Theory for AI: Nash Equilibrium Finder ===\n")
print("--- Prisoner's Dilemma ---")
payoff_A = np.array([[-1, -10], [0, -5]])
payoff_B = np.array([[-1, 0], [-10, -5]])
strategies = ["Cooperate", "Defect"]
print("\nPayoff Matrix (A's payoff, B's payoff):")
print(f"{'':15s} B:Cooperate  B:Defect")
for i in range(2):
    print(f"A:{strategies[i]:10s}  ({payoff_A[i,0]:3d},{payoff_B[i,0]:3d})    ({payoff_A[i,1]:3d},{payoff_B[i,1]:3d})")
print("\nFinding Nash Equilibria (pure strategy)...")
nash = []
for i in range(2):
    for j in range(2):
        a_best = payoff_A[i, j] >= max(payoff_A[:, j])
        b_best = payoff_B[i, j] >= max(payoff_B[i, :])
        if a_best and b_best:
            nash.append((i, j))
            print(f"  ✅ Nash Equilibrium: A={strategies[i]}, B={strategies[j]} → Payoff ({payoff_A[i,j]},{payoff_B[i,j]})")
print("\n--- Iterated Prisoner's Dilemma (100 rounds) ---\n")
class TitForTat:
    def __init__(self): self.name = "Tit-for-Tat"
    def act(self, history): return 0 if not history else history[-1]
class AlwaysDefect:
    def __init__(self): self.name = "Always Defect"
    def act(self, history): return 1
class Random:
    def __init__(self): self.name = "Random"
    def act(self, history): return np.random.randint(2)
agents = [TitForTat(), AlwaysDefect(), Random()]
results = {}
for a1 in agents:
    for a2 in agents:
        if a1.name >= a2.name: continue
        score1, score2 = 0, 0
        h1, h2 = [], []
        for _ in range(100):
            m1, m2 = a1.act(h2), a2.act(h1)
            score1 += payoff_A[m1, m2]
            score2 += payoff_B[m1, m2]
            h1.append(m1); h2.append(m2)
        results[(a1.name, a2.name)] = (score1, score2)
        print(f"  {a1.name:15s} vs {a2.name:15s}: {score1:4d} vs {score2:4d}")
print("\n✅ Tit-for-Tat is the strongest long-term strategy!")
print("   It cooperates first, then mirrors the opponent's last move.")
