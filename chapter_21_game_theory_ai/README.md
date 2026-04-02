<div align="center">
  <img src="cover.png" alt="Game Theory for AI Cover" width="800"/>
</div>

# Chapter 21: Game Theory for AI

**🎯 The Big Goal:** Understand the mathematical foundations of strategic decision-making — Nash Equilibrium, the Prisoner's Dilemma, and minimax strategies — and see how these concepts underpin multi-agent AI systems.

## Core Concepts

**Game Theory** studies how rational agents make decisions when their outcomes depend on each other's actions. It's the mathematical backbone of competitive AI, mechanism design, and adversarial reasoning.

### Key Concepts

- **Players:** The decision-making agents.
- **Strategies:** The set of actions each player can take.
- **Payoff Matrix:** The reward each player receives for each combination of strategies.
- **Nash Equilibrium:** A state where no player can improve their payoff by changing their strategy alone — everyone is playing their best response to everyone else.

### The Prisoner's Dilemma

Two suspects are arrested. Each can Cooperate (stay silent) or Defect (betray the other):
- Both cooperate → mild punishment (-1, -1)
- Both defect → moderate punishment (-5, -5)
- One defects, one cooperates → defector goes free (0), cooperator gets max (-10)

The **Nash Equilibrium** is (Defect, Defect) even though (Cooperate, Cooperate) gives a better joint outcome. This is the tragedy — rational individual behavior leads to collectively worse results.

### Tit-for-Tat Strategy

In repeated games, **Tit-for-Tat** is famously effective: cooperate first, then mirror the opponent's last move. It's simple, forgiving, and retaliatory — all desirable properties for robust long-term strategy.

---

## 🤔 Reflection Questions

<details>
<summary>💡 View Answer: How does game theory connect to multi-agent reinforcement learning?</summary>

In multi-agent RL, each agent's environment includes other learning agents, making the "environment" non-stationary. Game theory provides the mathematical framework for analyzing these interactions — Nash Equilibria define stable outcomes, and solution concepts like correlated equilibrium and stackelberg games guide the design of learning algorithms that converge to desirable outcomes.
</details>

<details>
<summary>💡 View Answer: What is the difference between zero-sum and non-zero-sum games?</summary>

In **zero-sum** games (chess, Go), one player's gain is exactly the other's loss — the payoffs sum to zero. In **non-zero-sum** games (business negotiations, climate agreements), it's possible for all players to gain or all to lose. Most real-world interactions are non-zero-sum, which is why cooperation strategies like Tit-for-Tat can outperform pure defection.
</details>

---

## 🐳 Hands-On Exercise: Nash Equilibrium & Iterated Prisoner's Dilemma

### Step 1: Build
```bash
cd exercise
docker build -t ch21-game-theory .
```

### Step 2: Run
```bash
docker run --rm ch21-game-theory
```

### Dockerfile
```dockerfile
FROM python:3.9-alpine
WORKDIR /app
RUN pip install numpy
COPY game_theory.py /app/
CMD ["python", "game_theory.py"]
```
