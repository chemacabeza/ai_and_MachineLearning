import random
import time

class TradingAgent:
    def __init__(self, name, strategy, starting_balance):
        self.name = name
        self.strategy = strategy
        self.balance = starting_balance
        self.inventory = 0

    def evaluate_market(self, market_price):
        if self.strategy == 'aggressive':
            # Aggressive agent buys often, sells only when profit is very high
            if self.balance > market_price and random.random() > 0.3:
                return 'buy'
            elif self.inventory > 0 and random.random() > 0.8:
                return 'sell'
                
        elif self.strategy == 'conservative':
            # Conservative agent buys only sparingly, sells instantly on minor profit
            if self.balance > market_price and random.random() > 0.7:
                return 'buy'
            elif self.inventory > 0 and random.random() > 0.4:
                return 'sell'
                
        return 'hold'

class MarketEnvironment:
    def __init__(self):
        self.price = 100
        
    def next_day(self):
        # The market fluctuates randomly each day between -15 and +15
        fluctuation = random.randint(-15, 15)
        self.price = max(10, self.price + fluctuation)
        return self.price

# Initialize Environment and Agents
market = MarketEnvironment()
agent_a = TradingAgent("Agent Alpha", "aggressive", 500)
agent_b = TradingAgent("Agent Beta", "conservative", 500)

agents = [agent_a, agent_b]

print("--- Multi-Agent Market Simulation Initiated ---")
print(f"Starting Balances -> Alpha (Aggressive): ${agent_a.balance} | Beta (Conservative): ${agent_b.balance}\n")

for day in range(1, 11):
    current_price = market.next_day()
    print(f"[Day {day}] Market Price: ${current_price}")
    
    for agent in agents:
        action = agent.evaluate_market(current_price)
        
        if action == 'buy' and agent.balance >= current_price:
            agent.balance -= current_price
            agent.inventory += 1
            print(f"  > {agent.name} decided to BUY. (Remaining Balance: ${agent.balance})")
        elif action == 'sell' and agent.inventory > 0:
            agent.balance += current_price
            agent.inventory -= 1
            print(f"  > {agent.name} decided to SELL. (New Balance: ${agent.balance})")
        else:
            print(f"  > {agent.name} is HOLDING.")
            
    print("-" * 30)
    time.sleep(0.5)

# Calculate final liquid value (Balance + Value of inventory at final price)
print("\n--- Final Results ---")
for a in agents:
    total_worth = a.balance + (a.inventory * market.price)
    profit = total_worth - 500
    print(f"{a.name} Total Worth: ${total_worth} -> Net Profit: ${profit}")
