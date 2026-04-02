import random

class AutonomousVehicle:
    def __init__(self):
        self.x, self.y = 0, 0
        self.heading = "NORTH"
        self.speed = 0
        self.steps = 0
    
    def read_sensors(self):
        return {
            "front": random.uniform(0.5, 20.0),
            "left":  random.uniform(0.5, 20.0),
            "right": random.uniform(0.5, 20.0),
            "rear":  random.uniform(2.0, 20.0),
        }
    
    def decide_action(self, sensors):
        DANGER_ZONE = 3.0
        if sensors["front"] > DANGER_ZONE:
            return "ACCELERATE", "Path clear ahead"
        elif sensors["left"] > sensors["right"] and sensors["left"] > DANGER_ZONE:
            return "TURN_LEFT", "Obstacle ahead, left is clear"
        elif sensors["right"] > DANGER_ZONE:
            return "TURN_RIGHT", "Obstacle ahead, right is clear"
        else:
            return "BRAKE", "Obstacles on all sides — emergency stop!"
    
    def execute(self, action):
        directions = ["NORTH", "EAST", "SOUTH", "WEST"]
        idx = directions.index(self.heading)
        if action == "ACCELERATE":
            self.speed = min(self.speed + 10, 60)
        elif action == "TURN_LEFT":
            self.heading = directions[(idx - 1) % 4]
            self.speed = max(self.speed - 5, 10)
        elif action == "TURN_RIGHT":
            self.heading = directions[(idx + 1) % 4]
            self.speed = max(self.speed - 5, 10)
        elif action == "BRAKE":
            self.speed = 0
        self.steps += 1

av = AutonomousVehicle()
print("=== Autonomous Vehicle Simulation ===\n")

for step in range(10):
    sensors = av.read_sensors()
    action, reason = av.decide_action(sensors)
    av.execute(action)
    print(f"Step {step+1:2d} | Sensors: F={sensors['front']:5.1f}m L={sensors['left']:5.1f}m R={sensors['right']:5.1f}m")
    print(f"         Action: {action:12s} | Reason: {reason}")
    print(f"         Speed: {av.speed} km/h | Heading: {av.heading}\n")

print(f"✅ Vehicle navigated {av.steps} steps safely!")
