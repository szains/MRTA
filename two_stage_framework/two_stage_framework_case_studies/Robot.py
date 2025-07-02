# class Robot(object):
#     def __init__(self,id,x_0,goal,linestyle):
#         self.id=id
#         self.x_0=x_0
#         self.goal=goal
#         self.linestyle=linestyle

# class Robot(object):
#     def __init__(self, id, x_0, goal, linestyle):
#         self.id = id
#         self.x_0 = x_0
#         self.goal = goal
#         self.linestyle = linestyle
        
#         # New attributes for adaptive task allocation
#         self.battery_level = 100  # Percentage, starts at full charge
#         self.sensor_health = 1.0  # Normalized (1.0 = perfect, <1.0 = degraded)
#         self.hazard_proximity = 0  # Distance from hazards
import numpy as np  # Ensure NumPy is available

def distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

class Robot(object):
    def __init__(self, id, x_0, goal, linestyle):
        self.id = id
        self.x_0 = x_0
        self.goal = goal
        self.linestyle = linestyle

        # New state attributes
        self.battery_level = 100  # Percentage
        self.sensor_health = 1.0  # Normalized (1.0 = perfect, <1.0 = degraded)
        self.hazard_proximity = 0  # Distance from hazards

    # New function to update state dynamically
    def update_robot_state(self, hazards, battery_usage):
        self.battery_level -= battery_usage  # Reduce battery each step
        self.sensor_health *= 0.99  # Simulate sensor wear
        if hazards:
            self.hazard_proximity = min([distance(self.x_0, h) for h in hazards])  # Closest hazard
