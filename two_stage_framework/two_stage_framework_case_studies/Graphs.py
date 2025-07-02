import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def plot_task_completion(task_data):
    time_steps = list(range(len(task_data)))
    plt.plot(time_steps, task_data, marker='o', linestyle='-', color='blue')
    plt.xlabel("Time Step")
    plt.ylabel("Tasks Completed")
    plt.title("Task Completion Rate Over Time")
    plt.grid()
    plt.show()


def plot_robot_utilization(robot_data):
    robot_ids = list(robot_data.keys())
    utilizations = list(robot_data.values())

    plt.bar(robot_ids, utilizations, color='green')
    plt.xlabel("Robot ID")
    plt.ylabel("Utilization (%)")
    plt.title("Robot Utilization Over Time")
    plt.show()


def plot_hazard_avoidance(hazard_data):
    time_steps = list(range(len(hazard_data)))
    plt.plot(time_steps, hazard_data, marker='o', linestyle='-', color='red')
    plt.xlabel("Time Step")
    plt.ylabel("Hazards Avoided")
    plt.title("Hazard Avoidance Success Rate")
    plt.grid()
    plt.show()


def plot_hazard_avoidance(hazard_data):
    import matplotlib.pyplot as plt

    time_steps = list(range(len(hazard_data)))
    plt.plot(time_steps, hazard_data, marker='o', linestyle='-', color='red')
    plt.xlabel("Time Step")
    plt.ylabel("Hazard Presence (cells)")
    plt.title("Hazard Spread Over Time")
    plt.grid()

    # Annotate spikes
    mean = np.mean(hazard_data)
    std = np.std(hazard_data)
    threshold = mean + 1.5 * std  # Customize sensitivity

    for t, val in zip(time_steps, hazard_data):
        if val > threshold:
            plt.annotate("Spike!", xy=(t, val), xytext=(t, val + 2),
                         arrowprops=dict(arrowstyle='->', color='blue'), fontsize=8)

    plt.show()



def plot_priority_completion(allocation, task_priority, robot_ids):
    # Build a robot vs. priority count map
    priority_counts = {robot: defaultdict(int) for robot in robot_ids}

    for robot, task in allocation:
        priority = task_priority.get(task, 1)
        priority_counts[robot][priority] += 1

    # Prepare data for stacked bar plot
    all_priorities = sorted({p for counts in priority_counts.values() for p in counts})
    bottom = [0] * len(robot_ids)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_priorities)))

    plt.figure(figsize=(8, 5))
    for i, p in enumerate(all_priorities):
        values = [priority_counts[robot].get(p, 0) for robot in robot_ids]
        plt.bar(robot_ids, values, bottom=bottom, label=f"Priority {p}", color=colors[i])
        bottom = [sum(x) for x in zip(bottom, values)]

    plt.title("Tasks Completed by Priority Level per Robot")
    plt.xlabel("Robot ID")
    plt.ylabel("Number of Tasks")
    plt.legend(title="Priority Level")
    plt.tight_layout()
    plt.show()


def plot_task_priority_heatmap(task_objects, task_priority):
    import matplotlib.pyplot as plt

    xs = [task.target[0] for task in task_objects]
    ys = [task.target[1] for task in task_objects]
    priorities = [task_priority.get(task.id, 1) for task in task_objects]

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(xs, ys, c=priorities, cmap='plasma', s=120, edgecolors='black')
    plt.colorbar(scatter, label="Task Priority")
    plt.title("Spatial Distribution of Task Priorities")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
