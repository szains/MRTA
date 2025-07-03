import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar

class Drawer(object):
    def __init__(self, path_planner):
        self.path_planner = path_planner
        self.parameters = path_planner.parameters

    def get_map_drawing(self, robots, tasks, legend=False, labels=False):
        fig = plt.figure(figsize=(7, 7))
        #fig=plt.figure(figsize=(7,6))         
        ax = fig.gca()

        ax.set_xlim([0 - 0.5, self.parameters.size[0] - 1])
        ax.set_ylim([0 - 0.5, self.parameters.size[1] - 1])

        xticks = np.arange(0, self.parameters.size[0])
        yticks = np.arange(0, self.parameters.size[1])
        xgrid = np.arange(-0.5, self.parameters.size[0] + 0.5)
        ygrid = np.arange(-0.5, self.parameters.size[1] + 0.5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticks(xgrid, minor=True)
        ax.set_yticks(ygrid, minor=True)
        ax.tick_params(axis='both', which='major', labelsize=10)

        self.draw_hazard_heat_map(fig, ax, self.parameters.N - 1)

        # Change color of obstacles and their positions
        x_obsticles = [e[0] for e in self.parameters.obsticles]
        y_obsticles = [e[1] for e in self.parameters.obsticles]
        plt.scatter(x_obsticles, y_obsticles, color='gray', marker="s", s=300, label="Obstacles")  # Orange color for obstacles

        # Change color and positions for hazards
        x_hazards = []
        y_hazards = []
        for hazard in self.parameters.hazards:
            x_hazards = x_hazards + [e[0] for e in hazard.y_0]
            y_hazards = y_hazards + [e[1] for e in hazard.y_0]
            if labels:
                for y_0_h in hazard.y_0:
                    plt.text(y_0_h[0], y_0_h[1], str(hazard.id), color='w', fontsize=9, fontweight='bold', horizontalalignment='center', verticalalignment='center')
        plt.scatter(x_hazards, y_hazards, color='black', marker="o", s=200, label="Hazards")  # Purple color for hazards

        # Change color and positions for robots
        x_robots = []
        y_robots = []
        for robot in self.parameters.robots:
            x_robots = x_robots + [robot.x_0[0]]
            y_robots = y_robots + [robot.x_0[1]]
            if labels:
                plt.text(robot.x_0[0], robot.x_0[1], str(robot.id), color='w', fontsize=9, fontweight='bold', horizontalalignment='center', verticalalignment='center')
        plt.scatter(x_robots, y_robots, color='green', marker="o", s=200, label="Robots")  # Blue color for robots

        # Change color and positions for tasks
        x_tasks = []
        y_tasks = []
        for task in tasks:
            x = task.target[0]
            y = task.target[1]
            urgency = getattr(task, 'urgency', 1.0)

            color = plt.cm.Reds(min(urgency / 10.0, 1.0))
            plt.scatter(x, y, color=color, marker="o", s=200)

            plt.text(x, y + 1.0, f"U:{task.urgency:.1f}\nT:{task.time_waited}",
            fontsize=7, color='gray', ha='center', va='center', zorder=3)


            #Add time waited label just above the task
            # plt.text(x, y + 0.7, f"T:{task.time_waited}", fontsize=6, color='gray', ha='center')

            if labels:
                plt.text(x, y, str(task.id), color='white', fontsize=9, weight='bold', ha='center', va='center')



        # Change goal color and position
        x_goal = self.parameters.goal[0]
        y_goal = self.parameters.goal[1]
        plt.scatter(x_goal, y_goal, color='cyan', marker=">", s=250, label="Goal")  # Cyan color for goal

        plt.grid(which='minor', linestyle="--", color='lightgray', lw=0.8)  # Grid lines
        if legend:
            legend = plt.legend(bbox_to_anchor=(0., 1.05, 1., 0.05), loc='lower left', ncol=5, mode="expand", borderaxespad=0., handletextpad=0.4, fontsize=12, markerscale=0.8)
            sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=1, vmax=10))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Task Urgency")
        return fig, ax

    def draw_hazard_heat_map(self, fig, ax, k):
        samples_k = self.path_planner.y_sampler.episodes[:, k, :]
        probabilities_T = np.mean(samples_k, axis=0)
        hazard_heat_map = np.zeros((self.parameters.size[1], self.parameters.size[0]))
        for i, x in enumerate(self.path_planner.X):
            hazard_heat_map[x[1], x[0]] = probabilities_T[i]
        X, Y = np.meshgrid(np.arange(self.parameters.size[0] + 1), np.arange(self.parameters.size[1] + 1))
        heat_map = ax.pcolor(X - 0.5, Y - 0.5, hazard_heat_map, cmap='Reds', vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(heat_map, ax=ax, orientation='horizontal', pad=0.07)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(label="Probability of cell contamination at time step k=" + str(k + 1), size=12)

    def draw_full_example(self):
        fig, ax = self.get_map_drawing(self.parameters.robots, self.parameters.tasks, legend=True, labels=True)
        plt.show()

    def draw_path_for_robot(self, path, robot, tasks):
        fig, ax = self.get_map_drawing([robot], tasks, legend=True, labels=True)

        N = len(path.domain[0])
        for t in range(1, N):
            x_t_1 = path.domain[1][np.where(path.matrix[t - 1, :])[0][0]]
            x_t = path.domain[1][np.where(path.matrix[t, :])[0][0]]
            if x_t != x_t_1:
                d_t = np.array(x_t) - np.array(x_t_1)
                d_t_side = np.array([[0, 1], [-1, 0]]).dot(d_t)
                d_t = tuple(0.7 * d_t)
                x_t_1_side = tuple(0.1 * d_t_side + np.array(x_t_1))
                plt.arrow(x_t_1_side[0], x_t_1_side[1], d_t[0], d_t[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
        plt.show()

    def draw_path_for_all_robots(self, paths, robots, tasks):
        fig, ax = self.get_map_drawing(robots, tasks, legend=True, labels=True)

        # Draw each robot's path
        for i, robot in enumerate(robots):
            path = paths[i]
            N = len(path.domain[0])

            d_list = []
            x_list = []
            for t in range(0, N):
                if t < N - 1:
                    x_t = path.domain[1][np.where(path.matrix[t, :])[0][0]]
                    x_t_1 = path.domain[1][np.where(path.matrix[t + 1, :])[0][0]]
                    if x_t_1 != x_t:
                        d_t = np.array(x_t_1) - np.array(x_t)
                        d_list.append(d_t)
                        x_list.append(x_t)
                else:
                    x_t = path.domain[1][np.where(path.matrix[t, :])[0][0]]
                    x_list.append(x_t)

            shift = 0.075 + i * 0.15

            e_0 = np.array([[0, 1], [-1, 0]]).dot(d_list[0])
            x_sh_0 = x_list[0] + shift * e_0
            x_sh_list = [x_sh_0]

            for t in range(1, len(d_list)):
                d_t_1 = d_list[t - 1]
                d_t = d_list[t]
                if np.all(np.array([[1, 0], [0, 1]]).dot(d_t_1) - d_t == 0):
                    e_t = np.array([[0, 1], [-1, 0]]).dot(d_t)
                    x_sh_t = x_list[t] + shift * e_t
                    x_sh_list.append(x_sh_t)
                elif np.all(np.array([[0, 1], [-1, 0]]).dot(d_t_1) - d_t == 0):
                    e_t = -d_t_1 + d_t
                    x_sh_t = x_list[t] + shift * e_t
                    x_sh_list.append(x_sh_t)
                elif np.all(np.array([[0, -1], [1, 0]]).dot(d_t_1) - d_t == 0):
                    e_t = d_t_1 - d_t
                    x_sh_t = x_list[t] + shift * e_t
                    x_sh_list.append(x_sh_t)
                else:
                    e_t_1 = d_t_1 + np.array([[0, 1], [-1, 0]]).dot(d_t_1)
                    e_t_2 = d_t_1 + np.array([[0, 1], [-1, 0]]).dot(d_t)
                    x_sh_t_1 = x_list[t] + shift * e_t_1
                    x_sh_t_2 = x_list[t] + shift * e_t_2
                    x_sh_list.extend([x_sh_t_1, x_sh_t_2])

            e_last = np.array([[0, 1], [-1, 0]]).dot(d_list[-1])
            x_sh_last = x_list[-1] + shift * e_last
            x_sh_list.append(x_sh_last)

            x_sh_list_x = [e[0] for e in x_sh_list]
            x_sh_list_y = [e[1] for e in x_sh_list]

            plt.plot(x_sh_list_x, x_sh_list_y, zorder=2, color='b', linestyle=robot.linestyle, linewidth=2)

        # Draw assignment arrows + labels (outside path loop to avoid repetition)
        for robot in robots:
            if hasattr(robot, 'assigned_task') and robot.assigned_task:
                x0, y0 = robot.x_0
                xt, yt = robot.assigned_task.target
                dx, dy = xt - x0, yt - y0

                plt.arrow(x0, y0, dx, dy, head_width=0.3, head_length=0.3,
                        fc='purple', ec='purple', linestyle='--', alpha=0.6, zorder=1)

                mid_x = (x0 + xt) / 2
                mid_y = (y0 + yt) / 2
                plt.text(mid_x, mid_y + 0.7, f"R{robot.id}", fontsize=7,
                        color='purple', ha='center', va='center', zorder=2)

        plt.show()

    def draw_path_for_all_robots_step_by_step(self, paths, robots, tasks, k):
        fig, ax = self.get_map_drawing_step_by_step(robots, tasks, k, legend=True, labels=True)

        if k > 0:
            for i, robot in enumerate(robots):
                path = paths[i]
                N = len(path.domain[0])

                d_list = []
                x_list = []
                for t in range(N):
                    x_t = path.domain[1][np.where(path.matrix[t, :])[0][0]]
                    if t < N - 1:
                        x_t_1 = path.domain[1][np.where(path.matrix[t + 1, :])[0][0]]
                        if x_t_1 != x_t:
                            d_t = np.array(x_t_1) - np.array(x_t)
                            d_list.append(d_t)
                            x_list.append(x_t)
                    else:
                        x_list.append(x_t)

                shift = 0.075 + (len(robots) - 1 - i) * 0.15
                e_0 = np.array([[0, 1], [-1, 0]]).dot(d_list[0])
                x_sh_0 = x_list[0] + shift * e_0
                x_sh_list = [x_sh_0]

                for t in range(1, min(k, len(d_list))):
                    d_t_1 = d_list[t - 1]
                    d_t = d_list[t]
                    if np.all(np.array([[1, 0], [0, 1]]).dot(d_t_1) - d_t == 0):
                        e_t = np.array([[0, 1], [-1, 0]]).dot(d_t)
                        x_sh_list.append(x_list[t] + shift * e_t)
                    elif np.all(np.array([[0, 1], [-1, 0]]).dot(d_t_1) - d_t == 0):
                        e_t = -d_t_1 + d_t
                        x_sh_list.append(x_list[t] + shift * e_t)
                    elif np.all(np.array([[0, -1], [1, 0]]).dot(d_t_1) - d_t == 0):
                        e_t = d_t_1 - d_t
                        x_sh_list.append(x_list[t] + shift * e_t)
                    else:
                        e_t_1 = d_t_1 + np.array([[0, 1], [-1, 0]]).dot(d_t_1)
                        e_t_2 = d_t_1 + np.array([[0, 1], [-1, 0]]).dot(d_t)
                        x_sh_list.extend([x_list[t] + shift * e_t_1, x_list[t] + shift * e_t_2])

                if min(k, len(d_list)) == len(d_list):
                    e_last = np.array([[0, 1], [-1, 0]]).dot(d_list[-1])
                    x_sh_last = x_list[-1] + shift * e_last
                    x_sh_list.append(x_sh_last)

                x_sh_list_x = [e[0] for e in x_sh_list]
                x_sh_list_y = [e[1] for e in x_sh_list]

                plt.plot(x_sh_list_x, x_sh_list_y, zorder=2, color='b',
                        linestyle=self.parameters.robot_linestyles[len(robots) - 1 - i], linewidth=2)

            # ⬇️ Move this outside the per-robot loop
            for robot in robots:
                if hasattr(robot, 'assigned_task') and robot.assigned_task:
                    x0, y0 = robot.x_0
                    xt, yt = robot.assigned_task.target
                    dx, dy = xt - x0, yt - y0

                    plt.arrow(x0, y0, dx, dy, head_width=0.3, head_length=0.3,
                            fc='purple', ec='purple', linestyle='--', alpha=0.6, zorder=1)

                    mid_x = (x0 + xt) / 2
                    mid_y = (y0 + yt) / 2
                    plt.text(mid_x, mid_y + 0.7, f"R{robot.id}", fontsize=7,
                            color='purple', ha='center', va='center', zorder=2)

        plt.show()


    def get_map_drawing_step_by_step(self, robots, tasks, k, legend=False, labels=False):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.gca()

        ax.set_xlim([0 - 0.5, self.parameters.size[0] - 1])
        ax.set_ylim([0 - 0.5, self.parameters.size[1] - 1])

        xticks = np.arange(0, self.parameters.size[0])
        yticks = np.arange(0, self.parameters.size[1])
        xgrid = np.arange(-0.5, self.parameters.size[0] + 0.5)
        ygrid = np.arange(-0.5, self.parameters.size[1] + 0.5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticks(xgrid, minor=True)
        ax.set_yticks(ygrid, minor=True)
        ax.tick_params(axis='both', which='major', labelsize=10)

        self.draw_hazard_heat_map(fig, ax, k)

        x_obsticles = [e[0] for e in self.parameters.obsticles]
        y_obsticles = [e[1] for e in self.parameters.obsticles]
        plt.scatter(x_obsticles, y_obsticles, color='k', marker="s", s=300, label="Obstacles")

        x_hazards = []
        y_hazards = []
        for hazard in self.parameters.hazards:
            x_hazards = x_hazards + [e[0] for e in hazard.y_0]
            y_hazards = y_hazards + [e[1] for e in hazard.y_0]
            if labels:
                for y_0_h in hazard.y_0:
                    plt.text(y_0_h[0], y_0_h[1], str(hazard.id), color='w', fontsize=9, fontweight='bold', horizontalalignment='center', verticalalignment='center')
        plt.scatter(x_hazards, y_hazards, color='r', marker="o", s=200, label="Hazards")

        x_robots = []
        y_robots = []
        for robot in self.parameters.robots:
            x_robots = x_robots + [robot.x_0[0]]
            y_robots = y_robots + [robot.x_0[1]]
            if labels:
                plt.text(robot.x_0[0], robot.x_0[1], str(robot.id), color='w', fontsize=9, fontweight='bold', horizontalalignment='center', verticalalignment='center')
        plt.scatter(x_robots, y_robots, color='b', marker="o", s=200, label="Robots")

        x_tasks = []
        y_tasks = []

        for task in tasks:
            x = task.target[0]
            y = task.target[1]
            urgency = getattr(task, 'urgency', 1.0)

            color = plt.cm.Reds(min(urgency / 10.0, 1.0))
            plt.scatter(x, y, color=color, marker="o", s=200)
            plt.text(x, y + 1.0, f"U:{task.urgency:.1f}\nT:{task.time_waited}",
            fontsize=7, color='gray', ha='center', va='center', zorder=3)


            # Add time waited label just above the task
            # plt.text(x, y + 0.7, f"T:{task.time_waited}", fontsize=6, color='gray', ha='center')

            if labels:
                plt.text(x, y, str(task.id), color='white', fontsize=9, weight='bold', ha='center', va='center')



        x_goal = self.parameters.goal[0]
        y_goal = self.parameters.goal[1]
        plt.scatter(x_goal, y_goal, color='g', marker=">", s=250, label="Goal")

        plt.grid(which='minor', linestyle=":", color='k', lw=0.5)  # ,marker="D")
        if legend:
            legend = plt.legend(bbox_to_anchor=(0., 1.05, 1., 0.05), loc='lower left', ncol=5, mode="expand", borderaxespad=0., handletextpad=0.4, fontsize=12, markerscale=0.8)
            sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=1, vmax=10))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Task Urgency")
        return fig, ax
