import numpy as np
import copy

from Allocation import Allocation
from Drawer import Drawer
from Instrument import Instrument

class Task_Allocator(object):
    def __init__(self,function_frame):
        Allocation.set_up(function_frame)
        self.function_frame=function_frame
        self.path_planner=function_frame.path_planner
        self.robots=copy.deepcopy(function_frame.robots)
        self.tasks=copy.deepcopy(function_frame.tasks)
        # Automatically assign urgency to each task
        for task in self.tasks:
            distances = [np.linalg.norm(np.array(task.target) - np.array(r.x_0)) for r in self.robots]
            avg_distance = np.mean(distances)
            task.urgency = max(1.0, 10.0 - avg_distance)  # Normalize to a 1–10 urgency scale

        self.instrument=Instrument()

    def compute_priority_score(self, robot, task):
        # Basic example: prioritize higher urgency and closer proximity
        distance = np.linalg.norm(np.array(robot.x_0) - np.array(task.target))
        score = task.urgency - distance  # You can tune this formula later
        return score


    def greedy_assign(self):  # ← Add it right here!
        for robot in self.robots:
            if hasattr(robot, 'assigned_task') and robot.assigned_task is not None:
                continue
            best_task = None
            best_score = -np.inf
            for task in self.tasks:
                if not task.is_assigned:
                    score = self.compute_priority_score(robot, task)
                    if score > best_score:
                        best_score = score
                        best_task = task
            if best_task:
                best_task.is_assigned = True
                best_task.time_waited = 0
                robot.assigned_task = best_task

    # def solve_problem(self):
    #     pass

    def solve_problem(self):
        max_timesteps = self.path_planner.parameters.N
        for t in range(max_timesteps):
            #Update urgency for unassigned tasks
            for task in self.tasks:
                if not task.is_assigned:
                    task.time_waited += 1
                    task.urgency = min(10.0, task.urgency + 0.2 * task.time_waited)

            # Simulate task completion logic
            for robot in self.robots:
                if hasattr(robot, 'assigned_task'):
                    task = robot.assigned_task
                    distance = np.linalg.norm(np.array(robot.x_0) - np.array(task.target))

                    if distance < 1.0:  # You can adjust threshold
                        print(f"[Time {t}] Robot {robot.id} completed Task {task.id}")
                        task.is_assigned = False
                        robot.assigned_task = None
                        # Optional: Update robot position if it “moved”
                        robot.x_0 = task.target

            # Reassign tasks if any robot is idle
            self.greedy_assign()

            # 🧠 Optional: Visual or console debug
            print(f"[Time {t}] {[f'T{task.id}:{task.urgency:.1f}' for task in self.tasks]}")


            drawer = Drawer(self.path_planner)
            task_targets = [robot.assigned_task.target for robot in self.robots if hasattr(robot, 'assigned_task')]
            drawer.draw_path_for_all_robots(task_targets, self.robots, self.tasks)


    def get_allocation(self):
        pass

    def get_objective_value(self):
        pass

    def postprocess_solution(self,solution):
        solution=self.add_optimal_policies(solution)
        solution=self.add_optimal_paths(solution)
        solution=self.add_group_objective(solution)

    def show_solution(self,solution):
        solution.print_solution()
        self.draw_solution(solution)

    def add_optimal_policies(self,solution):
        print("...Calculating optimal policies...")
        allocation=solution.allocation.get_printable()
        Mu=[]
        V=[]
        V_ret=[]
        for r in self.robots:
            allocation_r=allocation.loc["robot_"+str(r.id)].to_numpy(dtype=bool)
            S_r=self.tasks.subset(allocation_r)

            targets_r=[e.target for e in S_r]
            goal_r=r.goal
            x_0_r=r.x_0

            V_ret_r,Mu_r,V_r=self.path_planner.get_solution(targets_r,goal_r,x_0_r,print_progress=True)
            Mu=Mu+[Mu_r]
            V=V+[V_r]
            V_ret=V_ret+[V_ret_r]        
        solution.Mu=Mu
        solution.V=V
        solution.V_ret=V_ret
        print("Finished calculating optimal policies!\n")
        return solution

    def add_optimal_paths(self,solution):
        print("...Calculating paths...")
        path=[]
        for i_r,r in enumerate(self.robots):
            Mu_r=solution.Mu[i_r]
            x_0_r=r.x_0
            path_r=self.path_planner.simulate_path(Mu_r,x_0_r,set([]))
            path=path+[path_r]
        solution.path=path
        print("Finished calculating paths!\n")
        return solution

    def add_group_objective(self,solution):
        print("...Calculating group objective...")
        
        E=self.path_planner.parameters.E
        group_successful_episodes=np.ones(E,dtype=bool)
        for i_r,r in enumerate(self.robots):
            path_r=solution.path[i_r]
            successful_episodes_r=self.path_planner.simulate_successful_episodes(path_r)
            group_successful_episodes=group_successful_episodes*successful_episodes_r
        group_objective_value=sum(group_successful_episodes)/E
        solution.objective_value={"multiplicative":solution.objective_value,"group":group_objective_value}
        print("Finished calculating group objective!\n")
        return solution

    def draw_solution(self,solution):
        drawer=Drawer(self.path_planner)
        drawer.draw_path_for_all_robots(solution.path,self.robots,self.tasks)

    def draw_solution_step_by_step(self,solution,k_delta):
        drawer=Drawer(self.path_planner)
        N=self.path_planner.parameters.N
        k=0
        while k<N-1:
            drawer.draw_path_for_all_robots_step_by_step(solution.path,self.robots,self.tasks,k)
            k=k+k_delta
        drawer.draw_path_for_all_robots_step_by_step(solution.path,self.robots,self.tasks,N-1)