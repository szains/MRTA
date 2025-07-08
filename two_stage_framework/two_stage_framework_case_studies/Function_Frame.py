import numpy as np
import copy
import itertools

from Set import Set
from DMatrix import DMatrix
from Matrix import Matrix
from Instrument import Instrument
from ProgressBar import ProgressBar

# class Function_Frame(DMatrix):
#     def __init__(self,parameters,path_planner):
#         super().__init__()
#         self.robots=parameters.robots
#         self.tasks=parameters.tasks
#         self.path_planner=path_planner
#         self.calculating_function_frame()

class Function_Frame(DMatrix):
    def __init__(self, parameters, path_planner):
        super().__init__()
        self.parameters = parameters  # ← add this line
        self.robots = parameters.robots
        self.tasks = parameters.tasks
        self.path_planner = path_planner
        
        # Initialize hazards from parameters
        self.hazards = parameters.y_0  # Assuming `y_0` contains hazard locations

        self.calculating_function_frame()

    # New function to update hazards dynamically
    def update_hazards(self, new_hazards):
        """Update the hazards dynamically based on real-time conditions."""
        self.hazards = new_hazards

        
    def calculating_function_frame(self):
        print("Calculating function frame...")
        self.instrument=Instrument()
        self.instrument.start()

        task_subsets=[]
        for i in range(len(self.tasks)+1):
            task_subsets=task_subsets+[Set(list(e)) for e in itertools.combinations(self.tasks,i)]
        task_subsets=Set(task_subsets)

        domain_robots=[e.id for e in self.robots]
        domain_tasks=[set([ee.id for ee in e]) for e in task_subsets]
        self.domain_dict=["value","time"]
        self.domain_matrix=[domain_robots,domain_tasks]
        self["value"]=Matrix(self.domain_matrix,np.zeros((len(domain_robots),len(domain_tasks))))
        self["time"]=Matrix(self.domain_matrix,np.zeros((len(domain_robots),len(domain_tasks))))

        N=len(task_subsets)
        progress_bar=ProgressBar(N)
        for i_allocation,target_allocation in enumerate(task_subsets):
            progress_bar.progress(i_allocation)
            V_ret,Mu,V,t_eval=self.get_all_function_values(target_allocation)
            #save Mu and V ???
            self["value"].matrix[:,i_allocation]=V_ret
            self["time"].matrix[:,i_allocation]=t_eval*np.ones(len(V_ret))
        progress_bar.progress(N,'Finished!\n')
        calculation_time=self.instrument.stop()
        self.instrument.save_measurement("calculation_time",calculation_time)
        print("Calculation time [s]:")
        print(self.instrument.data)
        print("Finished calculating function frame!\n")

    def get_all_function_values(self,target_allocation):
        targets=[]
        for a in target_allocation:
            targets=targets+[a.target]
        goal=self.robots[0].goal
        x_0=[r.x_0 for r in self.robots]
        instrument=Instrument()
        instrument.start()
        V_ret,Mu,V=self.path_planner.get_solution(targets,goal,x_0)
        t_eval=instrument.stop()
        return V_ret,Mu,V,t_eval

    def get_value(self,r_id,S_r_id):
        return self["value"].get([r_id,S_r_id]),self["time"].get([r_id,S_r_id])
    
    # def find_best_alternative(self, robot_id, task_id):
    #     candidates = self.function_frame.get_feasible_tasks(robot_id)
    #     scores = {}

    #     for candidate in candidates:
    #         hazard_score = self.function_frame.estimate_hazard(candidate)
    #         path_score = self.function_frame.estimate_path_safety(robot_id, candidate)
    #         priority = self.function_frame.get_task_priority(candidate)

    #         scores[candidate] = (priority * path_score) / (1 + hazard_score)

    #     return max(scores, key=scores.get)
    
    # def get_feasible_tasks(self, robot_id):
    # # Return list of tasks that are reachable or relevant to this robot
    #     return self.parameters.task_ids

    # def estimate_hazard(self, task_id):
    #     # Estimate hazard around the task’s location
    #     task = self.parameters.tasks[task_id]
    #     return np.mean(self.path_planner.hazard_map[task])

    # def estimate_path_safety(self, robot_id, task_id):
    #     # Return a simple safety score based on predicted path
    #     path = self.path_planner.predict_path(robot_id, task_id)
    #     return 1.0 - np.mean(self.path_planner.hazard_map[path])

    # def get_task_priority(self, task_id):
    #     return self.parameters.task_priority.get(task_id, 1)

