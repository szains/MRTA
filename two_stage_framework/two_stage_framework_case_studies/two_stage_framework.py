import numpy as np
import warnings
import pickle
import os

from Parameters import Parameters
from Drawer import Drawer
from Matrix import Matrix
from Path_Planner import Path_Planner
from Function_Frame import Function_Frame
from Forward_Greedy_Allocator import Forward_Greedy_Allocator
from Reverse_Greedy_Allocator import Reverse_Greedy_Allocator
from Brute_Force_Allocator import Brute_Force_Allocator
from Graphs import plot_task_completion, plot_robot_utilization, plot_hazard_avoidance
from Graphs import plot_hazard_avoidance
from Graphs import plot_task_priority_heatmap, plot_priority_completion
import random
from Set import Set  # make sure it's imported at the top


from collections import Counter

def compute_robot_utilization(allocation, robot_ids):
    task_counts = Counter([robot for robot, _ in allocation])
    total_tasks = sum(task_counts.values())

    utilization = {}
    for robot in robot_ids:
        count = task_counts.get(robot, 0)
        utilization[robot] = round((count / total_tasks) * 100, 2) if total_tasks > 0 else 0.0
    return utilization

# Parameter Functions
def generate_Tau_X(self):
    p_stay=self.p_stay

    for u in self.U_x.U:
        matrix_u=Matrix(self.domain_matrix,np.zeros(tuple([len(e) for e in self.domain_matrix])))
        if u=='0':
            for x in self.X:
                matrix_u.set([x,x],1)
        else:
            for x in self.X:
                if self.U_x.is_u_in_U_x_(u,x):
                    xx=self.U_x.get_xx_u(x,u)
                    matrix_u.set([x,x],p_stay)
                    matrix_u.set([x,xx],1-p_stay)
        self[u]=matrix_u

def get_robot_workload_percentages(solution):
    # Counts how many tasks each robot got
    from collections import Counter
    robot_task_counts = Counter([r_id for (r_id, _) in solution.allocation.pairs])
    total_tasks = len(set([t_id for (_, t_id) in solution.allocation.pairs]))

    workload = {}
    for robot_id in parameters.robot_ids:
        count = robot_task_counts.get(robot_id, 0)
        workload[robot_id] = round((count / total_tasks) * 100, 2) if total_tasks > 0 else 0
    return workload

def sample_Tau_Ys(self,p_f,ys_k_1):
    ys_0=~ys_k_1
    ys_1=ys_k_1

    N_ys=self.X.adj_matrix.T.dot(ys_1.T.astype(int)).T
    D_ys=self.X.adj_diag_matrix.T.dot(ys_1.T.astype(int)).T

    p_cont=1-(((1-p_f)**N_ys)*((1-p_f/np.sqrt(2))**D_ys))
    rand=np.random.rand(*ys_1.shape)
    ys_cont=rand<=p_cont

    ys_k=ys_k_1
    ys_k[ys_0]=ys_cont[ys_0]
    return ys_k

# Parameters
example_name = "case_study_1"
parameters = Parameters(name=example_name)
open_case_study = False
rel_path = f"/case_studies/{example_name}/"
path = os.getcwd() + rel_path

if not os.path.exists(path):
    os.makedirs(path)



parameters.map=np.array([[1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1],
                         [1,0,0,0,0,0,1,1,0,1,0,0,1,0,0,0,1],
                         [1,0,1,0,1,0,1,1,0,1,0,0,0,0,1,0,1],
                         [1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,0,1],
                         [1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,1,1],
                         [1,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,1],
                         [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1],
                         [1,1,1,0,1,1,1,1,0,0,1,0,0,0,0,0,1],
                         [1,1,1,0,1,1,1,1,0,0,1,0,0,0,0,0,1],
                         [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                         [1,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,1],
                         [1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,1],
                         [1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1]])

# Add this line to fix the error:
parameters.size = (parameters.map.shape[1], parameters.map.shape[0])

n_tasks = 10
n_robots = 5
near_hazard_tasks = 4

# Hazard setup
parameters.y_0 = [[(13,12)], [(2,1)], [(11,2)], [(3,11)], [(13,6)]]
parameters.hazard_ids = ["a", "b", "c", "d", "e"]
parameters.p_f = [0.002, 0.004, 0.012, 0.012, 0.012]

# Robot setup — defer positions for now, see below

# Task setup
parameters.task_ids = [f"t{i+1}" for i in range(n_tasks)]
parameters.targets = []

# Place some tasks near hazard zones
for hz in parameters.y_0[:near_hazard_tasks]:
    x, y = hz[0]
    tx = x + random.choice([-1, 0, 1])
    ty = y + random.choice([-1, 0, 1])
    parameters.targets.append((tx, ty))

# Place remaining tasks safely
while len(parameters.targets) < n_tasks:
    x = random.randint(0, parameters.map.shape[1] - 1)
    y = random.randint(0, parameters.map.shape[0] - 1)
    if parameters.map[y][x] == 0:
        parameters.targets.append((x, y))

# Assign descending priorities
parameters.task_priority = {t_id: n_tasks - i for i, t_id in enumerate(parameters.task_ids)}

# Other environment settings
parameters.goal = (16, 9)
parameters.E = 5000
parameters.N = 75
parameters.p_stay = 0

# Generate obstacle and hazard maps
parameters.generate_obsticles()
parameters.generate_Hazards()

# Bind dynamic function hooks
parameters.generate_Tau_X = generate_Tau_X
parameters.sample_Tau_Ys = sample_Tau_Ys

# File management setup
parameters_file = {"Read": open_case_study, "Name": "parameters"}
samples_file = {"Read": open_case_study, "Name": "samples"}
function_frame_file = {"Read": open_case_study, "Name": "function_frame"}
solution_file = {"Read": open_case_study, "Name": "solution"}

parameters.parameters_file = parameters_file
parameters.samples_file = samples_file
parameters.function_frame_file = function_frame_file
parameters.solution_file = solution_file

# Planner setup
path_planner = Path_Planner(parameters)
path_planner.set_up(path)

# Validate robot positions AFTER path_planner is ready
valid_X = list(path_planner.X)
random.shuffle(valid_X)

if len(valid_X) < n_robots:
    raise ValueError("Not enough valid positions to place all robots.")

robot_ids = [f"r{i+1}" for i in range(n_robots)]
robot_positions = valid_X[:n_robots]
base_styles = [(0,()), (0,(3,3)), (0,(1,2)), (0,(2,4)), (0,(1,1)), (0,(5,2,2,2))]

# Assign robots and tasks
parameters.generate_Tasks()
# Extract safe robot start positions directly from V[0]'s domain
Mu, V = path_planner.solve_problem(print_progress=True)

# ✅ Extract robot start positions from V[0]'s domain with empty task set

robot_start_candidates = [x for (s, x) in V[0].domain if s == Set([])]
random.shuffle(robot_start_candidates)

if len(robot_start_candidates) < n_robots:
    raise ValueError("Not enough valid start positions in V[0].domain")

parameters.robot_ids = [f"r{i+1}" for i in range(n_robots)]
parameters.robot_positions = robot_start_candidates[:n_robots]

# Apply line styles (optional)
base_styles = [(0,()), (0,(3,3)), (0,(1,2)), (0,(2,4)), (0,(1,1)), (0,(5,2,2,2))]
parameters.robot_linestyles = base_styles[:n_robots]

# Generate robot objects using now-validated positions
parameters.generate_Robots()

# Then generate robots based on those valid starts
parameters.generate_Robots()


### Main ###
warnings.filterwarnings("ignore")

rel_path='/case_studies/'+example_name+'/'
path=os.getcwd()+rel_path
if not os.path.exists(path):
    os.makedirs(path)

# Parameters
if parameters_file["Read"]:
    infile=open(path+parameters_file["Name"],'rb')
    parameters=pickle.load(infile)
    infile.close()
else:
    outfile=open(path+parameters_file["Name"],'wb')
    pickle.dump(parameters,outfile)
    outfile.close()


# Setting up
path_planner=Path_Planner(parameters)
path_planner.set_up(path)

Drawer(path_planner).draw_full_example()

# Function frame
if parameters.function_frame_file["Read"]:
    print("...Reading function frame...")
    infile=open(path+parameters.function_frame_file["Name"],'rb')
    function_frame=pickle.load(infile)
    infile.close()
else:
    function_frame=Function_Frame(parameters,path_planner)
    print("...Saving function frame...")
    outfile=open(path+parameters.function_frame_file["Name"],'wb')
    pickle.dump(function_frame,outfile)
    outfile.close()

# Forward greedy
allocator_fg=Forward_Greedy_Allocator(function_frame)
if parameters.solution_file["Read"]:
    infile=open(path+parameters.solution_file["Name"]+"_fg",'rb')
    fg_solution=pickle.load(infile)
    infile.close()
else:
    fg_solution=allocator_fg.solve_problem()
    allocator_fg.postprocess_solution(fg_solution)
    fg_solution.save_solution(path+parameters.solution_file["Name"]+"_fg")
allocator_fg.show_solution(fg_solution)

for e in fg_solution.allocation:
    print("FG allocation element:", e)
    break  # Just to peek at one element


# Reverse greedy
allocator_rg=Reverse_Greedy_Allocator(function_frame)
if parameters.solution_file["Read"]:
    infile=open(path+parameters.solution_file["Name"]+"_rg",'rb')
    rg_solution=pickle.load(infile)
    infile.close()
else:
    rg_solution=allocator_rg.solve_problem()
    allocator_rg.postprocess_solution(rg_solution)
    rg_solution.save_solution(path+parameters.solution_file["Name"]+"_rg")
allocator_rg.show_solution(rg_solution)

# Brute force
allocator_bf=Brute_Force_Allocator(function_frame)
if parameters.solution_file["Read"]:
    infile=open(path+parameters.solution_file["Name"]+"_bf",'rb')
    bf_solution=pickle.load(infile)
    infile.close()
    infile=open(path+parameters.solution_file["Name"]+"_worst",'rb')
    worst_solution=pickle.load(infile)
    infile.close()
else:
    bf_solution,worst_solution=allocator_bf.solve_problem()
    allocator_bf.postprocess_solution(bf_solution)
    allocator_bf.postprocess_solution(worst_solution)
    bf_solution.save_solution(path+parameters.solution_file["Name"]+"_bf")
    worst_solution.save_solution(path+parameters.solution_file["Name"]+"_worst")

allocator_bf.show_solution(bf_solution)
allocator_bf.show_solution(worst_solution)

task_completion_data = [
    len(set(task for _, task in fg_solution.allocation)),
    len(set(task for _, task in rg_solution.allocation)),
    len(set(task for _, task in bf_solution.allocation))
]

fg_utilization = compute_robot_utilization(fg_solution.allocation, parameters.robot_ids)

plot_task_completion(task_completion_data)
plot_robot_utilization(fg_utilization)
plot_hazard_avoidance(path_planner.hazard_avoidance_data)

plot_priority_completion(fg_solution.allocation, parameters.task_priority, parameters.robot_ids)
plot_task_priority_heatmap(parameters.tasks, parameters.task_priority)

