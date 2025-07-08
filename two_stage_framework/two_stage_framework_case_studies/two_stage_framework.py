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

from X_set import X_set
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

     # 🔐 Add this consistency check before dot product
    if self.X.adj_matrix.shape[0] != ys_1.shape[1]:
        raise ValueError(
            f"Mismatch: adj_matrix is {self.X.adj_matrix.shape}, but ys_1 has {ys_1.shape[1]} states. "
            "Make sure you regenerated X and its adjacency matrix after updating hazards or map."
        )

    print("adj_matrix shape:", self.X.adj_matrix.shape)
    print("ys_1 shape:", ys_1.shape)

    print("DEBUG — len(self.X):", len(self.X))
    print("DEBUG — ys_1.shape[1]:", ys_1.shape[1])



    N_ys=self.X.adj_matrix.T.dot(ys_1.T.astype(int)).T
    D_ys=self.X.adj_diag_matrix.T.dot(ys_1.T.astype(int)).T

    p_cont=1-(((1-p_f)**N_ys)*((1-p_f/np.sqrt(2))**D_ys))
    rand=np.random.rand(*ys_1.shape)
    ys_cont=rand<=p_cont

    ys_k=ys_k_1
    ys_k[ys_0]=ys_cont[ys_0]
    return ys_k

# Parameters




example_name="case_study_13"
parameters=Parameters(name=example_name)
open_case_study=False

rel_path = '/case_studies/' + example_name + '/'
path = os.getcwd() + rel_path

if not os.path.exists(path):
    os.makedirs(path)

parameters_file={"Read":open_case_study,"Name":"parameters"}
samples_file={"Read":open_case_study,"Name":"samples"}
function_frame_file={"Read":open_case_study,"Name":"function_frame"}
solution_file={"Read":open_case_study,"Name":"solution"}


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


# parameters.targets = [(3,9), (5,1), (8,6), (11,11), (14,1), (2,6), (7,9)]
# parameters.targets = [(3,9), (5,1), (8,6), (11,11), (14,1), (2,6), (7,9)]
# parameters.task_ids = ["i", "ii", "iii", "iv", "v", "vi"]

parameters.task_ids = ["i", "ii", "iii", "iv", "v", "vi", "vii"]
parameters.targets = [(3,9), (5,1), (8,6), (11,11), (14,1), (2,6), (7,9)]


parameters.task_priority = {
    "i": 3,
    "ii": 5,
    "iii": 2,
    "iv": 4,
    "v": 1,
    "vi": 2,
    "vii": 4
}


parameters.robot_positions = [(0,6), (8,12), (10,0), (2,11), (6,6)]


parameters.robot_ids = ["1", "2", "3", "4", "5"]
parameters.robot_linestyles = [
    (0, ()),           # solid
    (0, (3,3)),        # dashed
    (0, (1,2)),        # dotted
    (0, (2,2,1,2)),    # dash-dot
    (0, (5,1))         # custom pattern
]

parameters.y_0 = [
    [(12,12)],  # was (13,12)
    [(2,1)],
    [(11,2)],
    [(3,11)],
    [(12,6)],  # also change (13,6) if needed
    [(5,10)],
    [(7,3)]
]

for group in parameters.y_0:
    for (y, x) in group:
        if y >= parameters.map.shape[0] or x >= parameters.map.shape[1]:
            print(f"⚠️ Hazard ({y},{x}) is out of map bounds!")
parameters.hazard_ids = ["a", "b", "c", "d", "e", "f", "g"]
parameters.p_f = [0.002, 0.004, 0.012, 0.012, 0.012, 0.01, 0.008]




parameters.goal=(16,9)
parameters.E=5000
parameters.N=75
parameters.p_stay=0

if not parameters_file["Read"]:
    parameters.generate_obsticles()
    # ✅ Build X using a temporary planner
    parameters.X = X_set(Path_Planner(parameters))

    # 🧹 Filter invalid hazard positions BEFORE generating hazards
    parameters.y_0 = [
        [pos for pos in group if pos in parameters.X]
        for group in parameters.y_0
    ]

    parameters.generate_Hazards()

    parameters.X = None
    parameters.generate_Tasks()

    parameters.X = X_set(Path_Planner(parameters))
    print("DEBUG — parameters.X is", type(parameters.X))
    # ✅ Now that X has been created, it's safe to validate hazards
    if parameters.X is not None:
        invalid_hazards = []
        for group in parameters.y_0:
            for pos in group:
                if pos not in parameters.X:
                    invalid_hazards.append(pos)

    if invalid_hazards:
        print("🚨 Invalid hazard positions not found in X:", invalid_hazards)
    parameters.generate_Robots()

    parameters.generate_Tau_X = generate_Tau_X
    parameters.sample_Tau_Ys = sample_Tau_Ys

    outfile = open(path + parameters_file["Name"], 'wb')
    pickle.dump(parameters, outfile)
    outfile.close()


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

parameters.parameters_file=parameters_file
parameters.samples_file=samples_file
parameters.function_frame_file=function_frame_file
parameters.solution_file=solution_file

# Setting up
path_planner=Path_Planner(parameters)
path_planner.set_up(path)
parameters.X = path_planner.X
print((2,7) in parameters.X)  # This should print False
print((2,6) in parameters.X)
print((2,8) in parameters.X)

print((2,6) in parameters.X)
print((2,8) in parameters.X)
print((7,12) in parameters.X)
print((7,14) in parameters.X)

print((7,11) in parameters.X)
print((7,13) in parameters.X)
print((6,12) in parameters.X)

print((7,10) in parameters.X)
print((7,8) in parameters.X)
print((6,11) in parameters.X)
print((8,11) in parameters.X)

print(f"Total walkable positions: {len(parameters.X)}")
print("First 10 valid positions:", parameters.X[:10])

valid_positions = list(parameters.X)
print("Total:", len(valid_positions))
print(valid_positions)

print("Current task targets:", parameters.targets)

# 🧪 Inspect walkability of robot start positions
for pos in parameters.robot_positions:
    print(f"{pos} walkable:", pos in parameters.X)

# 🚨 Validate robot starting positions
for pos in parameters.robot_positions:
    if pos not in parameters.X:
        print(f"🚨 Robot position {pos} is not in X! Check map and update robot_positions.")

for pos in parameters.targets:
    if pos not in parameters.X:
        print(f"🚨 Task target {pos} is not in X!")


parameters.y_0 = [
    [pos for pos in group if pos in parameters.X]
    for group in parameters.y_0
]


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


