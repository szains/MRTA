from Parameters import Parameters
from Path_Planner import Path_Planner
from Function_Frame import Function_Frame

def run_hybrid_allocation(parameters, function_frame):
    from HybridController import HybridController
    from SCOBA_Allocator import SCOBA_Allocator

    allocator = HybridController(parameters, function_frame)
    solution = allocator.solve()
    allocator.greedy_allocator.show_solution(solution)


if __name__ == "__main__":
    parameters = Parameters(name="case_study_1")
    planner = Path_Planner(parameters)
    planner.set_up(path="/your/path/here")  # Adjust as needed
    function_frame = Function_Frame(parameters, planner)

    run_hybrid_allocation(parameters, function_frame)
