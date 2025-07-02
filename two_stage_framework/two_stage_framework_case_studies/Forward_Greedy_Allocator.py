import numpy as np
import copy

from Set import Set
from Greedy_Allocator import Greedy_Allocator
from Robot import distance


class Forward_Greedy_Allocator(Greedy_Allocator):
    def __init__(self,function_frame):
        super().__init__(function_frame)
        self.algorithm_name="Forward_Greedy"
        S_r_0 = Set(self.tasks)
        self.set_up(S_r_0)

    def solve_problem(self):
        N = len(self.robots[0].S_r)  # Safely gets task count from robot’s set
        if not N:
            raise ValueError("No tasks available: make sure allocator is properly initialized.")
        forward_greedy_solution = super().solve_problem(N)
        return forward_greedy_solution
    
    def get_task_priority(self, r, a):
        """Return pre-assigned task priority or fallback to zero."""
        return self.function_frame.parameters.task_priority.get(a.id, 0)


    # def make_step(self,V_k_1,R_k_1):
    #     rho_F_vec,f_F_vec,t_F_vec=self.collect_bets(V_k_1,R_k_1)
    #     i_r_k=np.argmin(rho_F_vec)

    #     r_k=self.robots[i_r_k]
    #     a_k=r_k.a_r
    #     self.history.add((r_k.id,a_k.id))

    #     r_k.S_r.add(a_k)
    #     r_k.f_r=r_k.f_r-r_k.rho_r
    #     V_k=V_k_1.remove(a_k)
    #     R_k=Set([r for r in self.robots if r.a_r==a_k])

        """
        print("\n Step:\n")
        print("   - rho_F_vec=",rho_F_vec)
        print("   - f_F_vec=",f_F_vec)
        print("   - t_F_vec=",t_F_vec)
        print("   - (a_k,r_k)=(",str(a_k.id),",",str(r_k.id),")")
        """
        # return V_k,R_k
    
    # new code by zain shah

    def estimate_battery_usage(self, r):
        """Estimate battery consumption per task based on distance and complexity."""
        base_usage = 1.0  # Base reduction per task
        distance_factor = distance(r.x_0, r.goal) * 0.01  # Scales with distance
        return base_usage + distance_factor
    
    def get_current_hazards(self):
        """Retrieve hazard locations affecting robot task allocation."""
        if hasattr(self.function_frame, "hazards"):  # Ensure hazards exist
            return self.function_frame.hazards  
        else:
            return []  # Return empty list if hazards aren’t initialized



    def make_step(self, V_k_1, R_k_1):
        for r in self.robots:
            battery_usage = self.estimate_battery_usage(r)  # Define battery usage logic
            hazards = self.get_current_hazards()  # Retrieve hazard data
            r.update_robot_state(hazards, battery_usage)  # Update robot state before bidding
        rho_F_vec, f_F_vec, t_F_vec = self.collect_bets(V_k_1, R_k_1)
        i_r_k = np.argmin(rho_F_vec)

        r_k = self.robots[i_r_k]
        a_k = r_k.a_r

        # Introduce adaptive reassignment based on robot state
        if self.should_reassign(r_k):
            a_k = self.reassign_task(r_k, V_k_1)  # Modify task selection if needed

        if a_k is None:
            print(f"[Step] Fallback: Assigning first available task to robot {r_k.id}")
            a_k = next(iter(V_k_1), None)
            if a_k is None:
                print("No tasks left to assign.")
                return V_k_1, R_k_1

        self.history.add((r_k.id, a_k.id))
        r_k.S_r.add(a_k)
        r_k.f_r -= r_k.rho_r

        V_k = copy.copy(V_k_1)
        V_k.remove(a_k)
        R_k = Set([r for r in self.robots if r.a_r == a_k])

        return V_k, R_k
    
    def calculate_hazard_penalty(self, r, a):
        """Calculate penalty based on the robot's proximity to hazards."""
        hazard_penalty = 0
        if hasattr(self.function_frame, "hazards"):  # Ensure hazards exist
            for hazard in self.function_frame.hazards:
                distance_to_hazard = distance(r.x_0, hazard)
                hazard_penalty += np.exp(-distance_to_hazard)  # Penalize tasks near hazards
        return hazard_penalty
    
    def get_battery_factor(self, r):
        """Adjust task allocation based on battery level."""
        battery_threshold = 20  # Example threshold (can be adjusted)
        
        if r.battery_level > battery_threshold:
            return 0  # No penalty for tasks if battery is sufficient
        else:
            return (battery_threshold - r.battery_level) * 0.1  # Increasing penalty as battery decreases
        
    def should_reassign(self, r):
        """Determine whether a robot should reassign its task based on battery level or hazard proximity."""
        battery_threshold = 20  # Define a threshold for battery reassignment
        hazard_threshold = 5  # Define a proximity threshold for hazards

        if len(self.tasks) <= 2:
            return True  # Let robots accept any task in small task cases
        return not (r.battery_level < battery_threshold or r.hazard_proximity < hazard_threshold)


    def is_safe_assignment(self, r, a):
        """Check if assigning task 'a' to robot 'r' is safe based on battery level and hazards."""
        battery_threshold = 20  # Example threshold for battery reassignment
        hazard_threshold = 5  # Example threshold for hazard proximity

        if r.battery_level < battery_threshold or r.hazard_proximity < hazard_threshold:
            return False  # Not safe if battery is too low or hazard is too close
        return True  # Safe if robot conditions allow task execution


    
    def reassign_task(self, r, V_k_1):
        alternative_tasks = [a for a in V_k_1 if self.is_safe_assignment(r, a)]
        
        if alternative_tasks:
            return max(alternative_tasks, key=lambda a: self.get_task_priority(r, a))  
        else:
            print(f"Warning: No safe tasks for robot {r.id}, assigning default task.")
            return next(iter(V_k_1), None)  # Assign first available task, or None if empty




    # def place_bet(self,r,V_k_1):
    #     rho_r=float('inf')
    #     a_r=None
    #     bet_time=0
    #     for a in V_k_1:
    #         S_r_a=copy.copy(r.S_r).add(a)
    #         f_r_a,time_f_r_a=r.objective.get_value(S_r_a)
    #         rho_r_a=r.f_r-f_r_a
    #         if rho_r_a<rho_r:
    #             rho_r=rho_r_a
    #             a_r=a
    #         bet_time=bet_time+time_f_r_a
    #     r.rho_r=rho_r
    #     r.a_r=a_r
    #     return bet_time

    # new code by zain shah
    def place_bet(self, r, V_k_1):
        rho_r = float('inf')
        a_r = None
        bet_time = 0
        for a in V_k_1:
            S_r_a = copy.copy(r.S_r).add(a)
            f_r_a, time_f_r_a = r.objective.get_value(S_r_a)
            hazard_penalty = self.calculate_hazard_penalty(r, a)  # Introduce hazard-aware penalty
            battery_factor = self.get_battery_factor(r)  # Weight selection based on battery level
            
            rho_r_a = r.f_r - f_r_a + hazard_penalty - battery_factor  # Adjust cost function
            if rho_r_a < rho_r:
                rho_r = rho_r_a
                a_r = a
            bet_time += time_f_r_a

        r.rho_r = rho_r
        r.a_r = a_r
        return bet_time
    
    def estimate_battery_usage(self, r):
        """Estimate battery consumption per task based on distance and complexity."""
        base_usage = 1.0  # Base reduction per task
        distance_factor = distance(r.x_0, r.goal) * 0.01  # Scales with distance
        return base_usage + distance_factor

    def get_current_hazards(self):
        """Retrieve hazard locations affecting robot task allocation."""
        return self.function_frame.hazards  # Ensure hazards exist in `function_frame`



    def get_alpha_G(self):
        K=len(self.tasks)
        return super().get_alpha_G(K)

    def get_gamma_G(self):
        K=len(self.tasks)
        return super().get_gamma_G(K)