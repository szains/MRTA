import numpy as np
import copy

from Set import Set
from Greedy_Allocator import Greedy_Allocator
from Robot import distance

class Reverse_Greedy_Allocator(Greedy_Allocator):
    def __init__(self,function_frame):
        super().__init__(function_frame)
        self.algorithm_name="Reverse_Greedy"
        S_r_0=self.tasks
        self.set_up(S_r_0)

    def solve_problem(self):
        N=len(self.tasks)*(len(self.robots)-1)
        reverse_greedy_solution=super().solve_problem(N)
        return reverse_greedy_solution

    # def make_step(self,V_k_1,R_k_1):
    #     rho_F_vec,f_F_vec,t_F_vec=self.collect_bets(V_k_1,R_k_1)
    #     i_r_k=np.nanargmax(rho_F_vec)

    #     r_k=self.robots[i_r_k]
    #     a_k=r_k.a_r
    #     self.history.add((r_k.id,a_k.id))

    #     r_k.S_r.remove(a_k)
    #     r_k.f_r=r_k.f_r+r_k.rho_r
    #     if len([r for r in self.robots if a_k in r.S_r])==1:
    #         V_k=V_k_1.remove(a_k)
    #         R_k=Set([r for r in self.robots if r.a_r==a_k])
    #     else:
    #         V_k=V_k_1
    #         R_k=Set([r_k])

    #     """
    #     print("\n Step:\n")
    #     print("   - rho_F_vec=",rho_F_vec)
    #     print("   - f_F_vec=",f_F_vec)
    #     print("   - t_F_vec=",t_F_vec)
    #     print("   - (a_k,r_k)=(",str(a_k.id),",",str(r_k.id),")")
    #     """
    #     return V_k,R_k

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
        battery_threshold = 20  
        hazard_threshold = 5  

        if r.battery_level < battery_threshold or r.hazard_proximity < hazard_threshold:
            return True  
        return False  
    
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
        
    def is_safe_assignment(self, r, a):
        """Check if assigning task 'a' to robot 'r' is safe based on battery level and hazards."""
        battery_threshold = 20  # Example threshold for battery reassignment
        hazard_threshold = 5  # Example threshold for hazard proximity

        if r.battery_level < battery_threshold or r.hazard_proximity < hazard_threshold:
            return False  # Not safe if battery is too low or hazard is too close
        return True  # Safe if robot conditions allow task execution




    def make_step(self, V_k_1, R_k_1):
        # Update robot states before bidding on tasks
        for r in self.robots:
            battery_usage = self.estimate_battery_usage(r)
            hazards = self.get_current_hazards()
            r.update_robot_state(hazards, battery_usage)  # Update before allocation
        
        rho_F_vec, f_F_vec, t_F_vec = self.collect_bets(V_k_1, R_k_1)
        i_r_k = np.argmax(rho_F_vec)  # Reverse greedy selects the *maximum* instead of minimum

        r_k = self.robots[i_r_k]
        a_k = r_k.a_r

        # Adaptive reassignment based on robot state
        if self.should_reassign(r_k):
            a_k = self.reassign_task(r_k, V_k_1)  # Modify task selection if needed

        if a_k is None:
            print(f"Warning: No valid task found for robot {r_k.id}, skipping assignment.")
            return V_k_1, R_k_1  # Skip this step if no valid task exists
        self.history.add((r_k.id, a_k.id))
        r_k.S_r.add(a_k)
        r_k.f_r -= r_k.rho_r

        V_k = V_k_1.remove(a_k)
        R_k = Set([r for r in self.robots if r.a_r == a_k])



        """
        print("\n Step:\n")
        print("   - rho_F_vec=",rho_F_vec)
        print("   - f_F_vec=",f_F_vec)
        print("   - t_F_vec=",t_F_vec)
        print("   - (a_k,r_k)=(",str(a_k.id),",",str(r_k.id),")")
        """

        return V_k, R_k

    # def place_bet(self,r,V_k_1):
    #     rho_r=-float('inf')
    #     a_r=None
    #     bet_time=0
    #     for a in copy.copy(V_k_1).intersect(r.S_r):
    #         S_r_a=copy.copy(r.S_r).remove(a)
    #         f_r_a,time_f_r_a=r.objective.get_value(S_r_a)
    #         rho_r_a=f_r_a-r.f_r
    #         if rho_r_a>rho_r:
    #             rho_r=rho_r_a
    #             a_r=a
    #         bet_time=bet_time+time_f_r_a
    #     r.rho_r=rho_r
    #     r.a_r=a_r
    #     return bet_time

    def place_bet(self, r, V_k_1):
        rho_r = float('-inf')  # Reverse greedy chooses the *highest gain*, starting from -inf
        a_r = None
        bet_time = 0
        for a in V_k_1:
            S_r_a = copy.copy(r.S_r).add(a)
            f_r_a, time_f_r_a = r.objective.get_value(S_r_a)
            hazard_penalty = self.calculate_hazard_penalty(r, a)  # Penalize hazardous tasks
            battery_factor = self.get_battery_factor(r)  # Adjust selection based on battery level
            
            rho_r_a = r.f_r - f_r_a + hazard_penalty - battery_factor  # Adjusted cost function
            if rho_r_a > rho_r:  # Reverse greedy selects the highest benefit
                rho_r = rho_r_a
                a_r = a
            bet_time += time_f_r_a

        r.rho_r = rho_r
        r.a_r = a_r
        return bet_time
    
    def reassign_task(self, r, V_k_1):
        alternative_tasks = [a for a in V_k_1 if self.is_safe_assignment(r, a)]
        
        if alternative_tasks:
            return max(alternative_tasks, key=lambda a: self.get_task_priority(r, a))  
        else:
            print(f"Warning: No safe tasks for robot {r.id}, assigning default task.")
            return next(iter(V_k_1), None)  # Assign first available task, or None if empty




    def get_alpha_G(self):
        K=len(self.tasks)*(len(self.robots)-1)
        return super().get_alpha_G(K)

    def get_gamma_G(self):
        K=len(self.tasks)*(len(self.robots)-1)
        return super().get_gamma_G(K)