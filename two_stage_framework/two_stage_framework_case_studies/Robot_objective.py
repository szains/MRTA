class Robot_objective(object):
    def __init__(self,r,function_frame):
        self.function_frame=function_frame
        self.r=r

    # def get_value(self,S_r,print_progress=False):
    #     r_id=self.r.id
    #     S_r_id=set([a.id for a in S_r])
    #     return self.function_frame.get_value(r_id,S_r_id)

    # def get_value(self, S_r, print_progress=False):
    #     r_id = self.r.id
    #     S_r_id = set([a.id for a in S_r])

    #     # ⚡ Apply task priorities
    #     priorities = self.function_frame.parameters.task_priority  # dict like {'i': 3, 'ii': 5, ...}
    #     priority_weights = sum(priorities.get(t_id, 1) for t_id in S_r_id)

    #     base_value = self.function_frame.get_value(r_id, S_r_id)

    #     # Scale the base value by the cumulative priority (optional tweak)
    #     adjusted_value = base_value * priority_weights

    #     return adjusted_value

    # def get_value(self, S_r, print_progress=False):
    #     r_id = self.r.id
    #     S_r_id = set(a.id for a in S_r)

    #     base_value, _ = self.function_frame.get_value(r_id, S_r_id)

    #     # Priority weighting
    #     priorities = self.function_frame.parameters.task_priority
    #     priority_sum = sum(priorities.get(t_id, 1) for t_id in S_r_id)

    #     adjusted_value = base_value * priority_sum
    #     return adjusted_value
    
    def get_value(self, S_r, print_progress=False):
        r_id = self.r.id
        S_r_id = set(a.id for a in S_r)

        base_value, base_time = self.function_frame.get_value(r_id, S_r_id)

        priorities = self.function_frame.parameters.task_priority
        priority_sum = sum(priorities.get(t_id, 1) for t_id in S_r_id)

        adjusted_value = base_value * priority_sum
        return adjusted_value, base_time



