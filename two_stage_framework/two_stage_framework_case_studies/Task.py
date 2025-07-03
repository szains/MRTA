class Task(object):
    def __init__(self,id,target, urgency=1.0):
        self.id=id
        self.target=target
        self.urgency = urgency  # New attribute for prioritization
        self.time_waited = 0  # Initialize when task is created
        self.is_assigned = False  # Add this in Task.__init__()

