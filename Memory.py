from Environment import animate_rollout

class ReplayBuffer:
    def __init__(environment, nSteps):
        self.environment = environment
        self.nSteps = nSteps
        self.buffer = None
    

