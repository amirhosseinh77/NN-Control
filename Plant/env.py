import gymnasium
from gymnasium import spaces
import numpy as np


class PlantEnv(gymnasium.Env):
    def __init__(self):
        super(PlantEnv, self).__init__()
        self.dynamics = \\
        self.action_space = spaces.Box(low=0, high=np.ones(), shape=(,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(,), dtype=np.float32) # HEIGHT, WIDTH, N_CHANNELS

    def step(self, action):
        return next_state.astype(np.float32), reward, done, False, info

    def reset(self, seed=None, options={}):
        return state.ravel().astype(np.float32), info

    def render(self):
        pass



