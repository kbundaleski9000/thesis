from pettingzoo.utils import env_registry
from GridWorldAEC import GridWorldAECEnv

def env_creator():
    return GridWorldAECEnv()

env_registry["gridworld_aec_v0"] = env_creator


