import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import numpy as np
from typing import Any, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from src.luxai_s3.wrappers import LuxAIS3GymEnv

def main(env, agent):
    observation, info = env.reset()
    unit_mask = np.array(observation['player_0'].units_mask[0])
    done = False
    actions = {}
    for player in ['player_0', 'player_1']:
        action = np.zeros((16, 3), dtype=int)
        for unit_id in np.where(unit_mask)[0]:
            action[unit_id] = [0,0,0]
        actions[player] = action
    obs, rewards ,terminated, truncated, info = env.step(action=actions)
    print('done')

if __name__ == '__main__':

    env = LuxAIS3GymEnv()
    agent = 1
    main(env, agent)    