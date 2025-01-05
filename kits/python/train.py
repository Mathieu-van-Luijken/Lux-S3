import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import numpy as np
from typing import Any, Tuple

from ppo_agent import PPOAgent

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from src.luxai_s3.env import LuxAIS3Env
from src.luxai_s3.wrappers import LuxAIS3GymEnv
from src.luxai_runner.utils import to_json


def main(env, agent):
    obs, info = env.reset()
    agent.create_board_tensor(obs=obs)
    done = False
    actions = {}
    while not done:
        for player in ["player_0", "player_1"]:
            action = np.zeros((16, 3), dtype=int)
            for unit_id in np.where(unit_mask)[0]:
                action[unit_id] = [np.random.choice(4), 0, 0]
            actions[player] = action
        obs, rewards, terminated, truncated, info = env.step(action=actions)
        if terminated:
            done = False
    print("done")


if __name__ == "__main__":

    env = LuxAIS3GymEnv(numpy_output=True)
    agent = PPOAgent(player="player_0")
    main(env, agent)
