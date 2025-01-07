import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import numpy as np
from typing import Any, Tuple

from agent import Agent

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from src.luxai_s3.env import LuxAIS3Env
from src.luxai_s3.wrappers import LuxAIS3GymEnv
from src.luxai_runner.utils import to_json


def main(env, agent):
    obs, info = env.reset()
    done = False
    actions = {}
    value, logits = agent.act(step=0, obs=obs["player_0"])


if __name__ == "__main__":
    env = LuxAIS3GymEnv(numpy_output=True)
    agent = Agent(player="player_0")
    main(env, agent)
