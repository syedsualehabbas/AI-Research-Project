# record_video.py  -  saves videos of the cheetah, before and after adapting.
# Run it AFTER steps 2 and 4 (it needs the saved weights). Videos go to results/videos/
import os
import copy
import numpy as np
import torch
import gymnasium as gym
import mujoco
import imageio.v2 as imageio
from common import *

# ----- settings -----
seed = 0
video_mass = 3.5            # which mass to film (pick one where adaptation mattered in step 5)
adapt_steps = 3             # how many adaptation steps before the "after" video
# --------------------

folder = output_folder("videos")


def film(policy, mass, name):
    # run the policy once with a camera on, save the video, return its score
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    model, data = env.unwrapped.model, env.unwrapped.data
    model.body_mass[:] = model.body_mass * mass               # make the robot heavier or lighter
    model.body_inertia[:] = model.body_inertia * mass
    mujoco.mj_setConst(model, data)
    obs, _ = env.reset(seed=seed)
    frames, total = [], 0.0
    for _ in range(episode_length):
        frames.append(env.render())
        with torch.no_grad():
            action = policy.distribution(torch.as_tensor(obs, dtype=torch.float32)).mean.numpy()
        obs, reward, _, _, _ = env.step(np.clip(action, -1, 1))
        total += reward
        distance = env.unwrapped.data.qpos[0]      # how far the cheetah ended up (x position)
    print(f"   distance travelled = {distance:.1f} m")
    env.close()
    try:
        path = f"{folder}/{name}.mp4"
        imageio.mimsave(path, frames, fps=30)
    except Exception:                                          # no mp4 support? save a gif instead
        path = f"{folder}/{name}.gif"
        imageio.mimsave(path, frames, fps=30)
    print(f"saved {os.path.basename(path)} | return = {total:.1f}")
    return total


def adapt(policy, mass):
    # same adaptation as in the experiment: a few practice runs, a few gradient steps
    cheetah = MassCheetah(seed=seed + 50)
    adapted = copy.deepcopy(policy)
    for _ in range(adapt_steps):
        batch = cheetah.collect(adapted, mass, trajectories_per_step)
        sgd_step(adapted, batch, inner_lr)
    return adapted


def load(method):
    path = weights_path(method, seed)
    if not os.path.exists(path):
        print(f"[skip] {method}: no weights found (run its step first)")
        return None
    policy = Policy()
    policy.load_state_dict(torch.load(path))
    return policy


torch.manual_seed(seed)
np.random.seed(seed)
tag = f"mass{video_mass}"

# 1) the policy trained only at normal mass (step 2), shown on the new mass
single = load("reinforce_mass1")
if single:
    film(single, video_mass, f"1_trained_at_mass1_{tag}")

# 2) each method: before adapting (step 0) and after adapting
for number, method in [(2, "mixed"), (3, "reptile"), (4, "fomaml")]:
    policy = load(method)
    if policy:
        film(policy, video_mass, f"{number}a_{method}_before_{tag}")
        film(adapt(policy, video_mass), video_mass, f"{number}b_{method}_after_{adapt_steps}steps_{tag}")

# 3) a cheetah with no training at all, for comparison
torch.manual_seed(seed)
film(Policy(), video_mass, f"0_untrained_{tag}")

print("\nAll videos saved in:", folder)