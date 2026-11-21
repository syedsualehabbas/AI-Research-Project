# common.py  -  shared code used by every step. You do NOT run this file directly.
import os
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mujoco
import matplotlib
matplotlib.use("Agg")            # lets us save graphs to files without opening windows
import matplotlib.pyplot as plt

# ======================= SETTINGS (shared by all steps) =======================
episode_length = 200                  # steps per robot run (default is 1000, we use less to be fast)
train_mass_range = (0.5, 2.0)         # during training, body mass is multiplied by a number in this range
test_masses_inside = [0.7, 1.3, 1.8]  # new masses INSIDE the training range (never seen exactly)
test_masses_outside = [2.5, 3.5]      # masses OUTSIDE the training range (out-of-distribution)
trajectories_per_step = 10            # K: robot runs used for ONE adaptation step
inner_lr = 0.1                        # step size of one adaptation step (0.05 to 0.5 all work, 2.0 breaks)
test_adapt_steps = 3                  # we test adaptation with 0, 1, 2, 3 steps
discount = 0.99                       # how much future reward counts
outer_lr = 0.003                      # step size (Adam) used when training the starting weights
tasks_per_iteration = 5               # how many different masses per training iteration
train_trajectories = 15000            # training budget (robot runs) given to EVERY method
seeds = [0]                           # use [0, 1, 2, 3, 4] for the final results
# ==============================================================================

here = os.path.dirname(os.path.abspath(__file__))


def output_folder(name):
    # makes results/<name> next to the code and returns its path
    path = os.path.join(here, "results", name)
    os.makedirs(path, exist_ok=True)
    return path


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def sample_mass():
    return np.random.uniform(*train_mass_range)


# ---------- the policy: a small neural network that controls the robot ----------
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(17, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 6),
        )
        self.log_std = nn.Parameter(torch.full((6,), -0.5))   # how random the actions are

    def distribution(self, obs):
        return torch.distributions.Normal(self.body(obs), self.log_std.exp())


# ---------- the robot: HalfCheetah whose body mass we can change ----------
class MassCheetah:
    def __init__(self, seed=0):
        self.env = gym.make("HalfCheetah-v5")
        self.env.reset(seed=seed)
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        self.base_mass = self.model.body_mass.copy()       # remember the normal values
        self.base_inertia = self.model.body_inertia.copy()

    def set_mass(self, scale):
        # scale = 1.0 is the normal robot, 2.0 is twice as heavy
        self.model.body_mass[:] = self.base_mass * scale
        self.model.body_inertia[:] = self.base_inertia * scale
        mujoco.mj_setConst(self.model, self.data)           # tell the physics engine about the change

    def total_mass(self):
        return float(self.model.body_mass.sum())

    def collect(self, policy, scale, how_many, explore=True):
        # run the robot 'how_many' times and record everything that happened
        self.set_mass(scale)
        all_obs, all_actions, all_rewards = [], [], []
        for _ in range(how_many):
            obs, _ = self.env.reset()
            run_obs, run_actions, run_rewards = [], [], []
            for _ in range(episode_length):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    dist = policy.distribution(obs_tensor)
                    action = dist.sample() if explore else dist.mean
                action = action.numpy()
                next_obs, reward, _, _, _ = self.env.step(np.clip(action, -1, 1))
                run_obs.append(obs)
                run_actions.append(action)
                run_rewards.append(reward)
                obs = next_obs
            all_obs.append(run_obs)
            all_actions.append(run_actions)
            all_rewards.append(run_rewards)
        return {"obs": np.array(all_obs, dtype=np.float32),
                "actions": np.array(all_actions, dtype=np.float32),
                "rewards": np.array(all_rewards, dtype=np.float32)}

    def average_return(self, policy, scale, episodes=5):
        # total reward of the policy (no random exploration), averaged over a few runs
        batch = self.collect(policy, scale, episodes, explore=False)
        return float(batch["rewards"].sum(axis=1).mean())


def combine(batches):
    # glue several batches together into one big batch
    return {key: np.concatenate([b[key] for b in batches], axis=0) for key in batches[0]}


# ---------- policy gradient (REINFORCE) ----------
def policy_gradient_loss(policy, batch):
    rewards = batch["rewards"]                     # shape: (runs, steps)
    runs, steps = rewards.shape
    # reward-to-go: how much reward the robot collected from each step onward
    reward_to_go = np.zeros_like(rewards)
    running = np.zeros(runs, dtype=np.float32)
    for t in reversed(range(steps)):
        running = rewards[:, t] + discount * running
        reward_to_go[:, t] = running
    # advantage: was this better or worse than average at the same time step?
    advantage = reward_to_go - reward_to_go.mean(axis=0)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    obs = torch.as_tensor(batch["obs"])
    actions = torch.as_tensor(batch["actions"])
    log_prob = policy.distribution(obs).log_prob(actions).sum(dim=-1)
    return -(log_prob * torch.as_tensor(advantage)).mean()


def sgd_step(policy, batch, lr):
    # one plain gradient step on the policy (used for adaptation)
    loss = policy_gradient_loss(policy, batch)
    grads = torch.autograd.grad(loss, list(policy.parameters()))
    with torch.no_grad():
        for p, g in zip(policy.parameters(), grads):
            p -= lr * g


def adapt_and_measure(cheetah, start_policy, scale, steps=test_adapt_steps):
    # adapt step by step to a new mass; return the robot's score after 0, 1, 2, ... steps
    policy = copy.deepcopy(start_policy)
    scores = [cheetah.average_return(policy, scale)]
    for _ in range(steps):
        batch = cheetah.collect(policy, scale, trajectories_per_step)   # robot practices
        sgd_step(policy, batch, inner_lr)                               # robot learns a bit
        scores.append(cheetah.average_return(policy, scale))
    return scores


def weights_path(method, seed):
    folder = output_folder("weights")
    return os.path.join(folder, f"{method}_seed{seed}.pt")
