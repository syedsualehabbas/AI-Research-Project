# step4b_oracle.py  (OPTIONAL, slow)  -  the "oracle": a robot trained ONLY on one test mass.
# It shows the best score we could hope for, so we know how close adaptation gets.
# It is used by step 5 as a dashed line. Skip this step if you are short on time.
import os
import numpy as np
import torch
from common import *

oracle_runs_per_iteration = 20
seed = 0

folder = output_folder("step4b_oracle")
file = f"{folder}/oracle_returns.csv"
done = {}
if os.path.exists(file):                                   # remember masses we already did
    for line in open(file).read().strip().split("\n")[1:]:
        mass, value = line.split(",")
        done[float(mass)] = float(value)

for mass in test_masses_inside + test_masses_outside:
    if mass in done:
        print(f"[skip] mass x{mass} already done")
        continue
    torch.manual_seed(seed)
    np.random.seed(seed)
    cheetah = MassCheetah(seed=seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=outer_lr)
    iterations = train_trajectories // oracle_runs_per_iteration      # same budget as the other methods
    for it in range(1, iterations + 1):
        batch = cheetah.collect(policy, mass, oracle_runs_per_iteration)
        loss = policy_gradient_loss(policy, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 50 == 0:
            print(f"mass x{mass} | iteration {it}/{iterations} | return {batch['rewards'].sum(axis=1).mean():.1f}")
    done[mass] = cheetah.average_return(policy, mass, episodes=10)
    print(f"mass x{mass}: oracle return = {done[mass]:.1f}")
    save_csv(file, ["mass_scale", "oracle_return"], [[m, round(v, 2)] for m, v in done.items()])

print("Saved to:", folder)
