# step2_reinforce.py  -  checks our RL code: can REINFORCE teach the cheetah to run at normal mass?
import time
import numpy as np
import torch
from common import *

# ----- settings for this step -----
iterations = 300                 # how many training iterations
runs_per_iteration = 20          # robot runs per iteration
seed = 0
mass = 1.0                       # normal robot
# ----------------------------------

folder = output_folder("step2_reinforce")
torch.manual_seed(seed)
np.random.seed(seed)
cheetah = MassCheetah(seed=seed)
policy = Policy()
optimizer = torch.optim.Adam(policy.parameters(), lr=outer_lr)

rows = []
start = time.time()
for it in range(1, iterations + 1):
    batch = cheetah.collect(policy, mass, runs_per_iteration)     # robot practices
    avg_return = float(batch["rewards"].sum(axis=1).mean())       # how well did it do?
    loss = policy_gradient_loss(policy, batch)                    # turn practice into a loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()                                              # learn
    rows.append([it, it * runs_per_iteration, round(avg_return, 2)])
    if it % 10 == 0:
        print(f"iteration {it:4d} | average return {avg_return:8.1f} | time {time.time() - start:5.0f}s")

# save everything useful
torch.save(policy.state_dict(), weights_path("reinforce_mass1", seed))
save_csv(f"{folder}/learning_curve.csv", ["iteration", "robot_runs_used", "avg_return"], rows)

returns = [r[2] for r in rows]
smooth = np.convolve(returns, np.ones(10) / 10, mode="valid")     # smoothed line is easier to read
plt.figure(figsize=(6, 4))
plt.plot([r[1] for r in rows], returns, alpha=0.3, label="each iteration")
plt.plot([r[1] for r in rows][9:], smooth, label="smoothed (10 iterations)")
plt.xlabel("Robot runs used for training")
plt.ylabel(f"Return (reward over {episode_length} steps)")
plt.title("REINFORCE on HalfCheetah (normal mass)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{folder}/learning_curve.png", dpi=150)
print("Saved to:", folder)
