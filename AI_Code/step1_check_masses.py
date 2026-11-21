# step1_check_masses.py  -  checks that changing the mass really changes the robot
import numpy as np
import torch
from common import *

folder = output_folder("step1_check_masses")
cheetah = MassCheetah(seed=0)
masses = [0.5, 1.0, 2.0, 3.5]

rows = []
for scale in masses:
    scores = []
    for seed in range(5):
        torch.manual_seed(seed)                  # same random policy for every mass
        policy = Policy()
        cheetah.set_mass(scale)
        scores.append(cheetah.average_return(policy, scale, episodes=3))
    rows.append([scale, round(cheetah.total_mass(), 2), round(np.mean(scores), 2), round(np.std(scores), 2)])
    print(f"mass x{scale}: total body mass = {cheetah.total_mass():.1f}, "
          f"reward of an untrained policy = {np.mean(scores):.1f}")

save_csv(f"{folder}/mass_check.csv", ["mass_scale", "total_mass", "avg_reward", "std_reward"], rows)

plt.figure(figsize=(5, 3.5))
plt.errorbar([r[0] for r in rows], [r[2] for r in rows], yerr=[r[3] for r in rows], marker="o")
plt.xlabel("Body mass (x normal)")
plt.ylabel("Reward of an untrained policy")
plt.title("Same policy, different mass")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{folder}/mass_check.png", dpi=150)
print("Saved to:", folder)
