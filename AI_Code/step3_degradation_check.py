# step3_degradation_check.py  -  does a policy trained at normal mass get worse at other masses?
# If yes, the robot really NEEDS to adapt, which is the reason meta-learning matters.
import numpy as np
import torch
from common import *

seed = 0
masses = [0.5, 0.7, 1.0, 1.3, 1.8, 2.5, 3.5]
episodes_per_mass = 10

folder = output_folder("step3_degradation")
cheetah = MassCheetah(seed=seed)
policy = Policy()
policy.load_state_dict(torch.load(weights_path("reinforce_mass1", seed)))   # made by step 2

rows = []
for scale in masses:
    batch = cheetah.collect(policy, scale, episodes_per_mass, explore=False)
    totals = batch["rewards"].sum(axis=1)
    rows.append([scale, round(float(totals.mean()), 2), round(float(totals.std()), 2)])
    print(f"mass x{scale}: return = {totals.mean():.1f} (+/- {totals.std():.1f})")

save_csv(f"{folder}/degradation.csv", ["mass_scale", "avg_return", "std_return"], rows)

plt.figure(figsize=(6, 4))
plt.errorbar([r[0] for r in rows], [r[1] for r in rows], yerr=[r[2] for r in rows], marker="o")
plt.axvline(1.0, color="gray", linestyle="--", label="mass it was trained on")
plt.xlabel("Body mass (x normal)")
plt.ylabel("Return")
plt.title("Policy trained at normal mass, tested at other masses")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{folder}/degradation.png", dpi=150)
print("Saved to:", folder)
