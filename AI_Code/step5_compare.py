# step5_compare.py  -  the main experiment.
# Every method starts from its own starting weights, gets the SAME adaptation budget
# (trajectories_per_step robot runs per step, 0 to test_adapt_steps steps) on NEW masses,
# and is scored. Everything is saved as tables and graphs.
import os
import numpy as np
import torch
from common import *

folder = output_folder("step5_compare")
all_masses = test_masses_inside + test_masses_outside


def starting_policy(method, seed):
    # returns the starting weights of each method
    policy = Policy()
    if method == "scratch":
        torch.manual_seed(seed)           # brand-new random network, no training
        return Policy()
    policy.load_state_dict(torch.load(weights_path(method, seed)))
    return policy


# names shown in tables and graphs
names = {
    "scratch": "Scratch (random start)",
    "mixed": "Mixed-mass training (step 0 = domain randomization)",
    "reptile": "Reptile",
    "fomaml": "First-order MAML",
}

# ---------- run all tests and save raw results ----------
rows = []
for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)
    cheetah = MassCheetah(seed=1000 + seed)
    for method in names:
        if method != "scratch" and not os.path.exists(weights_path(method, seed)):
            print(f"[missing] {method} seed {seed}: run step 4 first")
            continue
        start = starting_policy(method, seed)
        for mass in all_masses:
            scores = adapt_and_measure(cheetah, start, mass)
            for step, score in enumerate(scores):
                rows.append([seed, method, mass, step, round(score, 2)])
            print(f"seed {seed} | {method:8s} | mass x{mass} | scores after 0..{test_adapt_steps} steps: "
                  f"{[round(s) for s in scores]}")
save_csv(f"{folder}/all_results.csv", ["seed", "method", "mass", "adapt_steps", "return"], rows)

# ---------- summary table: mean and spread across seeds ----------
summary = []
for method in names:
    for mass in all_masses:
        for step in range(test_adapt_steps + 1):
            values = [r[4] for r in rows if r[1] == method and r[2] == mass and r[3] == step]
            if values:
                summary.append([method, mass, step, round(np.mean(values), 1), round(np.std(values), 1)])
save_csv(f"{folder}/summary_table.csv", ["method", "mass", "adapt_steps", "mean_return", "std_across_seeds"], summary)

with open(f"{folder}/summary_table.txt", "w") as f:
    for mass in all_masses:
        kind = "inside training range" if mass in test_masses_inside else "OUTSIDE training range"
        f.write(f"\nMass x{mass} ({kind}) - mean return (std across seeds)\n")
        f.write(f"{'method':55s}" + "".join(f"  step{s:<8d}" for s in range(test_adapt_steps + 1)) + "\n")
        for method in names:
            cells = [x for x in summary if x[0] == method and x[1] == mass]
            if cells:
                f.write(f"{names[method]:55s}" + "".join(f"  {c[3]:6.1f}({c[4]:4.1f})" for c in cells) + "\n")
print(open(f"{folder}/summary_table.txt").read())

# ---------- optional oracle values (from step 4b) ----------
oracle = {}
oracle_file = os.path.join(output_folder("step4b_oracle"), "oracle_returns.csv")
if os.path.exists(oracle_file):
    for line in open(oracle_file).read().strip().split("\n")[1:]:
        mass, value = line.split(",")
        oracle[float(mass)] = float(value)

# ---------- graph: one panel per test mass ----------
fig, axes = plt.subplots(1, len(all_masses), figsize=(4 * len(all_masses), 4), sharex=True)
for ax, mass in zip(axes, all_masses):
    for method in names:
        cells = [x for x in summary if x[0] == method and x[1] == mass]
        if cells:
            means = np.array([c[3] for c in cells])
            spread = np.array([c[4] for c in cells])
            ax.plot(range(len(means)), means, marker="o", label=names[method])
            ax.fill_between(range(len(means)), means - spread, means + spread, alpha=0.15)
    if mass in oracle:
        ax.axhline(oracle[mass], color="black", linestyle="--", label="Oracle (trained on this mass)")
    kind = "inside range" if mass in test_masses_inside else "OUTSIDE range"
    ax.set_title(f"Mass x{mass} ({kind})")
    ax.set_xlabel("Adaptation steps")
    ax.set_xticks(range(test_adapt_steps + 1))
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Return")
axes[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{folder}/adaptation_curves.png", dpi=150)

# ---------- graph: improvement gained by adapting (step 3 minus step 0) ----------
plt.figure(figsize=(7, 4))
width = 0.2
for i, method in enumerate(names):
    gains = []
    for mass in all_masses:
        a = [x[3] for x in summary if x[0] == method and x[1] == mass and x[2] == test_adapt_steps]
        b = [x[3] for x in summary if x[0] == method and x[1] == mass and x[2] == 0]
        gains.append(a[0] - b[0] if a and b else 0)
    plt.bar(np.arange(len(all_masses)) + i * width, gains, width, label=names[method])
plt.xticks(np.arange(len(all_masses)) + 1.5 * width, [f"x{m}" for m in all_masses])
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Test mass")
plt.ylabel(f"Return gained by {test_adapt_steps} adaptation steps")
plt.title("How much does adapting help?")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{folder}/adaptation_gain.png", dpi=150)
print("Saved to:", folder)
