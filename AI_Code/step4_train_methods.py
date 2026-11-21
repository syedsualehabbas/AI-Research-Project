# step4_train_methods.py  -  trains the three methods that learn from MANY masses:
#   1) "mixed"   : normal REINFORCE on random masses (this is also our domain-randomization baseline)
#   2) "fomaml"  : first-order MAML for RL
#   3) "reptile" : Reptile for RL
# Every method gets the SAME training budget (train_trajectories robot runs), so the comparison is fair.
# If a weights file already exists, that run is skipped, so you can stop and restart safely.
import os
import time
import numpy as np
import torch
from common import *

reptile_inner_steps = 3     # how many adaptation steps Reptile does per task
reptile_lr = 0.5            # how far Reptile moves toward the adapted weights

folder = output_folder("step4_training")


def train_mixed(cheetah, policy, iterations):
    # normal learning: every iteration uses runs from several different masses mixed together
    optimizer = torch.optim.Adam(policy.parameters(), lr=outer_lr)
    log = []
    for it in range(1, iterations + 1):
        batches = [cheetah.collect(policy, sample_mass(), trajectories_per_step)
                   for _ in range(tasks_per_iteration)]
        big_batch = combine(batches)
        loss = policy_gradient_loss(policy, big_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log.append(float(big_batch["rewards"].sum(axis=1).mean()))
        yield it, log[-1]


def train_fomaml(cheetah, policy, iterations):
    optimizer = torch.optim.Adam(policy.parameters(), lr=outer_lr)
    for it in range(1, iterations + 1):
        total_grads = [torch.zeros_like(p) for p in policy.parameters()]
        scores = []
        for _ in range(tasks_per_iteration):
            mass = sample_mass()
            fast = copy.deepcopy(policy)
            # inner step: practice with the current weights, learn a little (this is the "adaptation")
            sgd_step(fast, cheetah.collect(policy, mass, trajectories_per_step), inner_lr)
            # outer step: practice with the ADAPTED weights and see how good they are
            after = cheetah.collect(fast, mass, trajectories_per_step)
            scores.append(float(after["rewards"].sum(axis=1).mean()))
            loss = policy_gradient_loss(fast, after)
            grads = torch.autograd.grad(loss, list(fast.parameters()))
            # first-order trick: use the gradient at the adapted weights as the gradient for the start weights
            for total, g in zip(total_grads, grads):
                total += g / tasks_per_iteration
        for p, g in zip(policy.parameters(), total_grads):
            p.grad = g
        optimizer.step()
        yield it, float(np.mean(scores))


def train_reptile(cheetah, policy, iterations):
    for it in range(1, iterations + 1):
        total_move = [torch.zeros_like(p) for p in policy.parameters()]
        scores = []
        for _ in range(tasks_per_iteration):
            mass = sample_mass()
            fast = copy.deepcopy(policy)
            for _ in range(reptile_inner_steps):             # train a copy on this one mass
                batch = cheetah.collect(fast, mass, trajectories_per_step)
                sgd_step(fast, batch, inner_lr)
            scores.append(float(batch["rewards"].sum(axis=1).mean()))
            for move, f, p in zip(total_move, fast.parameters(), policy.parameters()):
                move += (f.detach() - p.detach()) / tasks_per_iteration
        with torch.no_grad():                                # move the start weights toward the copies
            for p, move in zip(policy.parameters(), total_move):
                p += reptile_lr * move
        yield it, float(np.mean(scores))


# robot runs used per iteration, so we can give every method the same total budget
runs_per_iteration = {
    "mixed": tasks_per_iteration * trajectories_per_step,
    "fomaml": tasks_per_iteration * 2 * trajectories_per_step,
    "reptile": tasks_per_iteration * reptile_inner_steps * trajectories_per_step,
}
trainers = {"mixed": train_mixed, "fomaml": train_fomaml, "reptile": train_reptile}

if __name__ == "__main__":
    for seed in seeds:
        for method, trainer in trainers.items():
            path = weights_path(method, seed)
            if os.path.exists(path):
                print(f"[skip] {method} seed {seed} already trained")
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            cheetah = MassCheetah(seed=seed)
            policy = Policy()
            iterations = train_trajectories // runs_per_iteration[method]
            print(f"\n=== training {method}, seed {seed}: {iterations} iterations ===")
            rows, start = [], time.time()
            for it, score in trainer(cheetah, policy, iterations):
                rows.append([it, it * runs_per_iteration[method], round(score, 2)])
                if it % 10 == 0:
                    print(f"iteration {it:4d} | return {score:8.1f} | time {time.time() - start:5.0f}s")
            torch.save(policy.state_dict(), path)
            save_csv(f"{folder}/curve_{method}_seed{seed}.csv", ["iteration", "robot_runs_used", "return"], rows)

    # training curves: one picture, one line per method (mean over seeds)
    plt.figure(figsize=(7, 4.5))
    for method in trainers:
        curves = []
        for seed in seeds:
            file = f"{folder}/curve_{method}_seed{seed}.csv"
            if os.path.exists(file):
                data = np.loadtxt(file, delimiter=",", skiprows=1)
                curves.append(data)
        if curves:
            n = min(len(c) for c in curves)
            used = curves[0][:n, 1]
            avg = np.mean([c[:n, 2] for c in curves], axis=0)
            smooth = np.convolve(avg, np.ones(5) / 5, mode="same") if n >= 5 else avg
            plt.plot(used, smooth, label=method)
    plt.xlabel("Robot runs used for training")
    plt.ylabel("Return during training")
    plt.title("Training curves (fomaml and reptile show return AFTER adapting)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{folder}/training_curves.png", dpi=150)
    print("Saved to:", folder)
