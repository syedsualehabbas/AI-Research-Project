import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------- settings (change these if you want) ----------
num_points = 10          # examples the model sees from a new task
inner_lr = 0.01          # step size when adapting to one task
outer_lr = 0.001         # step size when training the starting weights
num_iterations = 5000    # how long to train (use 1000 for a quick test)
tasks_per_batch = 10     # how many tasks per training step - adaptation
test_steps = 10          # how many adaptation steps we test at the end
seed = 4

torch.manual_seed(seed)
np.random.seed(seed)


# ---------- the model: a small neural network ----------
def make_weights():
    # 1 input -> 40 -> 40 -> 1 output
    layer_sizes = [1, 40, 40, 1]
    weights = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        w = (torch.randn(n_in, n_out) / np.sqrt(n_in)).requires_grad_()
        b = torch.zeros(n_out, requires_grad=True)
        weights += [w, b]
    return weights


def predict(x, weights):
    # pass x through every layer (ReLU between layers)
    out = x
    for i in range(0, len(weights), 2):
        out = out @ weights[i] + weights[i + 1]
        if i < len(weights) - 2:
            out = torch.relu(out)
    return out


def get_loss(x, y, weights):
    # mean squared error between prediction and the true values
    return ((predict(x, weights) - y) ** 2).mean()


# ---------- tasks: each task is one sine wave ----------
def make_task():
    amplitude = np.random.uniform(0.1, 5.0)
    phase = np.random.uniform(0, np.pi)
    return amplitude, phase


def get_data(task, how_many):
    # random x values and the sine wave's y values for them
    amplitude, phase = task
    x = torch.rand(how_many, 1) * 10 - 5
    y = amplitude * torch.sin(x + phase)
    return x, y


def adapt(weights, x, y, steps, keep_graph):
    # a few gradient descent steps on one task
    # keep_graph=True lets MAML differentiate through these steps
    for _ in range(steps):
        grads = torch.autograd.grad(get_loss(x, y, weights), weights, create_graph=keep_graph)
        weights = [w - inner_lr * g for w, g in zip(weights, grads)]
    return weights


# ---------- test tasks (same for every method, so the comparison is fair) ----------
test_tasks = []
for _ in range(100):
    task = make_task()
    x_support, y_support = get_data(task, num_points)             # few points to adapt on
    x_test = torch.linspace(-5, 5, 100).unsqueeze(1)              # many points to check on
    y_test = task[0] * torch.sin(x_test + task[1])
    test_tasks.append((x_support, y_support, x_test, y_test))


# ---------- training methods ----------
def train_maml(first_order):
    weights = make_weights()
    optimizer = torch.optim.Adam(weights, lr=outer_lr)
    for _ in range(num_iterations):
        total_loss = 0
        for _ in range(tasks_per_batch):
            task = make_task()
            x_support, y_support = get_data(task, num_points)
            x_query, y_query = get_data(task, num_points)
            # adapt for one step, then measure the error AFTER adapting
            # STEP A: Practice on the support set (creates temporary weights)
            adapted = adapt(weights, x_support, y_support, 1, keep_graph=not first_order)

            # STEP B: Test on the query set (measures error AFTER practicing)
            total_loss = total_loss + get_loss(x_query, y_query, adapted)

        # STEP C: Update the permanent starting 'weights'

        # clears out any old gradients stored inside weights from the previous iteration.
        optimizer.zero_grad()

        # computes the average query error across all 10 sine waves.
        (total_loss / tasks_per_batch).backward()
        optimizer.step()
    return weights


def train_reptile():
    weights = make_weights()
    for i in range(num_iterations):
        step_size = 1.0 * (1 - i / num_iterations)       # starts at 1, shrinks to 0
        moves = [torch.zeros_like(w) for w in weights]
        for _ in range(tasks_per_batch):
            task = make_task()
            # train a copy of the weights on this task for 5 steps
            copy = [w.detach().clone().requires_grad_() for w in weights]
            for _ in range(5):
                x, y = get_data(task, num_points)
                grads = torch.autograd.grad(get_loss(x, y, copy), copy)
                copy = [(c - inner_lr * g).detach().requires_grad_() for c, g in zip(copy, grads)]
            # remember how far the copy moved from the original
            for m, c, w in zip(moves, copy, weights):
                m += (c.detach() - w.detach()) / tasks_per_batch
        # move the original weights toward the trained copies
        with torch.no_grad():
            for w, m in zip(weights, moves):
                w += step_size * m
    return weights


def train_pretrained():
    # normal training on all tasks mixed together (no meta-learning)
    weights = make_weights()
    optimizer = torch.optim.Adam(weights, lr=outer_lr)
    for _ in range(num_iterations):
        all_x, all_y = [], []
        for _ in range(tasks_per_batch):
            x, y = get_data(make_task(), 2 * num_points)   # same amount of data as MAML uses
            all_x.append(x)
            all_y.append(y)
        loss = get_loss(torch.cat(all_x), torch.cat(all_y), weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return weights


# ---------- testing: adapt on a few points, check error on new points ----------
def evaluate(start_weights):
    errors = np.zeros((len(test_tasks), test_steps + 1))
    for i, (x_support, y_support, x_test, y_test) in enumerate(test_tasks):
        weights = [w.detach().clone().requires_grad_() for w in start_weights]
        errors[i, 0] = get_loss(x_test, y_test, weights).item()      # before any adapting
        for step in range(1, test_steps + 1):
            grads = torch.autograd.grad(get_loss(x_support, y_support, weights), weights)
            weights = [(w - inner_lr * g).detach().requires_grad_() for w, g in zip(weights, grads)]
            errors[i, step] = get_loss(x_test, y_test, weights).item()
    average = errors.mean(axis=0)
    std_error = errors.std(axis=0) / np.sqrt(len(test_tasks))
    return average, std_error


# ---------- run everything ----------
if __name__ == "__main__":
    methods = {
        "Random start (no training)": make_weights,
        "Pretrained + fine-tune": train_pretrained,
        "Reptile": train_reptile,
        "First-order MAML": lambda: train_maml(first_order=True),
        "MAML (second-order)": lambda: train_maml(first_order=False),
    }

    results = {}
    for name, train in methods.items():
        start_time = time.time()
        weights = train()
        results[name] = evaluate(weights)
        avg = results[name][0]
        print(f"{name:27s} time {time.time() - start_time:6.1f}s | "
              f"error: step0 {avg[0]:.3f}  step1 {avg[1]:.3f}  step5 {avg[5]:.3f}  step10 {avg[10]:.3f}")

    # plot error vs number of adaptation steps
    plt.figure(figsize=(7, 4.5))
    for name, (avg, std_error) in results.items():
        steps = range(test_steps + 1)
        plt.plot(steps, avg, marker="o", label=name)
        plt.fill_between(steps, avg - std_error, avg + std_error, alpha=0.15)
    plt.yscale("log")
    plt.xlabel("Gradient steps on the new task (10 points)")
    plt.ylabel("Error on unseen points of the new sine")
    plt.title("Sine regression: how fast each method adapts")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("sine_results.png", dpi=150)
    plt.show()