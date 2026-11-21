import numpy as np, gymnasium as gym, imageio

def run(mass_scale, seed=0, steps=200, record=False):
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    env.reset(seed=seed)
    env.unwrapped.model.body_mass[:] *= mass_scale   # the "task": change body mass
    rng = np.random.default_rng(seed)
    frames, total = [], 0.0
    for _ in range(steps):
        a = rng.uniform(-1, 1, size=6)
        obs, r, term, trunc, info = env.step(a)
        total += r
        if record: frames.append(env.render())
    env.close()
    return total, frames

for scale in [1.0, 2.0]:
    total, frames = run(scale, record=True)
    print(f"mass x{scale}: total reward = {total:.2f}")
    imageio.mimsave(f"cheetah_mass_{scale}.mp4", frames, fps=30)