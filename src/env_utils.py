# src/env_utils.py

import gymnasium as gym

def create_cartpole_env(gravity=9.8, seed=42, render_mode=None):
    env = gym.make(
        "CartPole-v1",
        render_mode=render_mode  # Esto es clave para capturar frames
    )

    # Aplicar valores personalizados si fuera necesario
    env.reset(seed=seed)
    if hasattr(env.env, 'gravity'):
        env.env.gravity = gravity

    return env