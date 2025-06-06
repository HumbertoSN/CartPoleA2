# src/evaluate.py

import os
import sys
import numpy as np
import imageio
import csv
from stable_baselines3 import PPO, A2C, DQN
from env_utils import create_cartpole_env

def evaluate_model(model_path, n_episodes=1, video_path=None, algorithm="PPO", config=None, write_csv=True):
    render_mode = "rgb_array"
    env = create_cartpole_env(render_mode=render_mode)
    
    # Cargar modelo según algoritmo
    model_class = {"PPO": PPO, "A2C": A2C, "DQN": DQN}.get(algorithm)
    if not model_class:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")
    
    model = model_class.load(model_path)
    all_frames = []

    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            obs_input = np.array(obs, dtype=np.float32)
            action, _ = model.predict(obs_input)
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()
            all_frames.append(frame)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    mean_reward = np.mean(rewards)

    # Ruta del video
    if video_path is None:
        base = os.path.basename(model_path).replace(".zip", "")
        video_path = f"video/cartpole_{base}_agent.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    imageio.mimsave(video_path, all_frames, fps=30)

    print(f"🎥 Video guardado en: {video_path}")
    print(f"✅ Recompensa media: {mean_reward:.2f}")

    # Guardar resultado en CSV si corresponde
    if config and write_csv:
        os.makedirs("results", exist_ok=True)
        csv_path = "results/experimentos.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Algoritmo", "Timesteps", "Learning_Rate", "Gamma", "Recompensa_Media"])
            writer.writerow([
                algorithm,
                config["total_timesteps"],
                config["learning_rate"],
                config["gamma"],
                round(mean_reward, 2)
            ])

    env.close()
    return mean_reward

if __name__ == "__main__":
    import json

    model_path = sys.argv[1]
    config_path = "../config/train_config.json"
    algorithm = "PPO"

    if len(sys.argv) > 2:
        config_path = sys.argv[2]
    if len(sys.argv) > 3:
        algorithm = sys.argv[3]

    with open(config_path, "r") as f:
        config = json.load(f)

    evaluate_model(model_path, algorithm=algorithm, config=config)