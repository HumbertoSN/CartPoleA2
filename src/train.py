# src/train.py

import json
import os
from stable_baselines3 import PPO
from env_utils import create_cartpole_env

def train_model(config_path):
    # Cargar configuración
    with open(config_path, "r") as f:
        config = json.load(f)

    # Crear entorno
    env = create_cartpole_env()

    # Inicializar modelo con política compatible con DQN
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        tensorboard_log="./logs/",
        seed=config["seed"]
    )

    # Entrenar
    model.learn(total_timesteps=config["total_timesteps"])

    # Crear carpeta models si no existe
    os.makedirs("../models", exist_ok=True)

    # Guardar modelo entrenado con nombre correcto
    model_path = f"../models/{config['algorithm']}_cartpole"
    model.save(model_path)

    env.close()
    print(f"✅ Entrenamiento completado. Modelo guardado en {model_path}")

if __name__ == "__main__":
    train_model("../config/train_config.json")