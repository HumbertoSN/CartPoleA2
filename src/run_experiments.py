import os
import json
import csv
from stable_baselines3 import PPO, A2C
from env_utils import create_cartpole_env
from evaluate import evaluate_model  # Asegúrate que evaluate_model está bien definido

CONFIG_PLAN_PATH = "../config/experimento_plan.csv"
RESULTS_CSV_PATH = "../results/experimentos.csv"

# Diccionario para acceder al algoritmo
ALGORITHM_MAP = {
    "PPO": PPO,
    "A2C": A2C
}

def train_and_evaluate(algorithm, timesteps, learning_rate, gamma):
    # Crear configuración temporal
    config = {
        "algorithm": algorithm,
        "total_timesteps": int(timesteps),
        "learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "gravity": 9.8,
        "seed": 42
    }

    # Crear entorno
    env = create_cartpole_env()

    # Crear modelo
    model_class = ALGORITHM_MAP[algorithm]
    model = model_class(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        verbose=0
    )

    # Entrenar
    model.learn(total_timesteps=config["total_timesteps"])

    # Guardar modelo temporal
    model_name = f"{algorithm}_{timesteps}_{learning_rate}_{gamma}".replace(".", "")
    model_path = f"../models/{model_name}.zip"
    model.save(model_path)
    env.close()

    # Evaluar modelo
    media_recompensa = evaluate_model(model_path, algorithm=algorithm, config=config, write_csv=False)

    # Guardar resultado en CSV
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    with open(RESULTS_CSV_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([algorithm, timesteps, learning_rate, gamma, round(media_recompensa, 2)])

def main():
    # Si el archivo aún no tiene encabezado, lo escribimos
    if not os.path.exists(RESULTS_CSV_PATH):
        with open(RESULTS_CSV_PATH, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Algoritmo", "Timesteps", "Learning_Rate", "Gamma", "Recompensa_Media"])

    # Leer combinaciones del CSV plan
    with open(CONFIG_PLAN_PATH, mode="r") as plan_file:
        reader = csv.DictReader(plan_file)
        for row in reader:
            algorithm = row["Algoritmo"]
            timesteps = row["Timesteps"]
            learning_rate = row["Learning_Rate"]
            gamma = row["Gamma"]
            print(f"🔧 Ejecutando experimento: {algorithm}, {timesteps}, {learning_rate}, {gamma}")
            train_and_evaluate(algorithm, timesteps, learning_rate, gamma)

if __name__ == "__main__":
    main()