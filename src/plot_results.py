import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = "../results/experimentos.csv"

def main():
    df = pd.read_csv(RESULTS_CSV)

    # Asegurarse que los valores numéricos sean del tipo correcto
    df["Learning_Rate"] = df["Learning_Rate"].astype(float)
    df["Gamma"] = df["Gamma"].astype(float)
    df["Timesteps"] = df["Timesteps"].astype(int)

    # Colores por algoritmo
    colors = {"PPO": "blue", "A2C": "green"}

    # 1. Gráfico por Learning Rate
    plt.figure()
    for algo in df["Algoritmo"].unique():
        subset = df[df["Algoritmo"] == algo]
        plt.plot(subset["Learning_Rate"], subset["Recompensa_Media"], 'o-', label=algo, color=colors[algo])
    plt.title("Recompensa Media vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Recompensa Media")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../results/plot_learning_rate.png")
    plt.show()

    # 2. Gráfico por Gamma
    plt.figure()
    for algo in df["Algoritmo"].unique():
        subset = df[df["Algoritmo"] == algo]
        plt.plot(subset["Gamma"], subset["Recompensa_Media"], 'o-', label=algo, color=colors[algo])
    plt.title("Recompensa Media vs Gamma")
    plt.xlabel("Gamma")
    plt.ylabel("Recompensa Media")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../results/plot_gamma.png")
    plt.show()

    # 3. Gráfico por Timesteps
    plt.figure()
    for algo in df["Algoritmo"].unique():
        subset = df[df["Algoritmo"] == algo]
        plt.plot(subset["Timesteps"], subset["Recompensa_Media"], 'o-', label=algo, color=colors[algo])
    plt.title("Recompensa Media vs Timesteps")
    plt.xlabel("Timesteps")
    plt.ylabel("Recompensa Media")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../results/plot_timesteps.png")
    plt.show()

if __name__ == "__main__":
    main()