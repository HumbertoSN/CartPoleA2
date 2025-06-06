import sys
import pygame
import numpy as np
from env_utils import create_cartpole_env
from stable_baselines3 import PPO, DQN, A2C

# Configuración de la ventana
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
CART_COLOR = (0, 100, 255)
POLE_COLOR = (220, 20, 60)
BG_COLOR = (240, 240, 240)

def run_simulation(model_path, gravity=9.8):
    env = create_cartpole_env(gravity=gravity)

    # Detectar el tipo de modelo por nombre
    if "PPO" in model_path:
        model = PPO.load(model_path)
    elif "DQN" in model_path:
        model = DQN.load(model_path)
    elif "A2C" in model_path:
        model = A2C.load(model_path)
    else:
        raise ValueError("Nombre de modelo no reconocido. Usa PPO, DQN o A2C en el nombre.")

    obs, _ = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("CartPole RL - Agente en Acción")
    clock = pygame.time.Clock()

    def scale_state(state):
        cart_x, _, pole_angle, _ = state
        cart_x_px = int(cart_x * 100 + SCREEN_WIDTH // 2)
        pole_length = 150
        pole_end_x = cart_x_px + pole_length * np.sin(pole_angle)
        pole_end_y = SCREEN_HEIGHT // 2 - pole_length * np.cos(pole_angle)
        return cart_x_px, pole_end_x, pole_end_y

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        screen.fill(BG_COLOR)
        cart_x, pole_x, pole_y = scale_state(obs)
        pygame.draw.rect(screen, CART_COLOR, (cart_x - 40, SCREEN_HEIGHT // 2 - 20, 80, 40))
        pygame.draw.line(screen, POLE_COLOR, (cart_x, SCREEN_HEIGHT // 2), (pole_x, pole_y), 6)

        font = pygame.font.SysFont(None, 36)
        action_text = font.render(f"Acción: {'IZQUIERDA' if action == 0 else 'DERECHA'}", True, (0, 0, 0))
        screen.blit(action_text, (20, 20))

        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            obs, _ = env.reset()

    pygame.quit()
    env.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Debes pasar la ruta al modelo como argumento. Ejemplo:\npython src/visualize.py ../models/DQN_cartpole.zip")
        sys.exit(1)

    model_path = sys.argv[1]
    run_simulation(model_path)