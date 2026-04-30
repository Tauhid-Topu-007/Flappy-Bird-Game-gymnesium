import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Create environment
env = gym.make("FlappyBird-v0", render_mode="human")
state, info = env.reset()

done = False

# Initialize pygame
pygame.init()

screen = pygame.display.get_surface()  # Gym created window already

while not done:
    action = 0  # default -> no flap (0), flap (1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1  # flap

    # Step environment
    state, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
pygame.quit()