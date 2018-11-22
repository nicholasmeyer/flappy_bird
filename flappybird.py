#!/usr/bin/env python
import sys
import random

import pygame
import numpy as np

# game screen width and height
WIDTH, HEIGHT = (400, 708)

clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Score {}'.format(0))
background = pygame.image.load('assets/background.png').convert()
bird_states = {
    'no_flap': pygame.image.load('assets/no_flap.png').convert_alpha(),
    'flap_up': pygame.image.load('assets/flap_up.png').convert_alpha(),
    'flap_down': pygame.image.load('assets/flap_down.png').convert_alpha(),
    'dead': pygame.image.load('assets/dead.png').convert_alpha()
}
pole_top = pygame.image.load("assets/top.png").convert_alpha()
pole_bottom = pygame.image.load("assets/bottom.png").convert_alpha()


class EnvironmentInfo():
    """
    Class that handles all environment information
    """

    def __init__(self, vector_observations, rewards, local_done):
        self.vector_observations = vector_observations
        self.rewards = rewards
        self.local_done = local_done


class FlappyEnvironment():
    """
    Class that creates the environment for Flappy Bird
    """

    def __init__(self):
        # attributes for the flappy bird
        self.flapping = 'no_flap'
        self.y = 50
        self.floating_time = 0
        self.jump_size = 10
        self.gravity = 5
        self.dead = False
        self.coordinates = pygame.Rect(65, self.y, 50, 50)
        # attributes for the poles
        self.gap = 130
        self.x = 400
        self.offset = random.randint(-110, 110)
        self.top_coordinates = pygame.Rect(
            self.x, 0 - self.gap - self.offset - 10, pole_top.get_width() - 10, pole_top.get_height())
        self.bottom_coordinates = pygame.Rect(
            self.x, 360 + self.gap - self.offset + 10, pole_bottom.get_width() - 10, pole_bottom.get_height())
        # game score
        self.counter = 0
        # initialise rendering
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)

    def step(self, action):
        clock.tick(60)
        if self.floating_time < 12 and not self.dead:
            self.flapping = 'no_flap'

        if action == 1 and not self.dead:
            self.floating_time = 17
            self.gravity = 5
            self.jump_size = 10
            self.flapping = 'flap_up'

        # controls movement of the bird
        if self.floating_time < 5 and not self.dead:
            self.flapping = 'flap_down'
        if self.floating_time:
            self.jump_size -= 1
            self.floating_time -= 1
            self.y -= self.jump_size
        else:
            self.y += self.gravity
            self.gravity += 0.2
        self.coordinates[1] = self.y

        # constrols movement of the poles
        self.x -= 2
        if self.x < -80:
            self.x = 400
            self.counter += 1
            self.offset = random.randint(-110, 110)
        self.top_coordinates = pygame.Rect(
            self.x, 0 - self.gap - self.offset - 10, pole_top.get_width() - 10, pole_top.get_height())
        self.bottom_coordinates = pygame.Rect(
            self.x, 360 + self.gap - self.offset + 10, pole_bottom.get_width() - 10, pole_bottom.get_height())
        # updates game status
        if self.top_coordinates.colliderect(self.coordinates):
            self.dead = True
        if self.bottom_coordinates.colliderect(self.coordinates):
            self.dead = True
        if not 0 < self.y < 720:
            self.dead = True
            # self.reset()
        if self.dead:
            self.flapping = 'dead'
        t = list(self.top_coordinates)
        b = list(self.bottom_coordinates)
        vector_observations = np.array(sum([list(self.coordinates), list((
            t[0], t[3], t[0] + t[2], t[3])), list((b[0], b[3], b[0] + b[2], b[3]))], []))
        if not self.dead:
            reward = 0.1 + self.counter
            local_done = False
        else:
            reward = -10
            local_done = True
        env_info = EnvironmentInfo(vector_observations, reward, local_done)

        # handle game visual updates
        screen.fill((255, 255, 255))
        screen.blit(background, (0, 0))
        screen.blit(bird_states[self.flapping], (70, self.y))
        screen.blit(pole_top, (self.x, 0 - self.gap - self.offset))
        screen.blit(pole_bottom, (self.x, 360 + self.gap - self.offset))
        pygame.display.set_caption('Score {}'.format(self.counter))
        pygame.display.update()

        return env_info

    def reset(self):
        """
        Method to reset the game state
        """
        self.x = 400
        self.y = 50
        self.gravity = 5
        self.counter = 0
        self.dead = False
        self.offset = random.randint(-110, 110)
        pygame.display.set_caption('Score {}'.format(self.counter))

        t = list(self.top_coordinates)
        b = list(self.bottom_coordinates)
        vector_observations = np.array(sum([list(self.coordinates), list((
            t[0], t[3], t[0] + t[2], t[3])), list((b[0], b[3], b[0] + b[2], b[3]))], []))
        reward = 0
        local_done = False
        env_info = EnvironmentInfo(vector_observations, reward, local_done)
        return env_info

    def action_size(self):
        return 2

    def state_size(self):
        return 12

    def close(self):
        sys.exit()


if __name__ == "__main__":
    env = FlappyEnvironment()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
            if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not env.dead:
                env.step(1)
        env.step(0)
        if env.dead:
            env.reset()
