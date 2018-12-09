#!/usr/bin/env python
import sys
import random

import pygame
import numpy as np

from flappybird_utils import preprocess_frame

# game screen width and height
WIDTH, HEIGHT = (400, 708)
ACTION_SIZE = 2
STATE_SIZE = (80, 80)
TERMINAL_REWARD = -5
ACHIEVEMENT_REWARD = 1
NON_TERMINAL_REWARD = 0.1
RED = (255,   0,   0)
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
    Stores all environment information that is needed to learn
    """

    def __init__(self, vector_observations, rewards, local_done):
        self.vector_observations = vector_observations
        self.rewards = rewards
        self.local_done = local_done


class FlappyEnvironment():
    """
    Creates the environment for Flappy Bird
    """

    def __init__(self):
        # attributes for the flappy bird
        self.flapping = 'no_flap'
        self.y = 350
        self.floating_time = 0
        self.jump_size = 10
        self.gravity = 5
        self.dead = False
        self.coordinates = pygame.Rect(65, self.y, 50, 50)
        # attributes for the poles
        self.gap = 150
        self.x = 400
        self.offset = random.randint(-110, 110)
        self.top_coordinates = pygame.Rect(
            self.x, 0 - self.gap - self.offset - 10, pole_top.get_width() - 10, pole_top.get_height())
        self.bottom_coordinates = pygame.Rect(
            self.x, 360 + self.gap - self.offset + 10, pole_bottom.get_width() - 10, pole_bottom.get_height())
        # game score
        self.game_score = 0
        # initialise rendering
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        self.add = False

    def step(self, action):
        reward = 0
        local_done = False
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
            self.game_score += 1
            self.offset = random.randint(-110, 110)
            reward = ACHIEVEMENT_REWARD
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
        # collect information to train network
        vector_observations = self.get_state()
        if not self.dead and not reward:
            reward = NON_TERMINAL_REWARD
        elif not reward:
            reward = TERMINAL_REWARD
            local_done = True
        env_info = EnvironmentInfo(vector_observations, reward, local_done)
        # handle game visual updates
        self.display()
        return env_info

    def reset(self):
        """
        Method to reset the game state
        """
        self.x = 400
        self.y = 350
        self.gravity = 5
        self.game_score = 0
        self.dead = False
        self.offset = random.randint(-110, 110)
        # reset information regarding environment state
        pygame.display.set_caption('Score {}'.format(self.game_score))
        self.display()
        vector_observations = self.get_state()
        reward = 0
        local_done = False
        env_info = EnvironmentInfo(vector_observations, reward, local_done)
        return env_info

    def display(self):
        screen.fill((0, 0, 0))
        screen.blit(background, (0, 0))
        screen.blit(bird_states[self.flapping], (70, self.y))
        screen.blit(pole_top, (self.x, 0 - self.gap - self.offset))
        screen.blit(pole_bottom, (self.x, 360 + self.gap - self.offset))
        pygame.display.set_caption('Score {}'.format(self.game_score))
        pygame.display.update()

    def get_state(self):
        frame = pygame.display.get_surface().convert_alpha()
        frame = pygame.surfarray.array3d(frame)
        frame = preprocess_frame(frame, self.y)
        return frame

    def action_size(self):
        return ACTION_SIZE

    def state_size(self):
        return STATE_SIZE

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
