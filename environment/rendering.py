import pygame
import numpy as np
from pygame import gfxdraw

class EnergyRenderer:
    def __init__(self, env):
        self.env = env
        self.cell_size = 80
        self.width = self.env.grid_size * self.cell_size
        self.height = self.env.grid_size * self.cell_size
        self.colors = {
            'background': (40, 40, 40),
            'grid': (70, 70, 70),
            'agent': (66, 135, 245),
            'appliance': (245, 158, 66),
            'goal': (66, 245, 132)
        }
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Energy Usage Optimization")
        self.font = pygame.font.SysFont('Arial', 14)
        
    def render(self):
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        for i in range(self.env.grid_size + 1):
            pygame.draw.line(self.screen, self.colors['grid'], 
                           (0, i * self.cell_size),
                           (self.width, i * self.cell_size))
            pygame.draw.line(self.screen, self.colors['grid'],
                           (i * self.cell_size, 0),
                           (i * self.cell_size, self.height))
        
        # Draw appliances
        for app, pos in self.env.appliances.items():
            color = self.colors['goal'] if app == 'Goal' else self.colors['appliance']
            rect = pygame.Rect(
                pos[1] * self.cell_size + 2,
                pos[0] * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4
            )
            pygame.draw.rect(self.screen, color, rect)
            
            # Add appliance label
            text = self.font.render(app, True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        # Draw agent
        x, y = self.env.agent_pos
        center = (
            y * self.cell_size + self.cell_size // 2,
            x * self.cell_size + self.cell_size // 2
        )
        gfxdraw.filled_circle(
            self.screen, center[0], center[1], 
            self.cell_size // 3, self.colors['agent']
        )
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()