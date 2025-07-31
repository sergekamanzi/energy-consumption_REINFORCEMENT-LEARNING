import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

class EnergyEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode=None):
        super(EnergyEnv, self).__init__()
        self.grid_size = 5
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, 
                                          shape=(self.grid_size, self.grid_size),
                                          dtype=np.float32)
        
        
        self.appliances = {
            'AC': {'pos': (0,0), 'energy': -3, 'desc': "High energy use"},
            'Fridge': {'pos': (0,3), 'energy': -1, 'desc': "Constant use"},
            'TV': {'pos': (1,1), 'energy': -2, 'desc': "Turn off when not needed"},
            'Oven': {'pos': (1,4), 'energy': -4, 'desc': "Very high energy"},
            'Meter': {'pos': (2,2), 'energy': 0, 'desc': "Energy monitor"},
            'Lights': {'pos': (3,0), 'energy': -1, 'desc': "Turn off lights"},
            'PC': {'pos': (3,1), 'energy': -2, 'desc': "Sleep mode helps"},
            'Washer': {'pos': (3,3), 'energy': -3, 'desc': "Run full loads"},
            'Solar': {'pos': (4,2), 'energy': +2, 'desc': "Renewable energy!"},
            'Goal': {'pos': (4,4), 'energy': 0, 'desc': "Target destination"}
        }
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.current_energy = 0
        self.max_energy = 10
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
       
        empty_positions = [(i,j) for i in range(self.grid_size) 
                         for j in range(self.grid_size) 
                         if (i,j) not in [app['pos'] for app in self.appliances.values()]]
        self.agent_pos = empty_positions[self.np_random.choice(len(empty_positions))]
        
        self.current_energy = 0
        self.visited = set()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}
    
    def step(self, action):
        
        x, y = self.agent_pos
        if action == 0: x = max(0, x-1)     
        elif action == 1: y = min(self.grid_size-1, y+1)  
        elif action == 2: x = min(self.grid_size-1, x+1)
        elif action == 3: y = max(0, y-1)  
        
        self.agent_pos = (x, y)
        
        
        reward = -0.1 
        terminated = False
        truncated = False
        current_appliance = None
        
        
        for app, data in self.appliances.items():
            if self.agent_pos == data['pos']:
                current_appliance = app
                self.current_energy += data['energy']
                
                if app == 'Goal':
                    
                    energy_score = max(0, 10 - abs(self.current_energy))
                    reward = 5 + energy_score
                    terminated = True
                elif app not in self.visited:
                    reward = max(0, 2 + data['energy']) 
                    self.visited.add(app)
        
       
        self.current_energy = max(-10, min(10, self.current_energy))
        
        if self.render_mode == "human":
            self._render_frame(current_appliance)
            
        return self._get_obs(), reward, terminated, truncated, {}
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _get_obs(self):
        
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        x, y = self.agent_pos
        obs[x,y] = 1.0
        
        
        for data in self.appliances.values():
            pos = data['pos']
            obs[pos] = 0.5
            
        return obs
    
    def _render_frame(self, current_appliance=None):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((600, 600))
            pygame.display.set_caption("Energy Usage Optimization")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((600, 600))
        canvas.fill((255, 255, 255))
        pix_square_size = 400 // self.grid_size
        
        
        for app, data in self.appliances.items():
            x, y = data['pos']
            energy = data['energy']
            
            # Color coding based on energy impact
            if energy < -2:  # High energy consumers
                color = (255, 100, 100)  # Red
            elif energy < 0:  # Moderate energy consumers
                color = (255, 200, 100)  # Orange
            elif energy > 0:  # Energy positive
                color = (100, 255, 100)  # Green
            else:  # Neutral
                color = (200, 200, 200)  # Gray
            
            if app == "Goal":
                color = (100, 100, 255)  # Blue for goal
            
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    y * pix_square_size + 100,  # Offset for legend
                    x * pix_square_size + 50,
                    pix_square_size,
                    pix_square_size,
                ),
            )
            
            # Draw appliance label
            font = pygame.font.SysFont('Arial', 12)
            text = font.render(app, True, (0, 0, 0))
            canvas.blit(text, (y * pix_square_size + 105, x * pix_square_size + 55))
        
        # Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent_pos[1] * pix_square_size + 100 + pix_square_size // 2,
             self.agent_pos[0] * pix_square_size + 50 + pix_square_size // 2),
            pix_square_size // 3,
        )
        
        # Draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (100, 50 + pix_square_size * x),
                (100 + 400, 50 + pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (100 + pix_square_size * x, 50),
                (100 + pix_square_size * x, 50 + 400),
                width=1,
            )

        # ====== Add Legend and Status ======
        legend_x = 20
        legend_y = 450
        
        # Energy meter
        energy_percent = (self.current_energy + 10) / 20  # Convert to 0-1 range
        energy_color = (
            int(255 * (1 - energy_percent)),  # More red when energy is high
            int(255 * energy_percent),        # More green when energy is low
            0
        )
        
        pygame.draw.rect(canvas, (200, 200, 200), (legend_x, legend_y, 200, 30))
        pygame.draw.rect(canvas, energy_color, (legend_x, legend_y, int(200 * energy_percent), 30))
        pygame.draw.rect(canvas, (0, 0, 0), (legend_x, legend_y, 200, 30), 2)  # Border
        
        font = pygame.font.SysFont('Arial', 16)
        energy_text = font.render(f"Energy Usage: {self.current_energy}", True, (0, 0, 0))
        canvas.blit(energy_text, (legend_x, legend_y - 25))
        
        # Color legend
        legend_items = [
            ((100, 255, 100), "Good (Solar)"),
            ((255, 200, 100), "Moderate (Fridge, Lights)"),
            ((255, 100, 100), "High (AC, Oven, Washer)"),
            ((100, 100, 255), "Goal")
        ]
        
        for i, (color, text) in enumerate(legend_items):
            pygame.draw.rect(canvas, color, (legend_x, legend_y + 50 + i*30, 20, 20))
            text_surface = font.render(text, True, (0, 0, 0))
            canvas.blit(text_surface, (legend_x + 25, legend_y + 50 + i*30))
        
        # Current appliance info
        if current_appliance:
            app_data = self.appliances[current_appliance]
            info_text = [
                f"Current: {current_appliance}",
                f"Impact: {app_data['energy']} energy",
                app_data['desc']
            ]
            
            for i, line in enumerate(info_text):
                text_surface = font.render(line, True, (0, 0, 0))
                canvas.blit(text_surface, (350, legend_y + 50 + i*25))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), 
                axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.quit()