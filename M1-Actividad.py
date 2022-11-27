from mesa import Agent, Model

from mesa.space import MultiGrid

from mesa.time import RandomActivation

from mesa.datacollection import DataCollector

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

import numpy as np
import pandas as pd

import time
import datetime

RAND = 1000

COUNTER = 0

class VacuumCleanerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.counter = 0

    def printCounter(self):
        print(self.counter)

    def step(self):
        if self.model.floor[self.pos[0]][self.pos[1]] == 1:
            self.model.floor[self.pos[0]][self.pos[1]] = 0

        choices = self.model.grid.get_neighborhood(self.pos, moore = False,
                                                   include_center = False)

        new_position = self.random.choice(choices)
        self.model.grid.move_agent(self, new_position)
            
        self.counter += 1

def get_grid(model):
    grid = np.zeros( (model.grid.width, model.grid.height) )
    for x in range (model.grid.width):
        for y in range (model.grid.height):
            if model.grid.is_cell_empty( (x, y) ) :
                grid[x][y] = model.floor[x][y] * 2
            else:
                grid[x][y] = 1
    return grid

class VacuumCleanerModel(Model):
    def __init__(self, width, height, num_agents, dirty_cells_percentage):
        self.num_agents = num_agents
        self.dirty_cells_percentage = dirty_cells_percentage
        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivation(self)
        self.floor = np.zeros( (width, height) )
        self.counter = 0
        self.all_clean = 0

        for i in range(self.num_agents):
            a = VacuumCleanerAgent(i, self)
            self.grid.place_agent(a, (0, 0))
            self.schedule.add(a)

        amount = int((width * height) * dirty_cells_percentage)
        for i in range(amount):
            finished = False
            while not finished:
                x = int(np.random.rand() * RAND) % width
                y = int(np.random.rand() * RAND) % height
                if self.floor[x][y] == 0:
                    self.floor[x][y] = 1
                    finished = True

        self.datacollector = DataCollector(model_reporters={"Grid": get_grid})

    def is_all_clean(self):
        return np.all(self.floor == 0)

    def if_is_all_clean(self):
        print("Number of movements by the agents: ", self.all_clean)

    def dirtyCells(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.floor[i][j] == 1:
                    self.counter += 1
        percentage = (GRID_SIZE * GRID_SIZE) * self.dirty_cells_percentage
        dirty = (self.counter * 100) / percentage
        print("Dirty cells percentage: {:,.2f}%".format(dirty))
            
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if not np.all(self.floor == 0):
            self.all_clean += 1


# Definimos el tamaño del Grid
GRID_SIZE = 40

# Definimos el número máximo de generaciones a correr
MAX_GENERATIONS = 500

# Registramos el tiempo de inicio y ejecutamos la simulación
start_time = time.time()
model = VacuumCleanerModel(GRID_SIZE, GRID_SIZE, 120, 0.5)
i = 1
while i <= MAX_GENERATIONS and not model.is_all_clean():
    model.step()
    i += 1

""" while not model.is_all_clean():
    model.step()
    i += 1 """

print("Steps: ", i)

model.dirtyCells()

model.if_is_all_clean()

    
# Imprimimos el tiempo que le tomó correr al modelo.
print('Tiempo de ejecución:', str(datetime.timedelta(seconds=(time.time() - start_time))))

# Obtenemos la información que almacenó el colector, este nos entregará un DataFrame de pandas que contiene toda la información.
all_grid = model.datacollector.get_model_vars_dataframe()

fig, axs = plt.subplots(figsize=(7,7))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(all_grid.iloc[0][0], cmap=plt.cm.binary)

def animate(i):
    patch.set_data(all_grid.iloc[i][0])
    
anim = animation.FuncAnimation(fig, animate, frames=MAX_GENERATIONS)
plt.show()