import threading
import numpy as np
import matplotlib.pyplot as plt
import time

from environment import Environment

env = Environment(10,10,0.01, True)

for i in range(2):
    env.add_target(np.random.random()*2.5+1.5, np.random.random()*2.5+1.5, 0.05, 0.05)
    env.add_agent(np.random.random()*2.5-2.5, np.random.random()*2.5-2.5, 0.01, 0.01)
    #env.add_target(np.random.random()*2.5-2.5, np.random.random()*2.5-2.5, 0.005, 0.005)

env.run()