import threading
import numpy as np
import matplotlib.pyplot as plt
import time

from environment import Environment

class Main(threading.Thread):
    def __init__(self, environment):
        threading.Thread.__init__(self)
        self.lock = threading.Lock()

        self.environment = environment

    def run(self):
        while(True):
            # Movement of targets eq.(2):
            if(np.random.random() > 0.999999):
                print("Change speed!")
                with self.lock:
                    self.environment.set_targets_command((np.random.random((self.environment.xi.shape))*10))
            
            self.environment.update()
            time.sleep(0.01)

env = Environment(10,10,0.001, True)

for i in range(4):
    env.add_target(np.random.random()*2.5-2.5, np.random.random()*2.5-2.5, 0.005, 0.005)
    env.add_agent(np.random.random()*2.5-2.5, np.random.random()*2.5-2.5, 0.005, 0.005)
    #env.add_target(np.random.random()*2.5-2.5, np.random.random()*2.5-2.5, 0.005, 0.005)

main = Main(env)
main.start()
plt.gcf()
plt.show()
main.join()