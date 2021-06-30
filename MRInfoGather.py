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
            if(np.random.random() > 0.9999):
                print("Change speed!")
                with self.lock:
                    self.environment.set_targets_command((np.random.random((self.environment.xi.shape))-0.5))
            
            self.environment.update()
            time.sleep(self.environment.t)

env = Environment(10,10,0.01, True)

for i in range(1):
    env.add_target(np.random.random()*2.5-2.5, np.random.random()*2.5-2.5, 0.005, 0.005)
    env.add_agent(np.random.random()*2.5-2.5, np.random.random()*2.5-2.5, 0.005, 0.005)
    #env.add_target(np.random.random()*2.5-2.5, np.random.random()*2.5-2.5, 0.005, 0.005)

main = Main(env)
main.start()
#main.run()
plt.gcf()
plt.show()
main.join()