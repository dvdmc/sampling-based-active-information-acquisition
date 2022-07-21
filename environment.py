from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import time


from graph import AIANode, AIATree
from utils import sigma_measure, kalman_predict, kalman_update
from sampling_based_active_information_acquisition import sampling_based_active_information_acquisition

#plt.rcParams["figure.figsize"] = (20,20)

class Environment:
    def __init__(self, width, height, dt, visualize=True):
        
        # Environment props
        self.width = width
        self.height = height
        self.dt = dt



        # AGENTS (ROW: agent/target, COL: pos x pos y)
        self.qi = np.zeros((0,2))
        #self.sigma_agents = np.zeros((0,2)) No motion noise
        self.n_agents = 0
        self.u_agents = np.zeros((0,2))
        # Secuence of actions to perform
        self.u_path = []

        # TARGETS Real state values
        self.xi = np.zeros((0,2))
        self.sigma_targets = np.zeros((0,2))
        self.n_targets = 0
        self.u_targets = np.zeros((0,2))
        self.Q = []
        # Estimated state values. These are represented differently: serialized
        self.x_est = []
        self.P_est = []

        self.visualize = visualize
        # Visualization objects
        self.fig, self.ax = plt.subplots()

        ##############################################################################
        ### ALGORITHM PARAMETERS TO TUNE
        # Number of trials of the algorithm
        self.max_n = 100
        # Minimum node cost admissible as solution
        self.delta = 1.8e-11
        # Max and min velocities for the agents: define the motion primitives
        self.max_vel = 0.1
        self.min_vel = 0.000001
        ## COVARIANCES from: targets (process noise) and measure (function in utils)
        ## P_v in sample_pv function in graph.py
        ##############################################################################

        self.tree = None

        # variables for metrics
        self.hist_qi = []
        self.hist_xi = []
        self.max_t = []
        self.length_path_chosen = []

    def add_agent(self, x, y, sigma_x, sigma_y):
        self.n_agents += 1
        # Expand matrix
        self.qi = np.vstack((self.qi, np.array([x,y])))
        self.u_agents = np.vstack((self.u_agents, np.array([0,0])))
        # Add commands of action (0,0) for the new agent until the current sequence is finished
        for i in range(len(self.u_path)):
            self.u_path[i] = np.vstack((self.u_path[i], np.array([0,0])))
    
    def add_target(self, x, y, sigma_x, sigma_y):
        self.n_targets += 1
        # Expand matrix
        self.xi = np.vstack((self.xi, np.array([x,y])))
        self.u_targets = np.vstack((self.u_targets, (np.random.random((1,2))*5-2.5)))
        self.sigma_targets = np.vstack((self.sigma_targets, np.array([sigma_x, sigma_y])))
        # Add estimated position to the matrix
        self.x_est.append(x)
        self.x_est.append(y)
        # Create Q matrix
        Q_aux = np.identity(len(self.Q)+2)
        # Add old values if any
        if np.size(self.Q) != 0:
            Q_aux[:np.shape(self.Q)[0],:np.shape(self.Q)[1]] = self.Q
        # Add new values
        Q_aux[-2, -2] = sigma_x ** 2
        Q_aux[-1, -1] = sigma_y ** 2
        self.Q = Q_aux
        # Add cov estimated to the matrix
        P_est_aux = np.identity(len(self.P_est)+2)
        if np.size(self.P_est) != 0:
            P_est_aux[:np.shape(self.P_est)[0],:np.shape(self.P_est)[1]] = self.P_est
        P_est_aux[-2, -2] = sigma_x ** 2
        P_est_aux[-1, -1] = sigma_y ** 2
        self.P_est = P_est_aux

    def update(self):
        '''
            Update the environment dynamics (agents and state)
        '''
        # self.update_agents_command()

        # Update agents according to Eq(1)
        if(self.n_agents > 0):
            self.hist_qi.append(self.qi)
            self.qi += self.u_agents # * self.dt Not needed since controls are motion primitives

        # Update state (targets) according to Eq(2)
        if(self.n_targets > 0):
            # Bounce if targets will go offscreen
            for idx, pos in enumerate(self.xi):
                if((pos[0] > self.width/2-0.5 and self.u_targets[idx][0] > 0) or (pos[0] < -self.width/2+0.5 and self.u_targets[idx][0] < 0)):
                    self.u_targets[idx][0] = -self.u_targets[idx][0]
                if((pos[1] > self.height/2-0.5 and self.u_targets[idx][1] > 0) or (pos[1] < -self.height/2+0.5 and self.u_targets[idx][1] < 0)):
                    self.u_targets[idx][1] = -self.u_targets[idx][1]
            # Apply movement to targets
            self.hist_xi.append(self.xi)
            noise_targets = np.random.randn(self.xi.shape[0], self.xi.shape[1]) * self.sigma_targets
            self.xi += self.u_targets * self.dt + noise_targets

        if(self.n_agents > 0):
            self.z, self.R = self.get_measurements(self.qi)
            x_est, P_est = kalman_predict(self.x_est, self.P_est, self.Q)
            for robot in range(self.n_agents):
                self.x_est, self.P_est = kalman_update(x_est, self.z[robot,:], P_est , np.diag(self.R[robot,:])) # get uncertainity in the new configuration

    def update_agents_command(self):
        '''
            Update the agents command before each environemnt update
        '''
        if self.n_agents > 0:
            # If new execution or previous path has finished
            if (len(self.u_path) == 0):
                self.u_path, self.tree = sampling_based_active_information_acquisition(self.max_n, self, self.delta)
            self.set_agents_command(self.u_path[0])
            self.u_path.pop(0)

    def set_agents_command(self, u_agents):
        self.u_agents = u_agents

    def set_targets_command(self, u_targets):
        self.u_targets = u_targets
    
    def free_space(self, p):
        # In our case, we assume that there is no inaccessible space
        return True
    
    def run(self):
        if self.visualize:
            # Configure plot
            self.ani = FuncAnimation(self.fig, self.main, interval=self.dt, blit=True)
            plt.show()

    def main(self,_):
        if(np.random.random() > 0.999999):
            print("Change speed!")
            with self.lock:
                self.set_targets_command((np.random.random((self.xi.shape)))*0.5)
        
        self.update()
        return self.plot_env()

    def sample_fu(self):
        # In the approach of the paper, where robots have a limited sensor range,
        # the approach taken is to bring each robot closer to each target until
        # they may sense the target. In our case, this "cheat" is not needed, as we
        # use an infinite sensor range, and the np.size(V[i])preferred final u will be always selected
        # using the covariance of the measurements. Thus, the sampling of u is completely
        # random.
        # TODO: implement sensor range and on-the-fly target assignment
        u_possibilities = [self.max_vel, -self.max_vel, self.min_vel, -self.min_vel]
        return np.random.choice(u_possibilities, (np.shape(self.u_agents)))

    def get_measurements(self, pi):
        '''
            Get the noisy position of the targets by each agent and stores it as follows:
            Row: Different agents
            Columns: Alternate x/y and targets for the same agent
            Also gets the covariance marix depending on the distance and stores it as follows:
            Row: DIfferent agents
            Columns: diagonal of the covariance matrix
        '''
        z = np.zeros((self.n_agents, 2*self.n_targets))
        R = np.zeros((self.n_agents, 2*self.n_targets))
        for j in range(self.n_agents):
            for i in range(self.n_targets):
                dist_x = pi[j,0] - self.xi[i,0]
                dist_y = pi[j,1] - self.xi[i,1]
                dist_l = np.sqrt(dist_x**2 + dist_y**2)
                # Construct uncertainity of the current measurements, depending on the distance
                sigma_x = sigma_measure(dist_l)
                sigma_y = sigma_measure(dist_l)
                z[j, 2*i] = self.xi[i,0] + np.random.random() * sigma_x
                z[j, 2*i+1] = self.xi[i,1] + np.random.random() * sigma_y
                R[j, 2*i] = sigma_x**2
                R[j, 2*i+1] = sigma_y**2

        return z, R
                

    def plot_env(self):
        plt.cla()
        plt.xlim(-self.width/2, self.width/2)
        plt.ylim(-self.height/2, self.height/2)
        self.fig.set_size_inches((10, 10))
        actors = []
        actors.append(self.ax.scatter(self.qi[:,0], self.qi[:,1], 14, 'b', 'o'))
        actors.append(self.ax.scatter(self.xi[:,0], self.xi[:,1], 14, 'r', 'x'))

        actors.append(self.ax.scatter(self.x_est[::2], self.x_est[1::2], 14, 'g', 'o'))
        print(self.P_est)
        for i in range(self.n_targets):
                ellipse = Ellipse((self.x_est[2*i],self.x_est[2*i+1]), width=np.sqrt(self.P_est[2*i][2*i])*5, height=np.sqrt(self.P_est[2*i+1][2*i+1])*5, color='b', alpha=0.2)
                actors.append(self.ax.add_patch(ellipse))

        # for nodes in self.tree.nodes:
        #     actors.append(plt.scatter(nodes.p[:,0], nodes.p[:,1], 1, c='orange', marker='x'))

        # data = []
        # for edge in self.tree.edges:
        #     for j in range(self.n_agents):
        #         data.append((edge[0].p[j,0], edge[1].p[j,0]))
        #         data.append((edge[0].p[j,1], edge[1].p[j,1]))
        #         data.append('gray')
        
        # actors.append(plt.plot(*data, linewidth=0.5))

        return actors
   