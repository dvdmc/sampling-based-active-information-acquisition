import numpy as np

class AIANode:
    '''
        Nodes of the graph. They are defined by:
        - ID: unique ID in the tree.
        - Timestamp: models the time of that node. It can be also understood as number of hops.
        - P: robot configuration. Shape depends on the number of robots.
        - X_est: estimated position of the targets. There is no Kalman prediction (identity).
            So this comes from the Kalman update of the parent node.
        - Cov: covariance estimated matrix. Must be below delta (eq 5.b).
        - Cost: accumulated cost from the root (parent_cost + det(cov)). Minimization value (eq 5).
        - Children: references of the children of this node.
        - Parents: references of the parents (actually one single parent as it is a tree).
        - Reaching motion: the control command used to reach this node from the parent and used to reconstruct the path.
    '''
    def __init__(self, p, x_est, cov, reaching_motion):
        self.id = None
        self.timestamp = None
        self.p = p
        self.x_est = x_est
        self.cov = cov
        self.cost = None
        self.children = []
        self.parents = []
        self.reaching_motion = reaching_motion

class AIATree:
    '''
        Tree used for the algorithm. It is only expanded so only the append_child method is implemented.
        It is defined by:
        - Nodes: of type AIANode (above).
        - Edges: tuples with references to: (parent, child).
        - id_count: unique identifier count for new nodes.
        - v_k: dictionary that models the set of different robot-configurations. The keys are the configurations converted
            to strings in order to make them hashable and each value is the set of nodes that have the same robot-configuration
        - t_max: current max number of hops. It is mainly used to sample fv
        - t_max_set: reference to the nodes that are at a t_max distance from the root and are more probable to be sampled.
    '''
    def __init__(self, root):
        assert root.id == 0 and root.timestamp == 0 and root.reaching_motion.all() != None and root.cost.all() != None, "Root not properly initialized"
        self.nodes = [root] # Hold a reference to all the nodes
        self.edges = [] # Edges are stored as: from-to tuple of nodes. However, nodes are connected by the Node class.
        self.id_count = 1 # This id is incremented every time a new child is added
        self.v_k = {(root.p).tostring():[root]} # v_k are the sets of nodes with the same robot-configuration. Main way of sampling nodes.
        self.t_max = 0 # Max number of hops
        self.t_max_set = [root] # Set of all the nodes that are at t_max 

    def append_child(self, parent, child, reach_cost):
        '''
            Inserts a new child and maintains the tree structure while doing it.
            - Parent: reference to the parent
            - Child: reference to the child
            - Reach cost: det(P(t)) for the child
        '''
        assert child.reaching_motion.all() != None, "Node not properly initialized"
        ## Init child node in the tree
        # Assign an unique id
        child.id = self.id_count
        self.id_count += 1
        # Assign one more timestamp/hops in the tree == branch length
        child.timestamp = parent.timestamp + 1
        # Compute cost to reach the node
        child.cost = parent.cost + reach_cost

        # Add node and edge
        self.nodes.append(child)
        self.edges.append((parent, child))

        ## Link nodes
        parent.children.append(child)
        child.parents.append(parent)

        ## Maintain the t_max and t_max_set in order to apply the biased sample of f_v
        if child.timestamp > self.t_max:
            self.t_max = child.timestamp
            self.t_max_set = [child]
        elif child.timestamp == self.t_max:
            self.t_max_set.append(child)

        # Introduce it into v_k or create a new v_k
        if child.p.tostring() in self.v_k.keys():
            self.v_k[(child.p).tostring()].append(child)
        else:
            self.v_k[(child.p).tostring()] = [child]


    def sample_fv(self, p_v = 0.7):
        '''
            The sampling strategy is based on sec V.A: gives more priority to leaves
            - p_v: probability of sampling v_k that are at t_max
        '''
        ## Construct k_max and V\k_max checking if the V in K_n are in t_max_set
        k_max = []
        k_other = []

        # Iterate over the robot-configurations in v_k
        for key, values in self.v_k.items():
            is_k_max = False
            for value in values:
                # If any is in the set of L_max, the configuration is in k_max
                if value in self.t_max_set:
                    is_k_max = True
                    k_max.append(np.fromstring(key, dtype=float))
                    break
            if not is_k_max:
                k_other.append(np.fromstring(key, dtype=float))
        
        # Construct a weight vector for sampling probabilities
        k_all = k_max + k_other
        weight_max = p_v * 1/len(k_max)

        # If all nodes are at t_max, reasign prob to sum up to 1
        if len(k_other) != 0:
            weight_other = (1 - p_v) * 1/len(k_other)
        else:
            weight_max += (1 - p_v)
            weight_other = 0
        
        # Generate a weights vector to bias the random sampling
        weights = []
        for i in range(len(k_all)):
            if i < len(k_max):
                weights.append(weight_max)
            else:
                weights.append(weight_other)

        idx = np.random.choice(len(k_all), p=weights)
        return k_all[idx]

    def get_min_path(self, delta):
        '''
            Searches the minimum cost node and gets the path to get to it
            - delta: minimum threshold required to obtain a solution not being the minimal cost but that is valid (eq. 5b)
        '''
        x_g = []
        min_node_cost_delta = 10000
        ## Find the possible nodes
        for node in self.nodes:
            node_cost = np.linalg.det(node.cov)
            if node_cost < delta:
                x_g.append(node)
                if node_cost < min_node_cost_delta:
                    min_node_cost_delta = node_cost

        ## Find the minimum node cost if there is any possible node. If not, repeat AIA tree construction
        if(len(x_g) > 1):
            min_cost_node = x_g[1]
        else:
            return -1
        for node in x_g:
            if node.cost < min_cost_node.cost and node.timestamp!=0:
                min_cost_node = node

        ## Get the path from that node to the root and reverse it
        path = [min_cost_node.reaching_motion]
        current_node = min_cost_node
        while len(current_node.parents) != 0:
            current_node = current_node.parents[0]
            path.append(current_node.reaching_motion)
        path.reverse()

        #print("Length: {}, Cost: {}".format(len(path), min_cost_node.cost))
        return path[1:], min_node_cost_delta