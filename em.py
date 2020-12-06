import numpy as np
from Tree import TreeMixture
from Kruskal_v1 import Graph

class ExpectationMaximation:

    # not necessary, since this code is not scalable anymore 
    DIMENSION = 2
    '''To do: check that all variables are initialized with 0 
    and I do not a step of computation just by initializing'''

    def __init__(self,num_nodes,num_clusters,samples, seed_val, dim):
        self.N = len(samples)
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.samples = samples
        self.r = np.zeros((self.N,self.num_clusters))
        self.pi = create_pi(self.num_clusters, seed_val)
        self.topology_list, self.theta_list = init_tree(self.num_clusters, self.samples, seed_val)
        self.px = prior()
        self.posterior = update_posterior()
        self.r = update_r()
        self.q2 = np.zeros((self.num_clusters,self.num_nodes, self.num_nodes))
        self.q1 = np.zeros((self.num_clusters, self.num_nodes, DIMENSION))
        self.I = np.zeros((self.num_clusters,self.num_nodes,self.num_nodes))
        self.G = np.zeros(self.num_clusters)
        self.MST = np.zeros(self.num_clusters)

    def init_tree(self,num_clusters, samples, seed_val):
        '''initialize a tree as done in 2_4'''

        tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
        tm.simulate_pi(seed_val=seed_val)
        tm.simulate_trees(seed_val=seed_val)
        tm.sample_mixtures(num_samples=samples.shape[0], seed_val=seed_val)
        topology_list = []
        theta_list = []
        for i in range(num_clusters):
            topology_list.append(tm.clusters[i].get_topology_array())
            theta_list.append(tm.clusters[i].get_theta_array())

        topology_list = np.array(topology_list)
        theta_list = np.array(theta_list)
        return topology_list, theta_list

    def create_pi(self,num_clusters,seed_val=None):
        '''initialize pi'''

        if seed_val is not None:
            np.random.seed(seed_val)

        pi = np.random.rand(num_clusters)
        pi = pi / np.sum(pi)
        return pi

    def prior(self):
        '''compute the prior p(x) by number x_i appears / number of samples'''

        p = np.zeros(self.N)
        for ixd,sample in enumerate(self.samples):
            c = self.samples.count(sample)
            p[ixd] = c/self.N
        return p

    def update_posterior(self):
        '''update the posterior p(x|T_k, theta_k) using the actual version of T and theta and p(x)'''

        p = np.ones((self.num_clusters,self.N))
        for k in range(self.num_clusters):
            for idx, sample in enumerate(self.samples):
                for node,value in enumerate(sample):
                    parent = self.topology_list[k][node]
                    # if node = root 
                    if np.isnan(parent):
                        p[k,idx] *= self.theta_list[k][node][value]
                    # else
                    parent_value = sample[parent]
                    p[k,idx] *= self.theta_list[k][node][parent_value][value]
        return p 



    def update_r(self):
        '''update r according to the paper'''

        r = np.zeros((self.num_clustersn,self.N))
        for k in range(self.num_clusters):
            r[:,k] = self.posterior[k,:] * self.pi[k]/self.px
        
        return r

    def update_pi(self):
        '''update pi according to the paper'''

        for k in range(self.num_clusters):
            self.pi[k] = np.sum(self.r[:,k])/self.N
            
    
    def update_q(self):
        '''this function does update both q matrices (q(xa) and q(xa,xb)) according to the paper
        it goes through all nodes twice and updates q(xa) in the second for loop and q(xa,xb)
        in the third. DIMENSION,DIMENSION as extra dimension for q2 represent the possible value combinations
        and they were used instead of just on to make the implementation for I easier'''

        q2 = np.zeros((self.num_clusters, self.num_nodes, self.num_nodes, DIMENSION, DIMENSION))
        q1 = np.zeros((self.num_clusters, self.num_nodes, DIMENSION))
        for k in range(self.num_clusters):
            denominator = np.sum(self.r[:,k])
            for xi in len(self.num_nodes):
                n1 = sum([r[n,k] for n in range(self.N) if samples[n][xi] == 0])
                n2 = sum([r[n,k] for n in range(self.N) if samples[n][xi] == 1])
                q1[k,xi,0] = n1/(n1+n2)
                q1[k,xi,1] = n2/(n1+n2)

                for xj in len(self.num_nodes):
                    aa = [r[n,k] for n in range(self.N) if (samples[n][xi] == 0 and samples[n][xj] == 0)]
                    ab = [r[n,k] for n in range(self.N) if (samples[n][xi] == 0 and samples[n][xj] == 1)]
                    bb = [r[n,k] for n in range(self.N) if (samples[n][xi] == 1 and samples[n][xj] == 1)]
                    ba = [r[n,k] for n in range(self.N) if (samples[n][xi] == 1 and samples[n][xj] == 0)]
                    q[k,xi,xj,0,0] = np.sum(aa)/denominator
                    q[k,xi,xj,0,1] = np.sum(ab)/denominator
                    q[k,xi,xj,1,1] = np.sum(bb)/denominator
                    q[k,xi,xj,1,0] = np.sum(ba)/denominator
        
        return q1, q2

    def update_I(self):
        '''this function does update I (k,n,n), it goes through all clusters and nodes (2 times) and finally 
        through the possible value combinations. It then implements I_{q^k} from the paper'''
        
        I = np.zeros((self.num_clusters,self.num_nodes,self.num_nodes))
        for k in range(self.num_clusters):
            for x1 in range(self.num_nodes):
                for x2 in range(self.num_nodes):
                    #go through ab (x1 = a, x2 = b)
                    for a in range(DIMENSION):
                        for b in range(DIMENSION):
                            if q2[k,x1,x2,a,b] != 0:
                                I[k,x1,x2] += q2[k,x1,x2,a,b]*np.log(q2[k,x1,x2,a,b]/(q1[k,x1,a]*q1[k,x2,b]))
                            else:
                                I[k,x1,x2] += 0

        return I


    def update_graph(self):
        '''This function does create k fully connected Graphs whith the edges from I
        Note that we connect two nodes twice (e.g we have a edges (n1, n2) and (n2, n1))'''

        G = np.array([k for k in range(self.num_clusters)])
        for k in range(self.num_clusters):
            g = Graph(self.num_nodes)
            for node1 in range(self.num_nodes):
                for node2 in range(len(self.num_nodes)):
                    w = self.I[k,node1,node2]
                    g.add(node1,node2)
            
            G[k] = g

        return G

    def compute_MST(self):
        '''compute the k maximum spanning trees
        return: array shape (k,)'''

        mst = np.zeros(self.num_clusters)
        for k in range(self.num_clusters):
            mst[k] = self.G[k].maximum_spanning_tree()

        return mst

    def update_topology(self):
        temp_mst = self.MST.copy()
        root = [[] for k in range(self.num_clusters)]
        #get root 0
        for k in range(self.num_clusters):
            for edge in self.MST[k]:
                if 0 in edge:
                    root.append(edge)
                    temp_mst[k].remove(edge)
        

    def update_theta(self):

        temp = self.theta_list
        
        for k in range(self.num_clusters):
            for node, entry in enumerate(temp[k,:,:]):
                parent = self.topology_list[k][node]
                #root
                if np.isnan(parent):
                    self.theta_list[k][node] = self.q1[k][node]
                else:
                    for a in range(2):
                        for b in range(2):
                            self.list_theta[k][node][b][a] = self.q2[k][node][parent][a][b]/self.q1[k][parent][b]
                

