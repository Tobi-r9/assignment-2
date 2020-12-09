import numpy as np
from Tree import TreeMixture
from Kruskal_v1 import Graph

class EM:

    # not necessary, since this code is not scalable anymore 
    DIMENSION = 2
    '''To do: check that all variables are initialized with 0 
    and I do not a step of computation just by initializing'''

    def __init__(self,num_nodes,num_clusters,samples, seed_val):
        self.N = len(samples)
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.samples = samples
        self.r = np.zeros((self.num_clusters,self.N))
        self.topology_list, self.theta_list, self.pi = self.init_tree(self.num_clusters, self.samples, seed_val)
        self.px = np.zeros(self.N)
        self.likelihood = np.zeros((self.num_clusters,self.N))
        self.q2 = np.zeros((self.num_clusters,self.num_nodes, self.num_nodes,EM.DIMENSION, EM.DIMENSION))
        self.q1 = np.zeros((self.num_clusters, self.num_nodes, EM.DIMENSION))
        self.I = np.zeros((self.num_clusters,self.num_nodes,self.num_nodes))
        self.G = np.zeros(self.num_clusters)
        self.MST = np.zeros(self.num_clusters)
        self.loglikelihood = []


    def init_tree(self,num_clusters, samples, seed_val):
        '''initialize a tree as done in 2_4'''

        tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
        tm.simulate_pi(seed_val=seed_val)
        tm.simulate_trees(seed_val=seed_val)
        tm.sample_mixtures(num_samples=samples.shape[0], seed_val=seed_val)
        pi = tm.pi
        topology_list = []
        theta_list = []
        for i in range(num_clusters):
            topology_list.append(tm.clusters[i].get_topology_array())
            theta_list.append(tm.clusters[i].get_theta_array())

        topology_list = np.array(topology_list)
        theta_list = np.array(theta_list)
        return topology_list, theta_list, pi


    def prior(self):
        '''compute the prior p(x) by number x_i appears / number of samples
            return: p, np.array, shape = (N,)'''
        

        p = np.zeros(self.N)
        temp = [list(sample) for sample in self.samples]
        for ixd,sample in enumerate(temp):
            c = temp.count(sample)
            p[ixd] = c/self.N
        return p

    def update_likelihood(self):
        '''update the likelihood p(x|T_k, theta_k) using the actual version of T and theta and p(x)
            return p, np.array, shape=(k,N)'''

        p = np.ones((self.num_clusters,self.N))
        for k in range(self.num_clusters):
            for idx, sample in enumerate(self.samples):
                for node,value in enumerate(sample):
                    parent = self.topology_list[k][node]
                    
                    # if node = root 
                    if np.isnan(parent):
                        p[k,idx] *= self.theta_list[k][node][value]
                    else:
                        parent = int(parent)
                        parent_value = sample[parent]
                        p[k,idx] *= self.theta_list[k][node][parent_value][value]
        return p 



    def update_r(self):
        '''update r according to the paper
            return r, np.array, shape = (k,N)'''

        r = np.zeros((self.num_clusters,self.N))
        for k in range(self.num_clusters):
            r[k,:] = (self.likelihood[k,:] * self.pi[k])/self.px
        #normalize r
        for n in range(self.N):
            r[:,n] = r[:,n]/sum(r[:,n])
        
        return r

    def update_pi(self):
        '''update pi according to the paper
            returns: pi, np.array, shape=(k,)'''
        pi = np.zeros(self.num_clusters)
        for k in range(self.num_clusters):
            pi[k] = np.sum(self.r[k,:])/self.N
        pi = pi/sum(pi)
        return pi
    
    def update_q(self):
        '''this function does update both q matrices (q(xa) and q(xa,xb)) according to the paper
        it goes through all nodes twice and updates q(xa) in the second for loop and q(xa,xb)
        in the third. DIMENSION,DIMENSION as extra dimension for q2 represent the possible value combinations
        and they were used instead of just on to make the implementation for I easier'''

        q2 = np.zeros((self.num_clusters, self.num_nodes, self.num_nodes, EM.DIMENSION, EM.DIMENSION))
        q1 = np.zeros((self.num_clusters, self.num_nodes, EM.DIMENSION))
        for k in range(self.num_clusters):
            denominator = np.sum(self.r[k,:])
            for xi in range(self.num_nodes):
                n1 = sum([self.r[k,n] for n in range(self.N) if self.samples[n][xi] == 0])
                n2 = sum([self.r[k,n] for n in range(self.N) if self.samples[n][xi] == 1])
                q1[k,xi,0] = n1/(n1+n2)
                q1[k,xi,1] = n2/(n1+n2)

                for xj in range(self.num_nodes):
                    aa = [self.r[k,n] for n in range(self.N) if (self.samples[n][xi] == 0 and self.samples[n][xj] == 0)]
                    ab = [self.r[k,n] for n in range(self.N) if (self.samples[n][xi] == 0 and self.samples[n][xj] == 1)]
                    bb = [self.r[k,n] for n in range(self.N) if (self.samples[n][xi] == 1 and self.samples[n][xj] == 1)]
                    ba = [self.r[k,n] for n in range(self.N) if (self.samples[n][xi] == 1 and self.samples[n][xj] == 0)]
                    q2[k,xi,xj,0,0] = np.sum(aa)/denominator
                    q2[k,xi,xj,0,1] = np.sum(ab)/denominator
                    q2[k,xi,xj,1,1] = np.sum(bb)/denominator
                    q2[k,xi,xj,1,0] = np.sum(ba)/denominator
        
        return q1, q2

    def update_I(self):
        '''this function does update I (k,n,n), it goes through all clusters and nodes (2 times) and finally 
        through the possible value combinations. It then implements I_{q^k} from the paper'''
        
        I = np.zeros((self.num_clusters,self.num_nodes,self.num_nodes))
        for k in range(self.num_clusters):
            for x1 in range(self.num_nodes):
                for x2 in range(self.num_nodes):
                    #go through ab (x1 = a, x2 = b)
                    for a in range(EM.DIMENSION):
                        for b in range(EM.DIMENSION):
                            if self.q2[k,x1,x2,a,b] != 0:
                                I[k,x1,x2] += self.q2[k,x1,x2,a,b]*np.log(self.q2[k,x1,x2,a,b]/(self.q1[k,x1,a]*self.q1[k,x2,b]))
                            else:
                                I[k,x1,x2] += 0

        return I


    def update_graph(self):
        '''This function does create k fully connected Graphs whith the edges from I
        Note that we connect two nodes twice (e.g we have a edges (n1, n2) and (n2, n1))'''

        G = []
        for k in range(self.num_clusters):
            g = Graph(self.num_nodes)
            for node1 in range(self.num_nodes):
                for node2 in range(self.num_nodes):
                    w = self.I[k,node1,node2]
                    g.addEdge(node1,node2,w)
            G.append(g) 

        return G

    def compute_MST(self):
        '''compute the k maximum spanning trees
        return: mst, array, shape=(k,)'''

        mst = []
        for k in range(self.num_clusters):
            mst.append(self.G[k].maximum_spanning_tree())
            
        return mst

    def update_topology(self):
        '''creates a topology from the maximum spanning tree 
        return: topology, list,  shape=(num_clusters, num_nodes)''' 

        topology = [[i for i in range(self.num_nodes)] for k in range(self.num_clusters)]
        for k in range(self.num_clusters):
            temp_tree = self.MST[k]
            #get root and children
            topology[k][0] = np.nan
            parents = [node for node in self.MST[k] if (node[0]==0 or node[1] == 0)]
            parent = []
            for node in  parents:
                temp_tree.remove(node)
                node.remove(0)
                topology[k][node[0]] = 0
                parent.append(node[0])

            #find parent node pair for all other nodes
            while temp_tree != []:
                parents = {}
                temp_parent = []
                #get children of parents
                for p in parent:
                    parents[p] = [node for node in temp_tree if (node[0]==p or node[1] == p)]
                
                # for every child in parents update the topology
                for p in parents.keys():
                    for node in parents[p]:
                        temp_tree.remove(node) #remove the parent so the firs entry has to be the child 
                        node.remove(p)
                        topology[k][node[0]] = p
                        temp_parent.append(node[0])
                parent = temp_parent
        return topology
                    

    def update_theta(self):
        '''updates self.theta directly''' 

        theta = self.theta_list.copy()
        
        for k in range(self.num_clusters):
            for node, entry in enumerate(self.theta_list[k,:,:]):
                parent = self.topology_list[k][node]
                #root
                if np.isnan(parent):
                    theta[k][node] = self.q1[k][node]
                else:
                    for a in range(2):
                        for b in range(2):
                            theta[k][node][b][a] = self.q2[k][node][parent][a][b]/self.q1[k][parent][b]
        return theta
                

    def compute_loglikelihood(self):
        '''compute the loglikelihood given the data,the current estimate of the likelihoop
         and the curen parameters
         returns: sum(loglikelihood), float'''
        loglikelihood = []
        for n in range(self.N):
            temp = np.log(np.sum(self.pi*self.likelihood[:,n]))
            loglikelihood.append(temp)
    
        return sum(loglikelihood)

