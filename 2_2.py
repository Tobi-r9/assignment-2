
import numpy as np
from Tree import Tree
from Tree import Node


def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.
    """
    print("Calculating the likelihood...")
    #get the root
    root = np.argwhere(np.isnan(tree_topology))[0][0]
    n = beta.shape[0]
    k = len(theta[root])
    print('n = {}, k = {}'.format(n,k))
    #S table filled with -1, to make sure empty entries do differ from probabilitiesd
    S = np.full(shape=(n,k),fill_value=-1.0)
    prob = [p for p in theta[root]]
    # TODO Add your code here
    l_array = np.array([s(root,i,beta,theta,tree_topology,S)*p 
                                    for i,p in enumerate(prob)])
    likelihood = np.sum(l_array)
   

    return likelihood

def s(x,i,beta,theta,tree_topology,S):
    '''this function does calculate p(Y_onx | x=i)
    params:beta, theta, tree_topology as defined above
    param: x (int) the current Node
    param: i (float) the value under consderration of node x
    return: float'''
    
    # check if the s(x,i) is already computed 
    if S[x,i] != -1.0:
        return S[x,i]
    # check if we at a leave and if i is the value assigned to this leave
    if not np.isnan(beta[x]):
        if i == beta[x]:
            S[x,i] = 1
            return 1
        else:
            S[x,i] = 0
            return 0
    else:
        # get children
        children = np.where(tree_topology == x)[0]
        s_xi = 1
        # for loop to be able to calc both cases (2 children or 1 child)
        for c in children:
            c = int(c)
            s_cj = np.array([s(c, j,beta,theta,tree_topology,S)*pij
                                     for j, pij in enumerate(theta[c][i])])
            s_xi *= sum(s_cj)
        S[x,i] = s_xi
        
        return S[x,i]



def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    #filename = "data/q2_2/q2_2_small_tree.pkl"
    #filename = "data/q2_2/q2_2_medium_tree.pkl"
    filename = "data/q2_2/q2_2_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.


    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(),
                                                 t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
