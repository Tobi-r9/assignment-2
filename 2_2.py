""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_2_small_tree, q_2_2_medium_tree, q_2_2_large_tree).
    Each tree has 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.

    Note: The alphabet "K" is K={0,1,2,3,4}.
"""

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
    l_array = np.array([s(root,i,beta,theta,tree_topology,S)*p for i,p in enumerate(prob)])
    likelihood = np.sum(l_array)
    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    
    # End: Example Code Segment

    return likelihood

def s(x,i,beta,theta,tree_topology,S):
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
        c1,c2 =  np.where(tree_topology == x)[0]
        c1,c2 = int(c1),int(c2) #to make sure no problems arise 
        #compute sum_j s(c,j)*p(c=j|x=i) for both children c in (c1,c2)
        s1 = np.array([s(c1, j,beta,theta,tree_topology,S)*pij for j, pij in enumerate(theta[c1][i])])
        s2 = np.array([s(c2, j,beta,theta,tree_topology,S)*pij for j, pij in enumerate(theta[c2][i])])
        S[x,i] = np.sum(s1)*np.sum(s2)
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
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
