""" This file is created as a template for question 2.4 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)! Previously, other students used NetworkX package to work with trees and graphs, keep in mind.

    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you a single tree mixture (q2_4_tree_mixture).
    The mixture has 3 clusters, 5 nodes and 100 samples.
    We want you to run your EM algorithm and compare the real and inferred results
    in terms of Robinson-Foulds metric and the likelihoods.
    """
import numpy as np
import matplotlib.pyplot as plt
from em import EM
import dendropy
from Tree import Tree, TreeMixture
import random
    

# TODO change your calculation of r and pi to make sure that pi sums to one 

def save_results(loglikelihood, topology_array, theta_array, filename):

    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)


def step(exp):
    exp.likelihood = exp.update_likelihood()
    exp.r = exp.update_r()

    ## step 2
    exp.pi = exp.update_pi()

    ## step 3
    exp.q1, exp.q2 = exp.update_q()
    exp.I = exp.update_I()
    exp.G = exp.update_graph()
    ## step 4
    exp.MST = exp.compute_MST()
    ## step 5
    exp.topology_list = exp.update_topology()
    exp.theta_list = exp.update_theta()
    exp.loglikelihood.append(exp.compute_loglikelihood())

def sieving(num_clusters, samples):
    seeds = random.sample(range(1, 500), 100)
    models = []
    num_samples, num_nodes = samples.shape
    print('\nrun the first 10 timesteps for 100 sedds ...\n')
    for seed in seeds:
        print('\n seed = {}\n'.format(seed))
        exp = EM(num_nodes,num_clusters,samples, seed)
        init = {'topology': exp.topology_list, 'theta': exp.theta_list, 'pi':exp.pi}
        logl, _, __ = em_algorithm(seed, samples, num_clusters, 10, exp)
        models.append((exp,seed,logl))
    # take the 10 best values
    models.sort(key=lambda x: x[2], reverse=True)
    best_models = models[:10]
    #best_models = models
    print('the best 10 are chosen ... \n')
    print('Run the best 10 models until  convergence ...')
    for i, model in enumerate(best_models):
        print('\n seed = {}\n'.format(seed))
        _, seed, __ = model
        exp = EM(num_nodes,num_clusters,samples, seed)
        init = {'pi':exp.pi, 'theta':exp.theta_list, 'topology':exp.topology_list}
        logl, topology, theta = em_algorithm(seed, samples, num_clusters, 0, exp)
        best_models[i] = (logl, topology, theta, seed, init)
    best_models.sort(key=lambda x: x[0], reverse=True)
    result = best_models[0]
    return result


def em_algorithm(seed_val, samples, num_clusters, max_num_iter, exp):

    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    This is a suggested template. Feel free to code however you want.
    """
    epsilon = 10e-4 
    # Set the seed
    # TODO: Implement EM algorithm here.
    #save initialization
    
    exp.px = exp.prior() 
    if max_num_iter == 0:
        #run until convergence
        diff = 10
        while diff > epsilon:
            step(exp)
            if len(exp.loglikelihood) >= 2:
                diff = abs(exp.loglikelihood[-1]-exp.loglikelihood[-2])
    else:
        for _ in range(max_num_iter):
            step(exp)

    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    
    theta_list = exp.theta_list
    topology_list = exp.topology_list
    loglikelihood = exp.loglikelihood
    #
    return loglikelihood[1:], topology_list, theta_list


def create_samples(num_nodes, num_clusters, num_samples):
    seed = np.random.randint(0,200)
    tm = TreeMixture(num_clusters=num_clusters, num_nodes=num_nodes)
    tm.simulate_pi(seed_val=seed)
    tm.simulate_trees(seed_val=seed)
    tm.sample_mixtures(num_samples=num_samples,seed_val=seed)
    return tm.samples

def compare():
    tns = dendropy.TaxonNamespace()
    filename = "data/q2_4/q2_4_tree_mixture.pkl_tree_0_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t0 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 0: ", t0.as_string("newick"))
    t0.print_plot()

    filename = "data/q2_4/q2_4_tree_mixture.pkl_tree_1_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t1 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 1: ", t1.as_string("newick"))
    t1.print_plot()

    filename = "data/q2_4/q2_4_tree_mixture.pkl_tree_2_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t2 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 2: ", t2.as_string("newick"))
    t2.print_plot()

    print("\n3.2 Compare trees and print Robinson-Foulds (RF) distance:\n")

    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, t1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, t2))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, t2))

    print("\n4. Load Inferred Trees")
    filename = "data/q2_4/q2_4_result_em_topology.npy"  # This is the result you have.
    topology_list = np.load(filename)
    print(topology_list.shape)
    print(topology_list)

    rt0 = Tree()
    rt0.load_tree_from_direct_arrays(topology_list[0])
    rt0 = dendropy.Tree.get(data=rt0.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred Tree 0: ", rt0.as_string("newick"))
    rt0.print_plot()

    rt1 = Tree()
    rt1.load_tree_from_direct_arrays(topology_list[1])
    rt1 = dendropy.Tree.get(data=rt1.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred Tree 1: ", rt1.as_string("newick"))
    rt1.print_plot()

    rt2 = Tree()
    rt2.load_tree_from_direct_arrays(topology_list[2])
    rt2 = dendropy.Tree.get(data=rt2.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred Tree 2: ", rt2.as_string("newick"))
    rt2.print_plot()

    print("\n4.2 Compare trees and print Robinson-Foulds (RF) distance:\n")

    print("\tt0 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt2))

    print("\tt1 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt2))

    print("\tt2 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt2))

    print("\nInvestigate")

    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.find_missing_bipartitions(t0, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.false_positives_and_negatives(t0, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.find_missing_bipartitions(t0, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.false_positives_and_negatives(t0, rt1))

def main():
    print("Hello World!")
    print("This file demonstrates the flow of function templates of question 2.4.")

    sample_filename = "data/q2_4/q2_4_tree_mixture.pkl_samples.txt"
    output_filename = "data/q2_4/q2_4_result"
    real_values_filename = "data/q2_4/q2_4_tree_mixture.pkl"
    num_clusters = 3

    print("\n1. Load samples from txt file.\n")

    samples = create_samples(5,6,100)
    #samples = np.loadtxt(sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")

    #loglikelihood, topology_array, theta_array, init = em_algorithm(seed_val, samples, num_clusters=num_clusters)
    loglikelihood, topology_array, theta_array, seed, init = sieving(num_clusters, samples)
    np.save('.\controll\loglikelihood.npy',loglikelihood)
    print("\n3. Save, print and plot the results.\n")
    for key in init.keys():
        print('{} = {}'.format(key,init[key]))
    print('\n Seed = \n',seed)
    save_results(loglikelihood, topology_array, theta_array, output_filename)
    for i in range(num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    if real_values_filename != "":
        print("\n4. Retrieve real results and compare.\n")
        print("\tComparing the results with real values...")

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        # TODO: Do RF Comparison
        compare()

        print("\t4.2. Make the likelihood comparison.\n")
        # TODO: Do Likelihood Comparison


if __name__ == "__main__":
    main()
