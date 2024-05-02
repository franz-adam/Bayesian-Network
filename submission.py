import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy͏󠄂͏️͏󠄌͏󠄎͏󠄎͏󠄊͏󠄁
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import random
import numpy

#You are not allowed to use following set of modules from 'pgmpy' Library.͏󠄂͏️͏󠄌͏󠄎͏󠄎͏󠄊͏󠄁
#
# pgmpy.sampling.*͏󠄂͏️͏󠄌͏󠄎͏󠄎͏󠄊͏󠄁
# pgmpy.factors.*͏󠄂͏️͏󠄌͏󠄎͏󠄎͏󠄊͏󠄁
# pgmpy.estimators.*͏󠄂͏️͏󠄌͏󠄎͏󠄎͏󠄊͏󠄁

def make_security_system_net():
    """Create a Bayes Net representation of the above security system problem. 
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    BayesNet = BayesianNetwork()
    # TODO: finish this function͏󠄂͏️͏󠄌͏󠄎͏󠄎͏󠄊͏󠄁
    BayesNet.add_node("H")
    BayesNet.add_node("C")
    BayesNet.add_node("M")
    BayesNet.add_node("B")
    BayesNet.add_node("Q")
    BayesNet.add_node("K")
    BayesNet.add_node("D")
    
    BayesNet.add_edge("H","Q")
    BayesNet.add_edge("C","Q")
    
    BayesNet.add_edge("B","K")
    BayesNet.add_edge("M","K")
    
    BayesNet.add_edge("Q","D")
    BayesNet.add_edge("K","D")
    
    #raise NotImplementedError
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the security system.
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    # Setting probabilities for H,C,M,B
    cpd_h = TabularCPD('H', 2, values=[[0.5], [0.5]])
    cpd_c = TabularCPD('C', 2, values=[[0.7], [0.3]])
    cpd_m = TabularCPD('M', 2, values=[[0.2], [0.8]])
    cpd_b = TabularCPD('B', 2, values=[[0.5], [0.5]])
    
    # Setting probabilities for Q given H and C
    cpd_qhc = TabularCPD('Q', 2, values=[[0.95, 0.75, 0.45, 0.1], \
                    [0.05, 0.25, 0.55, 0.9]], evidence=['H', 'C'], evidence_card=[2, 2])
    
    print("qhc cpd: ",cpd_qhc)
    
    # Setting probabilities for K given B and M
    cpd_kbm = TabularCPD('K', 2, values=[[0.25, 0.05, 0.99, 0.85], \
                    [0.75, 0.95, 0.01, 0.15]], evidence=['B', 'M'], evidence_card=[2, 2])
    
    print("kbm cpd: ",cpd_kbm)
    
    # Setting probabilities for D given Q and K
    cpd_dqk = TabularCPD('D', 2, values=[[0.98, 0.65, 0.4, 0.01], \
                    [0.02, 0.35, 0.6, 0.99]], evidence=['Q', 'K'], evidence_card=[2, 2])
    
    print("dqk cpd: ",cpd_dqk)
    
    #raise NotImplementedError    
    
    bayes_net.add_cpds(cpd_h, cpd_c, cpd_m, cpd_b, cpd_qhc, cpd_kbm, cpd_dqk)
    
    return bayes_net


def get_marginal_double0(bayes_net):
    """Calculate the marginal probability that Double-0 gets compromised.
    """
    # Calculate probability
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], joint=False)
    double0_prob = marginal_prob['D'].values
    
    return double0_prob[1]


def get_conditional_double0_given_no_contra(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down.
    """
    # Calculate marginal probability of D given not C
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':0}, joint=False)
    double0_prob = conditional_prob['D'].values
    
    return double0_prob[1]


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down and Bond is reassigned to protect M.
    """
    # Calculate marginal probability of D given not C and B
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':0, 'B':1}, joint=False)
    double0_prob = conditional_prob['D'].values
    
    return double0_prob[1]

def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianNetwork()
    # Set up
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")
    
    BayesNet.add_edge("A","AvB")
    BayesNet.add_edge("B","AvB")
    
    BayesNet.add_edge("B","BvC")
    BayesNet.add_edge("C","BvC")
    
    BayesNet.add_edge("A","CvA")
    BayesNet.add_edge("C","CvA")
    
    # Setting probabilities for A, B and C
    cpd_a = TabularCPD('A', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_b = TabularCPD('B', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_c = TabularCPD('C', 4, values=[[0.15], [0.45], [0.30], [0.10]])
  
    
    # Setting probabilities for AvB, BvC, CvA given difference in skill level
    cpd_avb = TabularCPD('AvB', 3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1], \
                                           [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1], \
                                           [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]],
                         evidence=['A', 'B'], evidence_card=[4, 4])
    
    #print("AvB cpd: ",cpd_avb)
    
    cpd_bvc = TabularCPD('BvC', 3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1], \
                                           [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1], \
                                           [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]],
                         evidence=['B', 'C'], evidence_card=[4, 4])
    
    #print("BcV cpd: ",cpd_bvc)
    
    cpd_cva = TabularCPD('CvA', 3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1], \
                                           [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1], \
                                           [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]],
                         evidence=['C', 'A'], evidence_card=[4, 4])
    
    #print("CvA cpd: ",cpd_cva)
    
    # Adding probabilities to our Network
    BayesNet.add_cpds(cpd_a, cpd_b, cpd_c, cpd_avb, cpd_bvc, cpd_cva)
      
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    
    # Calculate posterior probability of BvC given A beats B and A draws with C
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'],evidence={'AvB':0, 'CvA':2}, joint=False)
    posterior = conditional_prob['BvC'].values
    
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    # If initial state is None or empty list, then return randomly selected state over uniform distribution
    if initial_state == None or len(initial_state) == 0:
        initial_state = (
            random.randint(0, 3), # A
            random.randint(0, 3), # B
            random.randint(0, 3), # C
            0,  # Keep evidence fixed, A beats B
            random.choice([0, 1, 2]),  # BvC
            2  # Keep evidence fixed, A draws with C
        )
    
    sample = list(initial_state)
    index_to_update = random.choice([0, 1, 2, 4])
    skill_levels = [0,1,2,3]
    skill_probs = [0.15, 0.45, 0.30, 0.1]
    
    if index_to_update == 0: # Case A gets selected 
        cpd_a = []
        for i in range(0,4):
            skill_b = sample[1]
            skill_c = sample[2]
            match_avb = sample[3]
            match_cva = sample[5]
            
            cpd_avb = bayes_net.get_cpds('AvB').values.flatten()
            cpd_cva = bayes_net.get_cpds('CvA').values.flatten()
        
            index_avb = int(i * 4 + skill_b + 16 * match_avb)
            index_cva = int(skill_c * 4 + i + 16 * match_cva)
            
            #print("Index AvB for a: ", i, " and b: ", skill_b, " : ", index_avb)
            #print("Index CvA for c: ", skill_c, " and a: ", i, " : ", index_cva)
            
            cpd_a.append(cpd_avb[index_avb] * cpd_cva[index_cva] * skill_probs[i])
        
        cpd_a = numpy.array(cpd_a) # Normalize values
        cpd_a_norm = cpd_a / cpd_a.sum()
        cpd_a_norm = list(cpd_a_norm)
        
        sample[index_to_update] = random.choices(skill_levels, weights=cpd_a_norm, k=1)[0] # Resample based on new CPD
        
    elif index_to_update == 1: # Case B gets selected
        cpd_b = []
        for i in range(0,4):
            skill_a = sample[0]
            skill_c = sample[2]
            match_avb = sample[3]
            match_bvc = sample[4]
            
            cpd_avb = bayes_net.get_cpds('AvB').values.flatten()
            cpd_bvc = bayes_net.get_cpds('BvC').values.flatten()
            
            index_avb = int(skill_a * 4 + i + 16 * match_avb)
            index_bvc = int(i * 4 + skill_c + 16 * match_bvc)
            
            #print("Index AvB for a: ", skill_a, " and b: ", i, " : ", index_avb)
            #print("Index BvC for b: ", i, " and c: ", skill_c, " : ", index_bvc)
            
            cpd_b.append(cpd_avb[index_avb] * cpd_bvc[index_bvc] * skill_probs[i])
        
        cpd_b = numpy.array(cpd_b) # Normalize values
        cpd_b_norm = cpd_b / cpd_b.sum()
        cpd_b_norm = list(cpd_b_norm)
        
        sample[index_to_update] = random.choices(skill_levels, weights=cpd_b_norm, k=1)[0] # Resample based on new CPD
            
    elif index_to_update == 2: # Case C gets selected
        cpd_c = []
        for i in range(0,4):
            skill_a = sample[0]
            skill_b = sample[1]
            match_bvc = sample[4]
            match_cva = sample[5]
            
            cpd_bvc = bayes_net.get_cpds('BvC').values.flatten()
            cpd_cva = bayes_net.get_cpds('CvA').values.flatten()
        
            index_bvc = int(skill_b * 4 + i + 16 * match_bvc)
            index_cva = int(i * 4 + skill_a + 16 * match_cva)
            
            #print("Index BvC for b: ", skill_b, " and c: ", i, " : ", index_bvc)
            #print("Index CvA for c: ", i, " and a: ", skill_a, " : ", index_cva)
            
            cpd_c.append(cpd_bvc[index_bvc] * cpd_cva[index_cva] * skill_probs[i])
        
        cpd_c = numpy.array(cpd_c) # Normalize values
        cpd_c_norm = cpd_c / cpd_c.sum()
        cpd_c_norm = list(cpd_c_norm)
        
        sample[index_to_update] = random.choices(skill_levels, weights=cpd_c_norm, k=1)[0] # Resample based on new CPD
        
    else: # Case BvC gets selected
        cpd_bvc = bayes_net.get_cpds('BvC').values.flatten()
        skill_b = sample[1]
        skill_c = sample[2]

        index = int(skill_b * 4 + skill_c) # Calculate the index for BvC's CPD based on B and C value
        positions = [index, index + 16, index + 32] # Get index positions for win (B), loss (B), draw
        
        #print("Index positions for b: ", skill_b, " and c: ", skill_c, " : ", positions)
        
        probs = [cpd_bvc[pos] for pos in positions] # Get probabilities given B and C
        
        sample[index_to_update] = random.choices([0, 1, 2], weights=probs, k=1)[0]

    return tuple(sample)


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    if initial_state == None or len(initial_state) == 0:
        initial_state = (
            random.randint(0, 3), # A
            random.randint(0, 3), # B
            random.randint(0, 3), # C
            0,  # Keep evidence fixed, A beats B
            random.choice([0, 1, 2]),  # BvC
            2  # Keep evidence fixed, A draws with C
        )
        
    sample = list(initial_state)
    
    skill_probs = [0.15, 0.45, 0.30, 0.1]
    cand_state = [random.randint(0, 3), random.randint(0, 3), random.randint(0, 3), 0, random.choice([0, 1, 2]), 2]  
    
    # Get target distribution for candidate state 
    a, b, c = cand_state[0], cand_state[1], cand_state[2]
    match_avb, match_bvc, match_cva = 0, cand_state[4], 2
    cpd_avb = bayes_net.get_cpds('AvB').values.flatten()
    cpd_bvc = bayes_net.get_cpds('BvC').values.flatten()
    cpd_cva = bayes_net.get_cpds('CvA').values.flatten()
    
    p_a, p_b, p_c = skill_probs[a], skill_probs[b], skill_probs[c]
    p_avb = cpd_avb[a*4 + b + match_avb * 16]
    p_bvc = cpd_bvc[b*4 + c + match_bvc * 16]
    p_cva = cpd_cva[c*4 + a + match_cva * 16]
    
    target_dist_pi_prime = (p_a * p_b * p_c * p_avb * p_bvc * p_cva)
    
    # Get target distribution for initial state 
    a, b, c = sample[0], sample[1], sample[2]
    match_avb, match_bvc, match_cva = 0, sample[4], 2
    
    p_a, p_b, p_c = skill_probs[a], skill_probs[b], skill_probs[c]
    p_avb = cpd_avb[a*4 + b + match_avb * 16]
    p_bvc = cpd_bvc[b*4 + c + match_bvc * 16]
    p_cva = cpd_cva[c*4 + a + match_cva * 16]
    
    target_dist_pi = (p_a * p_b * p_c * p_avb * p_bvc * p_cva)
    
    alpha = min(1, target_dist_pi_prime/target_dist_pi)
    
    sample = random.choices([sample, cand_state], weights=[1-alpha, alpha], k=1)[0] #??
    
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = -1
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    N = 50
    delta = 0.00005
    max_samples = 1000000
    cont_sample = 500000
    E_bvc, count_0, count_1, count_2, n_counter, dist = [[0, 0, 0]], 0, 0, 0, 0, []
    state_0 = initial_state
    
    for i in range(1, max_samples):
        initial_state = Gibbs_sampler(bayes_net, initial_state)
        
        if initial_state[4] == 0:
            count_0 += 1
        elif initial_state[4] == 1:
            count_1 += 1
        else: count_2 += 1
        
        sum_ = count_0 + count_1 + count_2
        E_bvc.append([count_0/sum_, count_1/sum_, count_2/sum_])
        
        diff = [abs(current - previous) for current, previous in zip(E_bvc[i], E_bvc[i-1])]
        diff = sum(diff)
        
        if diff <= delta:
            n_counter += 1
        else: n_counter = 0
            
        if n_counter >= N:
            Gibbs_count = i
            for _ in range(cont_sample):
                initial_state = Gibbs_sampler(bayes_net, initial_state)
                dist.append(initial_state[4])
            break
    
    Gibbs_convergence = [dist.count(0)/cont_sample, dist.count(1)/cont_sample, dist.count(2)/cont_sample]
    
    #print("Gibbs count: ", Gibbs_count)
    #print("Gibbs convergence: ", Gibbs_convergence)
    
    ########## MH Sampler
    E_bvc, count_0, count_1, count_2, n_counter, dist = [[0, 0, 0]], 0, 0, 0, 0, []
    initial_state = state_0
    
    for i in range(1, max_samples):
        previous_state = initial_state
        initial_state = MH_sampler(bayes_net, initial_state)
        
        if previous_state == initial_state:
            MH_rejection_count += 1
        
        if initial_state[4] == 0:
            count_0 += 1
        elif initial_state[4] == 1:
            count_1 += 1
        else: count_2 += 1
        
        sum_ = count_0 + count_1 + count_2
        E_bvc.append([count_0/sum_, count_1/sum_, count_2/sum_])
        
        diff = [abs(current - previous) for current, previous in zip(E_bvc[i], E_bvc[i-1])]
        diff = sum(diff)
        
        if diff <= delta:
            n_counter += 1
        else: n_counter = 0
            
        if n_counter >= N:
            MH_count = i
            for _ in range(cont_sample):
                initial_state = MH_sampler(bayes_net, initial_state)
                dist.append(initial_state[4])
            break
    
    MH_convergence = [dist.count(0)/cont_sample, dist.count(1)/cont_sample, dist.count(2)/cont_sample]
    
    #print("MH count: ", MH_count)
    #print("MH convergence: ", MH_convergence)
    
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    
    #raise NotImplementedError
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.1
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function͏󠄂͏️͏󠄌͏󠄎͏󠄎͏󠄊͏󠄁
    return "Franz Adam"
    #raise NotImplementedError

    
################################################### TESTING - DO NOT DISTURB
#net = make_security_system_net()
"""

net = get_game_network()
initial_state = []
dist = []

running_means = []
running_variances = []
mean = 0
M2 = 0  # For Welford's algorithm for variance

#for i in range(150000):
#    initial_state = Gibbs_sampler(net, initial_state)
#    dist.append(initial_state[4])  # Focus on 'BvC' match outcome for simplicity

#dist = dist
count_0 = dist.count(0)
count_1 = dist.count(1)
count_2 = dist.count(2)
count_sum = count_0 + count_1 + count_2

print("Posterior: ", calculate_posterior(net))

percent_0 = (count_0 / count_sum) * 100
percent_1 = (count_1 / count_sum) * 100
percent_2 = (count_2 / count_sum) * 100

# Print the percentages
#print(f"Gibbs Percentage of 0: {percent_0:.2f}%")
#print(f"Gibbs Percentage of 1: {percent_1:.2f}%")
#print(f"Gibbs Percentage of 2: {percent_2:.2f}%")

initial_state, dist = [], []

for i in range(150000):
    initial_state = MH_sampler(net, initial_state)
    dist.append(initial_state[4])
    #print("Hello", i)
    
count_0 = dist.count(0)
count_1 = dist.count(1)
count_2 = dist.count(2)
count_sum = count_0 + count_1 + count_2

percent_0 = (count_0 / count_sum) * 100
percent_1 = (count_1 / count_sum) * 100
percent_2 = (count_2 / count_sum) * 100

# Print the percentages
#print(f"MH Percentage of 0: {percent_0:.2f}%")
#print(f"MH Percentage of 1: {percent_1:.2f}%")
#print(f"MH Percentage of 2: {percent_2:.2f}%")

initial_state, gibbs_count, mh_count = [], [], []
for i in range(10):
    print("------ ITERATION : ", i, " ---------")
    _, _, gc, mhc, _ = compare_sampling(net, initial_state)
    gibbs_count.append(gc)
    mh_count.append(mhc)
    print("\n")
    
print("Mean Gibbs: ", numpy.mean(gibbs_count))
print("Mean MH: ", numpy.mean(mh_count))
"""