#%%
import numpy as np
import pandas as pd


def deteriorate(T, N):

    all_replacement_cost = []

    for n in range(N):
        prob_of_breaking = 0.1
        replacement_cost = 0
        for t in range(T):
            if np.random.uniform() < prob_of_breaking:
                replacement_cost += 1
                prob_of_breaking = 0.1
            else:
                prob_of_breaking += 0.01

        all_replacement_cost.append(replacement_cost/T)
        
    print(np.mean(all_replacement_cost))


def solve_Poisson(alpha = None, policy_iteration=False):

    # set up probability matrix
    P = np.zeros((91,91))
    P[0,1] = 0.9
    for i in range(1,91):
        P[i,0] = P[i-1,0] + 0.01
        if i < 90:
            P[i,i+1] = P[i-1,i] - 0.01

    r = -(0.1 + P[:,0])
    r[0] = -0.1

    if policy_iteration:
        # change p(y|x) to p(y|x,alpha(x))
        location_of_1 = np.where(alpha==1)[0][0]
        P[location_of_1:, location_of_1+1:] = 0

        # change r(x) to r(x, alpha)
        r[location_of_1] = -0.5
        r[location_of_1+1:] = 0

    A = np.zeros((91,91))
    A[:, :-1] = np.identity(91)[:,1:]

    # fill in coefficient for phi at last column
    A[:,-1] = 1

    # fill in probabilities
    A[:,:-1] -= P[:,1:]

    # solve system of equations
    x = np.linalg.solve(A, r)
    phi = x[-1]

    return phi


def policy_iteration():
    """
    Performs policy iteration to find optimal policy
    """
    alpha_tilde = np.identity(91) # columns are different policies

    # get (V, phi) for all alphas:
    all_phis = np.array([solve_Poisson(alpha=alpha, policy_iteration=True) for alpha in alpha_tilde[:,]])
    best_alpha = np.argmax(all_phis)
    print(f'Policy iteration:\n     Best time to replace: {best_alpha+1}.\n     φ at replacement: {all_phis[best_alpha]}.')

def value_iteration(iteration):
    # set up probability matrix
    P = np.zeros((91,91))
    P[0,1] = 0.9
    for i in range(1,91):
        P[i,0] = P[i-1,0] + 0.01
        if i < 90:
            P[i,i+1] = P[i-1,i] - 0.01

    r = -(0.1 + P[:,0])
    r[0] = -0.1

    #Start Policy Iteration
    V = np.zeros((91))
    newV = np.zeros((91))
    for j in range(iteration):
        for i in range(90):
            newV[i] = max(r[i] + P[i,0]*V[0] + P[i,i+1]*V[i+1], -0.5 + V[0])
        newV[90] = -0.5 + V[0]
        V = newV

    print("Convergence at Step: ",len(list(set(V))))
    # print(V)

if __name__ == '__main__':

    T = 10000
    N = 1000
    iteration = 10000

    # deteriorate(T, N)
    # solve_Poisson()
    # policy_iteration()
    value_iteration(iteration)
# %%
