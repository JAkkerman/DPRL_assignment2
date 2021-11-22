import numpy as np
import pandas as pd


def stationary_distribution():
    """
    Computes the stationary distribution to get the average cost
    """

    # set up probability matrix P:
    P = np.zeros((91,91))
    P[0,1] = 0.9
    P[0,0] = 0.1
    for i in range(1,91):
        P[i,0] = P[i-1,0] + 0.01
        if i < 90:
            P[i,i+1] = P[i-1,i] - 0.01

    # set up A matrix and b vector
    A = np.zeros((92,91))
    A[:-1,:] = P.T - np.identity(91)
    A[-1,:] = 1

    b = np.zeros(92)
    b[-1] = 1

    np.set_printoptions(suppress=True)
    x = np.linalg.solve(A[1:,:], b[1:])
    print('Stationary distribution:\n', x)


def solve_Poisson(alpha = None, policy_iteration=False):
    """
    Solves the Poisson equation to get the average cost
    """

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
    """
    Performs value iteration to get the optimal policy
    """
    # set up probability matrix
    P = np.zeros((91,91))
    P[0,1] = 0.9
    for i in range(1,91):
        P[i,0] = P[i-1,0] + 0.01
        if i < 90:
            P[i,i+1] = P[i-1,i] - 0.01

    P[:,0] += 0.1

    r = -P[:,0]
    r[0] = -0.1

    #Start Policy Iteration
    V = np.zeros((91))
    newV = np.zeros((91))
    newV[0] = 1

    for j in range(iteration):
        for i in range(90):
            newV[i] = max(r[i] + P[i,0]*V[0] + P[i,i+1]*V[i+1], -0.5 + V[0])
        newV[90] = -0.5 + V[0]

        V = newV.copy()

    print("Value iteration:\n     Best time to replace: ", len(list(set(V))))

if __name__ == '__main__':

    T = 10000
    N = 1000
    iterations = 100000

    stationary_distribution()
    phi = solve_Poisson()
    print(f'Poisson equation:\n     Average cost (φ): {phi}')
    policy_iteration()
    value_iteration(iterations)
