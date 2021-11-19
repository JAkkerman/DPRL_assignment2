#%%
import numpy as np


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


def stationary():
    P = np.zeros((91,91))
    P[0,0] = 0.1
    P[0,1] = 0.9
    for i in range(1,91):
        P[i,0] = P[i-1,0] + 0.01
        if i < 90:
            P[i,i+1] = P[i-1,i] - 0.01

    r = np.zeros(91).T
    r[0] = -1

    A = np.identity(91)
    A = np.zeros((91,91))
    A[:, :-1] = np.identity(91)[:,1:]
    A[:,-1] = 1  # fill in coefficient for phi
    A[:,:-1] -= P[:,1:]

    x = np.linalg.solve(A, r)
    print(x)
    print(sum(x))

#%%
if __name__ == '__main__':

    T = 10000
    N = 1000

    # deteriorate(T, N)
    stationary()
# %%
