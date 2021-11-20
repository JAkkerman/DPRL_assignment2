# %%
#initialising variables
import numpy as np

beta = 0.01
initial_rate = 0.1
states = 91
# %%
# defining probabilities
def pπ(i):
    return(initial_rate+beta*i)
def prob(i):
    return(initial_rate+beta*i, 1-(initial_rate+beta*i))
# %%
#probability of state transaition through action 1
p_a1 = np.zeros((states, states))
for x in range(len(p_a1[0])-1):
    p_a1[0][x], p_a1[x+1][x] = prob(x)
p_a1[0][90] = 1
# %%
# declaring costs
cost = p_a1[0]
# %%
# calculating stationary Distribution 'π'
πp = []
πp.append(1)
πp.append(0.9)
for i in range(2, states):
    πp.append(πp[i-1]*(πp[1]-(0.01*(i-1))))

π = []
π.append(1/sum(πp))

# print(π)
for j in range(1, states):
    π.append(π[0]*πp[j])
# print(π)
# print(sum(π))  # checking if it's 1
# %%
# calculating ø = ∑ π•c
ø = 0
for i in range(states):
    ø += cost[i]*π[i]
print("ø = ",ø)

# %%
# solving average cost Poisson's Equation
V = np.zeros((states))#initializing all values and taking V[0] as zero for computations
V[states-1] = 1 - ø + V[0]
for i in range(states-2):
    j = states-2-i
    V[j] = cost[j] + p_a1[0][j]*V[0] + p_a1[j+1][j]*V[j+1] - ø
# print(V)
# %%
#copyting V before value iteration
oldV = V
#%%
#Starting Value Iteration
V = np.zeros((states))
newV = np.zeros((states))
for j in range(1000):
    for i in range(states-1):
        newV[i] = min(cost[i] + p_a1[0][i]*V[i] + p_a1[i+1][i]*V[i+1], 0.5 + V[0])
    newV[states-1] = 0.5 + V[0]
    V = newV
print("Convergence at Step: ",len(list(set(V))))

# %%
