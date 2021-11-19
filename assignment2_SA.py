#%%
import numpy as np

beta = 0.01
initial_rate = 0.1
states=91
#%%

def pπ(i):
    return(initial_rate+beta*i)
def prob(i):
    return(initial_rate+beta*i,1-(initial_rate+beta*i))
#%%
p = np.zeros((states,states))
for x in range(len(p[0])-1):
    p[0][x],p[x+1][x] = prob(x)
p[0][90]=1

#%%
cost = p[0]

#%%
#calculating stationary Distribution 'π'
πp = []
πp.append(1)
πp.append(0.9)
for i in range(2,states):
    πp.append(πp[i-1]*(πp[1]-(0.01*(i-1))))

π = []
π.append(1/sum(πp))

print(π)
for j in range(1,states):
    π.append(π[0]*πp[j])
print(π)
print(sum(π))#checking if it's 1
# %%
#calculating ø = ∑ π•c
ø=0
for i in range(states):
    ø += cost[i]*π[i]
print(ø)

# %%
