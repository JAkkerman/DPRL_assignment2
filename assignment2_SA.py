#%%
beta = 0.01
initial_rate = 0.1
states=91
#%%
def p(i):
    return(initial_rate+beta*i)
#%%
sum2 = 0
for j in range(1,states):
    prod = 1
    for i in range(1,j+1):
        prod*=p(i)
    sum2+=prod

π = []
π.append(1/(1+sum2))

print(π)
for j in range(1,states):
    prod = 1
    for i in range(0,j):
        prod*=p(i+1)
    π.append(prod*π[0])
    
# %%
print(sum(π))
# %%
print(π)
# %%
ø=0
for i in range(len(π)-1):
    ø+=(i+1)*π[i]
print(ø)
# %%
