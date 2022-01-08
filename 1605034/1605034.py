# %%
import numpy as np
from scipy.stats import norm

# read file 

data = open("Input/data.txt", "r")
param = open("Input/parameters.txt", "r")



# %% [markdown]
# # data preprocessing and functions

# %%
# read lines from file as numbers   

rainfall = []
for line in data:
    rainfall.append(float(line))


N = int(param.readline())

trans_matrix = []

for i in range(N):
    p = [float(x) for x in param.readline().split()]
    trans_matrix.append(p)


# read distro

means = [float(x) for x in param.readline().split()]
sds = [np.sqrt(float(x)) for x in param.readline().split()]





# %%
# gaussian distribution function

def gauss(x , mean , sd):
    prob_density = norm.pdf(x, loc=mean, scale=sd)
    return prob_density


# find stationary distribution for given transition matrix

def get_stationary_distribution(A,pi):
    A = np.array(A)
    pi = np.array(pi)
    
    for _ in range(1000):
        pi = np.dot(pi,A)
   
    
    return pi



# %%

# emission probabilities
emisson_matrix = []

n = len(trans_matrix)

for i in range(n):
    prob = []
    for d in rainfall:
        prob.append(gauss(d, means[i], sds[i]))
    emisson_matrix.append(prob)



start = get_stationary_distribution(trans_matrix,[0.5,0.5])

# %% [markdown]
# # viterbi algorithm

# %%

def viterbi(A,B,pi) :
    A = np.array(A)
    B = np.array(B)
    pi = np.array(pi)
 
    N = len(A)
    T = len(B[0])
    
    # initial state 
    delta = np.zeros((N,T))
    psi = np.zeros((N,T))
    delta[:,0] = np.log( pi*B[:,0] )
    psi[:,0] = 0


    for i in range(1,T) :
        for j in range(N) :
            delta[j,i] = np.max(delta[:,i-1] + np.log(A[:,j]*B[j,i]))
            psi[j,i] = np.argmax(delta[:,i-1] + np.log(A[:,j]*B[j,i]))

    
    # backtracking
    
    sol = np.zeros(T)
    sol[T-1] = np.argmax(delta[:,T-1])
    
    for t in range(T-1,0,-1):
        sol[t-1] = psi[int(sol[t]),t]
    
    return sol


Sol =  viterbi(trans_matrix,emisson_matrix,start)


# solution write in output file

out1 = open("my_output/wo_learing.txt","w")

for s in Sol :
    if s == 1 :
        out1.write("La Nina\n")
    else:
        out1.write("El Nino\n")



    

# %% [markdown]
# # Baum Welch Learning

# %%
# baum welch algorithm for learning parameters

def Baum_welch(Ob) :

    # initial distribution 
    A = np.array(trans_matrix)
    Mus = np.array(means)
    Sds = np.array(sds)
    pi = np.array(start)

    N = len(A)
    T = len(Ob)

    iter = 5

    for _ in range(iter):
    
        B = np.zeros((N,T))
        for i in range(N):
            for t in range(T):
                B[i,t] = gauss(Ob[t],Mus[i],Sds[i])


        forward = np.zeros((N,T))
        backward = np.zeros((N,T))

        forward[:,0] = pi
        backward[:,T-1] = 1
    
        # forward
        for t in range(1,T):
            for i in range(N):
                for j in range(N):
                    forward[i,t] += forward[j,t-1] * A[j,i] * B[i,t]
            forward[:,t] /= np.sum(forward[:,t])

        # backward
        for t in range(T-2,-1,-1):
            for i in range(N):
                for j in range(N):
                    backward[i,t] += A[i,j] * B[j,t+1] * backward[j,t+1]
            backward[:,t] /= np.sum(backward[:,t])


        # sum of last col of forward
        alpha = np.sum(forward[:,T-1])


        gamma = np.zeros((N,T))
        for t in range(T):
            for i in range(N):
                gamma[i,t] = forward[i,t] * backward[i,t] / alpha
            gamma[:,t] /= np.sum(gamma[:,t])
        
        
        zeye = np.zeros((N,N,T))
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    zeye[i,j,t] = forward[i,t] * A[i,j] * B[j,t+1] * backward[j,t+1] / alpha
            zeye[:,:,t] /= np.sum(zeye[:,:,t])

   
        # update transition matrix
        A = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                for t in range(T-1):
                    A[i,j] += zeye[i,j,t]
        
        # normalize transition matrix
        for i in range(N):
            A[i,:] /= np.sum(A[i,:])
        
        # update means
        Mus = np.zeros(N)
        for i in range(N):
            for t in range(T):
                Mus[i] += gamma[i,t] * Ob[t]
            Mus[i] /= np.sum(gamma[i,:])

        # update sds
        Sds = np.zeros(N)
        for i in range(N):
            for t in range(T):
                Sds[i] += gamma[i,t] * (Ob[t] - Mus[i])**2
            Sds[i] /= np.sum(gamma[i,:])
            Sds[i] = np.sqrt(Sds[i])
        

        # print("iteration : ",_)
        # print("transition matrix : \n",A)
        # print("means : \n",Mus)
        # print("sds : \n",Sds)
        # print("pi : \n",pi)
        # print("\n")



    return A,Mus,Sds,pi

    



# %%
trans_prob,means,sds,start = Baum_welch(rainfall)

print("trans :\n",trans_prob)
print("means :",means)
print("sds :",sds)
print("start",start)

# running viterbi on learned parameters

emisson_matrix = []

n = len(trans_prob)

for i in range(n):
    prob = []
    for d in rainfall:
        prob.append(gauss(d, means[i], sds[i]))
    emisson_matrix.append(prob)


Sol =  viterbi(trans_prob,emisson_matrix,start)


# solution write in output file

out2 = open("my_output/baum_welch_wo_learing.txt","w")

for s in Sol :
    if s == 1 :
        out2.write("La Nina\n")
    else:
        out2.write("El Nino\n")

out3 = open("my_output/parameters_learned.txt","w")

out3.write(str(n)+"\n")
for i in range(n):
    for j in range(n):
        out3.write(str(trans_prob[i][j])+" ")
    out3.write("\n")


for i in range(n):
    out3.write(str(means[i])+" ")
out3.write("\n")

vars = np.square(sds)
for i in range(n):
    out3.write(str(vars[i])+" ")
out3.write("\n")

for i in range(n):
    out3.write(str(start[i])+" ")
out3.write("\n")

out1.close()
out2.close()
out3.close()



