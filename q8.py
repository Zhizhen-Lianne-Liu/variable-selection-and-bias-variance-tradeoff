import numpy as np
#from sklearn import linear_model
import random
import math
from monotonic_lars import *

with open('prostate.txt', 'r') as f:
    prostate = [[num for num in line.split()] for line in f]

prostate = [[float(y) for y in x] for x in prostate]
prostate = np.asarray(prostate)
#print(prostate)
p = prostate.shape[1]-1


#generating 4 zero mean, unit-variance covariates sampled from a distribution of your liking(normal)
def augment():
    #take normal distribution
    var = 1 #unit variance
    Xtr = np.random.normal(loc = 0, scale =1, size = (4, p))#loc = 0 for zero mean
    Etr = np.random.normal(loc = 0, scale = var**(1/2), size = (4, 1))#loc = 0 for zero mean
    #take some random beta
    beta = [[random.uniform(0, 1)]for i in range(p)]
    #but just because the data is generated from zero mean unit variance normal distribution
    #does not mean data is zero mean and unit variance, so we need to standardize the covariates.
    mean = np.mean(Xtr, axis = 1)
    std = np.std(Xtr, axis = 1)
    for i in range(p):
        for j in range(4):
            Xtr[j][i] = (Xtr[j][i] - mean[i])/(std[i])
    Y = np.matmul(Xtr, beta) + Etr
    
    A = np.column_stack((Y, Xtr))

    prostate = np.vstack(prostate, A)

def RSS(beta, D, A):
    XM = D[:, 1:]
    YM = D[:, 0]
    RSSval = 0
    A = int(A)
    for j in range(A):
        RSSval += (1/A)*(YM[j]-XM[j]@beta)**2

    return RSSval
    


#here I am using bestsubset as a test for this algorithm
def crossval(T, estimator):
    #assume T is taken as np array with first column as Y

    N = T.shape[0]
    PE = [0 for j in range(p)]
    
    for k in range(1, 11):
        Tk = [0 for j in range(p+1)]
        Tk = np.asarray(Tk)
        Tk_complement = T
        #first we need to generate a random Tk
        pi = np.random.permutation(N)

        if math.ceil(N*(k-1)/10) == N*(k-1)/10:
            n = N*(k-1)/10 + 1
            
        else:
            n = math.ceil(N*(k-1)/10)
            
        lower = n 
        upper = math.floor((N*k)/10)
        rowsToDelete = []
        while n <= (N*k)/10:
            Tk = np.vstack((Tk, T[pi[int(n-1)]-1, :]))
            rowsToDelete.append(int(pi[int(n-1)])-1)
            
            n += 1
        Tk = np.delete(Tk, 0, 0)
        Tk_complement = np.delete(Tk_complement, rowsToDelete, 0)
        np.asarray(Tk)
        np.asarray(Tk_complement)
        B = estimator(Tk_complement)
        for j in range(p):
            betaj = B[:, j]

            PE[j] += (1/10)* RSS(betaj, Tk, upper-lower+1)

        #finally, argmin PE
    
    #actual index is this +1 since indexing starts from 0 in python.
    cvIndex = np.argmin(PE)

    B_final = estimator(T)
    betaCV = B_final[:, cvIndex]
    
    print(betaCV)
    return betaCV

def main():


    #first separate into training and test sets:
    Train = prostate[0:70, :]
    Test = prostate[71:, :]

    #in monotonic_lars.m, lambda2( = alpha) is always set to 0 for lasso.
    #fit_intercept is set to false because data is already centered
    #reg = linear_model.LassoLars(alpha=0, fit_intercept = False, normalize=False)

    #using lasso lars as the sparse estimator
    betaCV = crossval(Train, monotonic_lars)
    #because reg.fit takes X and y separately, I have made some changes to crossval to accomodate this

    print()


if __name__ == '__main__':
    main()