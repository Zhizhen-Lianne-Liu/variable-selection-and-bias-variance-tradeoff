
import numpy as np
import math
import itertools


def generateTrain(p, Ntr):
    var = 1
    Xtr = np.random.normal(loc = 0, scale =1, size = (Ntr, p))
    Etr = np.random.normal(loc = 0, scale = var**(1/2), size = (Ntr, 1))
    beta = np.array([[-0.5], [-0.45], [-0.4], [0.35], [-0.3], [0.25], [-0.2], [0.15], [-0.1], [0.05]])
    Ytr = np.matmul(Xtr, beta) + Etr

    T = np.column_stack((Ytr, Xtr))

    return T

#this is copied over from q3.py and now actually takes T
def bestsubset(T):
    #assume T is taken as np array with first column as Y
    TY = T[:, 0]
    TX = T[:, 1:]
    p = T.shape[1]-1
    Ntr = T.shape[0]
    
    wholemodel = [i for i in range(p)]
    B = []#append chosen betas of least RSS, first in rows, then take transpose
    
    #For testing
    #input of generateModel takes p and size of training model Ntr
    
    for i in range (1, p+1):
        betas = []#saves all betas of different models of size i
        RSS = []#saves all corresponding RSS of models of size i relative to betas
        for M in itertools.combinations(wholemodel, i):
            XM = TX[:, M]
            RSS.append(0)
            betaM = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(XM), XM)), np.transpose(XM)), TY)
            for j in range(Ntr):
                RSS[len(RSS)-1] += (1/Ntr)*(TY[j]-XM[j]@betaM)**2

            temp = [0 for m in range (p)]
            count = 0
            for k in M:
                temp[k] = betaM[count]
                count += 1
            betaM = temp
            
            betas.append(betaM)
        index_min = np.argmin(RSS)
        B.append(betas[index_min])
        
    #now take transpose of B
    B = np.asarray(B)
    B = np.transpose(B)

    return B


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





p = 10
N = 200
#here i have just used same training model as q2, but replace this with x an y values of any training data desired.
crossval(generateTrain(p, N), bestsubset)
