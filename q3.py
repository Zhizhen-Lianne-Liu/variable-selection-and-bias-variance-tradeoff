import numpy as np
import itertools

def generateTrain(p, Ntr):
    var = 1
    Xtr = np.random.normal(loc = 0, scale =1, size = (Ntr, p))
    Etr = np.random.normal(loc = 0, scale = var**(1/2), size = (Ntr, 1))
    beta = np.array([[-0.5], [-0.45], [-0.4], [0.35], [-0.3], [0.25], [-0.2], [0.15], [-0.1], [0.05]])
    Ytr = np.matmul(Xtr, beta) + Etr

    T = np.column_stack((Ytr, Xtr))

    return T



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
    # p = 10
    # Ntr = 200
    # TX, TY = generateTrain(p, Ntr)#here i have just used same training model as q2, but replace this with x an y values of any training data desired.

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


p = 10
N = 200
T = generateTrain(p, N)
bestsubset(T)