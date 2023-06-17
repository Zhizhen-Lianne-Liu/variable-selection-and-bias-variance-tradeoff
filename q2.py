#######################
#generate training and test data sets

from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

def RSStrte(p):
  
    Ntr = 200
    Nte = 1000
    var = 1
    #training:
    Xtr = np.random.normal(loc = 0, scale =1, size = (Ntr, p))
    Xte = np.random.normal(loc = 0, scale =1, size = (Nte, p))

    Etr = np.random.normal(loc = 0, scale = var**(1/2), size = (Ntr, 1))
    Ete = np.random.normal(loc = 0, scale = var**(1/2), size = (Nte, 1))

    beta = np.array([[-0.5], [-0.45], [-0.4], [0.35], [-0.3], [0.25], [-0.2], [0.15], [-0.1], [0.05]])

    Ytr = np.matmul(Xtr, beta) + Etr
    Yte = np.matmul(Xte, beta) + Ete

    #generate models:
    M = [[i for i in range(1, j+1)]for j in range(1, p+1)]
    betaM = [[] for i in range(p)]
    RSStr = [0 for j in range(p)]
    RSSte = [0 for j in range(p)]

    for j in range(p):
        model = M[j]
        XM = Xtr[:, :(j+1)]

        betaM = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(XM), XM)), np.transpose(XM)), Ytr)
        
        #finding training error
        for i in range(Ntr):
            RSStr[j] += (1/Ntr)*(Ytr[i]-XM[i]@betaM)**2

        XMte = Xte[:, :(j+1)]
        #now for test error
        for k in range(Nte):
            RSSte[j] += (1/Nte)*(Yte[k] - XMte[i]@betaM)**2

    return RSStr, RSSte

 

def main():
    p = 10
    RSStr_avg = [0 for i in range(p)]
    RSSte_avg = [0 for i in range(p)]
    model_size = [i+1 for i in range(p)]
    for i in range(100):
        RSStr, RSSte = RSStrte(p)
        RSStr_avg = list(map(np.add, np.array(RSStr_avg), (1/100)* np.array(RSStr)))
        RSSte_avg = list(map(np.add, np.array(RSSte_avg), (1/100)*np.array(RSSte)))

    plt.scatter(model_size, RSStr_avg, label = 'RSS Training average', s = 5)
    plt.scatter(model_size, RSSte_avg, label = 'RSS Testing average', s = 5)

    plt.title('Averge RSS against model size with N_tr = 200')
    plt.xlabel('Model Size')

    plt.legend(loc = 'center right')

    #Ntr = 30
    #plt.savefig('q2_1.jpg', dpi = 300)

    #Ntr = 200
    plt.savefig('q2_2.jpg', dpi = 300)
    plt.show()
    print()

if __name__ == '__main__':
    main()