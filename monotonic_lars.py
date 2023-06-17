#modified lars from monotonic_lars in MATLAB
#need a lot more testing to be sure
import numpy as np
from scipy.sparse.linalg.isolve.lsqr import _sym_ortho
#import matlab.engine
#eng = matlab.engine.start_matlab()

def minus1(L):
    return [x-1 for x in L]

def monotonic_lars(T):
    #input checking
    method = 'lasso'
    y = T[:, 0]
    X = T[:, 1:]


    if method == 'lasso':
        lasso = 1
        lambda2 = 0
    
    p = X.shape[1]
    n = X.shape[0]

    maxk = 8*(n+p)

    #according to documentation: eps = 2^(-52)
    if lambda2 < 2**(-52):
        nvars = min([n-1, p])
    else:
        nvars = p
    
    if stop > 0:
        stop = stop/((1+lambda2)**(1/2))

    if stop == 0:
        beta = np.zeros((2*nvars, p))
    elif stop < 0:
        beta = np.zeros((2*round(-stop), p))
    else:
        beta = np.zeros((100, p))

    mu = np.zeros((n, 1))
    I = [i for i in range(p)]
    A = []
    R = np.array([])
    lassocond = 0
    stopcond = 0
    k = 0
    vars = 0

    d1 = (lambda2)**(1/2)
    d2 = 1/((1 + lambda2)**(1/2))

    if trace == 1:
        print('Step Added Dropped  Active set size')

    #elastic net main loop
    while vars < nvars and stopcond == 0 and k < maxk:
        k  = k + 1
        c = np.matmul(np.transpose(X), (y - mu)) * d2
        temp = np.reshape(c, (1,np.product(c.shape)))
        temp = temp[:, I]
        temp = temp[0]
        C,j = max(temp), np.argmax(temp)

        for i in range(len(j)):
            j[i] = I[j[i]]

        #confused, why do we need this?
        #####################check check

        if lassocond == 0:
            R = cholinsert(R, X[:, j], X[:, A], lambda2)
            A.append(j)

            for i in range(len(I)-1, -1, -1):
                if I[i]  == j[i]:
                    I.pop(i)
            vars = vars + 1
            if trace == 1:
                print(k)
                print(j)
                print(vars)
        
        s = np.sign(c[A, :]);

        GA1 = np.linalg.solve(R, np.linalg.solve(np.transpose(R), s))
        AA = 1/((np.sum(GA1*s))^(1/2))
        w = AA*GA1
        u2 = np.zeros(p, 1)
        u2 = d1*d2*w
        u1 = X[:, A]*w*d2
        a = np.transpose(X)@u1*d2 + d1*u2*d2

        if vars == nvars:
            gamma = C/AA
        else:
            temp = np.vstack(np.divide((C-c[I]), (AA-a[I])), np.divide((C+c[I]), (AA+a[I])))
            gamma = min(np.vstack(temp[temp>0], np.divide(C, AA)))
        
        if lasso:
            lassocond = 0
            temp = np.divide(-beta[k, A], np.transpose(w))
            gamma_tilde = min(np.hstack(temp[temp>0], gamma))
            j = np.argwhere(temp == gamma_tilde)

            if gamma_tilde < gamma:
                gamma = gamma_tilde
                lassocond = 1

        mu = mu + gamma * u1
        if beta.shape[0] < k+1:
            beta = np.vstack(beta, np.zeros(beta.shape[0], p))
        beta[k+1, A] = beta[k, A] + gamma* np.transpose(w)

        #early stopping at specified bound on L1 norm of beta
        if stop > 0:
            t2 = np.sum(abs(beta[k+1, :]))
            if t2 >= stop:
                t1 = np.sum(abs(beta[k, :]))
                s = (stop-t1)/(t2-t1)
                beta[k+1, :] = beta[k, :] + s*(beta[k+1, :] - beta[k, :])
                stopcond = 1

        if lassocond == 1:
            R = choldelete(R, j)
            I = I + A[j]
            A[j] = []
            vars = vars-1
            if trace ==1:
                print(k)
                print(j)
                print(vars)

        if stop < 0:
            stopcond = int(vars >= -stop)

    #trim beta
    if beta.shape[0] > k+1:
        beta[k+2:, :] = np.array([])

    #divide by d2 to avoid double shrinkage
    beta = beta/d2
    beta = np.transpose(beta[2:, :])

    #turn output monotonic in model size
    betaold = beta
    beta = []

    nos = []
    for j in range(betaold.shape[1]):
        nos.append(np.argwhere(betaold[:, j]).shape[0])
    nos = np.asarray(nos)

    for j in range(p):
        #in the very unlikely case where one model size is not represented, cheat
        if not (np.argwhere(nos ==j)):
            beta[:, j] = betaold[:, np.argwhere(nos==(j-1))[0]]
        else:
            beta[:, j] = betaold[:, np.argwhere(nos == j)[len(np.argwhere(nos==j))-1]]

    if k == maxk:
        print('Lars WARNING: forced exit, Max number of iteration reached')

    return beta
############################################
##fast cholesky insert and remove functions
############################################

def cholinsert(R, x, X, lambd):
    diag_k = np.divide((np.transpose(x)@x + lambd), (1 + lambd))
    if not R:
        R = np.sqrt(diag_k)
    else:
        col_k = np.divide(np.transpose(x)@X, (1+lambd))
        R_k = np.linalg.solve(np.transpose(R), np.transpose(col_k))
        R_kk = np.sqrt(np.subtract(diag_k, np.transpose(R_k)@R_k))
        temp1 = np.hstack(R, R_k)
        temp2 = np.hstack(np.zeros(1, R.shape[1]), R_kk)
        R = np.vstack(temp1, temp2)
    return R

def choldelete(R, j):
    R[:, j] = []
    n = R.shape[1]

    for k in range(j, n):
        p = [k, k+1]
        [G, R[p, k]] = planerot(R[p, k])
        if k<n:
            R[p, k+1:n] = G * R[p, k+1:n]
    R[R.shape[1]-1, :] = []

def planerot(x):
    # This is like symGivens2 in pylearn2 and planerot in Matlab.
    a, b = x
    c, s, r = _sym_ortho(a, b)
    G = np.array([
        [c, s],
        [s, -c]])
    return G, r


        