function betaCV = crossval(T, estimator)
    p = size(T, 2)-1;
    N = size(T, 1);
    PE = zeros(1, p);
    
    for k = 1:10
       Tk = zeros(1, p + 1);
       Tk_complement = T;
       
       pi = randperm(N);
       
       if ceil(N*(k-1)/10) == N*(k-1)/10
           n = N*(k-1)/10 + 1;
       else
           n = ceil(N*(k-1)/10);
       end
       lower = n;
       upper = floor((N*k)/10);
       rowsToDelete = [];
       while n <= ((N*k)/10)
           Tk = [Tk; T(pi(n), :)];
           rowsToDelete = [rowsToDelete pi(n)];
           n = n+1;
       end
       Tk(1, :) = [];
       Tk_complement(rowsToDelete, :) = [];
       B = feval(estimator,Tk_complement);
       for j = 1:p
          betaj = B(:, j);
          PE(j) = PE(j) + (1/10)* RSS(betaj, Tk, upper-lower+1);
       end
       [A, cvIndex] = min(PE);
       B_final = feval(estimator,T);
       betaCV = B_final(:, cvIndex);
       
    end
    
    
    
    
    
    
    
    
    
    