function T = generateTrain(p, Ntr)
%Ntr is size of training set, 
    Xtr = normrnd(0, 1, Ntr, p);
    
    Etr = normrnd(0, 1, Ntr, 1);
    beta = [-0.5 -0.45 -0.4 0.35 -0.3 0.25 -0.2 0.15 -0.1 0.05]';
    
    Ytr = Xtr*beta + Etr;
    
    T = [Ytr Xtr];