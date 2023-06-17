function [RSStr, RSSte] = RSStrte(p)
    Ntr = 200;
    Nte = 1000;
   
    
    %training:
    Xtr = normrnd(0, 1, Ntr, p);
    Xte = normrnd(0, 1, Nte, p);
    
    Etr = normrnd(0, 1, Ntr, 1);
    Ete = normrnd(0, 1, Nte, 1);
    
    beta = [-0.5 -0.45 -0.4 0.35 -0.3 0.25 -0.2 0.15 -0.1 0.05]';
    
    Ytr = Xtr*beta + Etr;
    Yte = Xte*beta + Ete;
    
    RSStr = zeros(1, p);
    RSSte = zeros(1, p);
    
    for j = 1:p 
       
        XM = Xtr(:, [1:j]);
        betaM = inv(XM' * XM) * XM' * Ytr;
        %find training error
        for i = 1:Ntr
            
            RSStr(j) = RSStr(j) + (1/Ntr)*(Ytr(i) - XM(i, :)*betaM).^2;
        end
        
        XMte = Xte(:, [1:j]);
        %now for test error
        for k = 1:Nte
            RSSte(j) = RSSte(j) + (1/Nte)*(Yte(k) - XMte(k, :)*betaM).^2;
        end
    end
       