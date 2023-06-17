function B = bestsubset(T)
    TY = T(:, 1);
    TX = T(:,[2:end]);
    
    p = size(TX, 2);
    Ntr = size(T, 1);
    
    wholemodel = 1:p;
    B = [];
    
    for i = 1:p
        betas = [];
        RSS = [];
        Models = nchoosek(wholemodel, i);
        for j = 1:size(Models, 1)
            M = Models(j, :);
            XM = TX(:, M);
            RSS = [RSS 0];
            betaM = inv(XM' * XM) * XM' * TY;
            
            for k = 1:Ntr
                l = size(RSS, 2);
                RSS(l) = RSS(l) + (1/Ntr)*(TY(k) - XM(k, :)*betaM).^2;
            end
            
            temp = zeros(1, p);
            
            for m = 1:size(M, 2)
                temp(M(m)) = betaM(m);
            end 
            betas = [betas; temp];
        end
        [A, index_min] = min(RSS);
        best_beta = betas(index_min, :);
        B = [B best_beta'];
        
    end
    
    
    
    