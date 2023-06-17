function B = greedysubset(T)
    TY = T(:, 1);
    TX = T(:,[2:end]);
    
    p = size(TX, 2);
    Ntr = size(T, 1);
    
    
    B = [];
    
    M = [];
    M_prev = [];
    
    for i = 1:p
        betas = [];
        RSS = [];
        addon_elements = [];
        for t = 1:p
            if ~ismember(t, M_prev)
                addon_elements = [addon_elements t];
            end
        end
        
        for k = 1:size(addon_elements, 2)
            M = [M_prev k];
            XM = TX(:, M);
            RSS = [RSS 0];
            betaM = inv(XM' * XM) * XM' * TY;
            for j = 1:Ntr
               l = size(RSS, 2);
               RSS(l) =  RSS(l) + (1/Ntr)*(TY(k) - XM(k, :)*betaM).^2;
            end
            temp = zeros(1, p);
            
            for m = 1:size(M, 2)
                temp(M(m)) = betaM(m);
            end
            betas = [betas; temp];
            
        end
        [A, index_min] = min(RSS);
        M_prev = [M_prev addon_elements(index_min)];
        B = [B betas(index_min, :)'];
    end
    
    
    
    
    
    
    
    