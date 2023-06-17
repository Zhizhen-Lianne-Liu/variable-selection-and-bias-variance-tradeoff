function bin = improvedfit(bestRSS, bestRSS_prev, N, d)
    F_statistic = (bestRSS_prev - bestRSS)/(bestRSS/(N-d-1));
    alpha = 0.05;
    
    p_value = fcdf(F_statistic, 1, N-d-1, 'upper');
    
    
    if p_value <= alpha
        bin = 1;
    else
        bin = 0;
    end
    