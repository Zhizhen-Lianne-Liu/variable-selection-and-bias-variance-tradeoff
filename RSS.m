function RSSval = RSS(beta, D, A)

    XM = D(:, 2:end);
    YM = D(:, 1);
    RSSval = 0;
    %A = int(A)
    for j = 1:A
        RSSval = RSSval + (1/A)*(YM(j)-XM(j, :)*beta)^2;
        
    end
        