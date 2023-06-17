%programs for q6: q6.m, crossval.m, RSS.m, generateTrain.m, bestsubset.m
clear
format long g

p = 10;
N = 200;
T = generateTrain(p, N);
f = @bestsubset;
betaCV = crossval(T, f);