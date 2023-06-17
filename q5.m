%q5 programs: q5.m, greedysubsetmodified.m, generateTrain.m, improvedfit.m
clear
format long g
p = 10;
N = 30;
T = generateTrain(p, N);
B = greedysubsetmodified(T);
K = bestsubsetmodified(T);