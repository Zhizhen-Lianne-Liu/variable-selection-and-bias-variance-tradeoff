clear
format long g

p = 10;
N = 30;
%program files for q4: q4.m, generateTrain.m, greedysubset.m
T = generateTrain(p, N);
B = greedysubset(T);