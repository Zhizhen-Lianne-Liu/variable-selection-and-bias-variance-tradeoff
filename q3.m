clear
format long g

%for this question the relevant program files are q3.m, bestsubset.m and
%generateTrain.m

p = 10;
N = 30;
T = generateTrain(p, N);

B = bestsubset(T);