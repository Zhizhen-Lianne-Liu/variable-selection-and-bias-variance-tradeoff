clear
format long g

%program files for question 2: q2.m, RSStrte.m

p = 10;
RSStr_avg = zeros(1, p);
RSSte_avg = zeros(1, p);

model_size = 1:p;

for i = 1:100
    [RSStr, RSSte] = RSStrte(p);
    RSStr_avg = RSStr_avg + (1/100)*RSStr;
    RSSte_avg = RSSte_avg + (1/100)*RSSte;
 
end

figure(1)
hold on;
scatter(model_size, RSStr_avg);
scatter(model_size, RSSte_avg);
l1 = 'RSS training';
l2 = 'RSS testing';
xlabel('Model size')
ylabel('average RSS')

legend(l1, l2)

% %training size Ntr = 30
% title('RSS when training set has size 30')
% exportgraphics(gcf,'q2_1.jpg','Resolution',300)

%when Ntr = 200
title('RSS when training set has size 200')
exportgraphics(gcf,'q2_2.jpg','Resolution',300)






