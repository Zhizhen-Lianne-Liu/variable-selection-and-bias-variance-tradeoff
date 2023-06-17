% this script demonstrates the use of function handles
% it makes use of greedysubset.m, bestsubset.m, and testerror.m

clear;

%%%%%%%%%%%%%%%
%%% create a training and a test dataset
seed = 2; p = 12; randn('state',seed); 
ntrain = 30; ntest = 1000;
A = randn(5*p,p); sigmatrue = A'*A./max(max(A'*A));
sigmatrue = 0.9*sigmatrue + 0.1*eye(p,p);
betatrue = 3*randn(p,1); 
betatrue(1) = 0; betatrue(2) = 0; betatrue(5) = 0; 
xtrain = randn(ntrain,p)*chol(sigmatrue);
ytrain = xtrain*betatrue + randn(ntrain,1);
xtest = randn(ntest,p)*chol(sigmatrue);
ytest = xtest*betatrue + randn(ntest,1);
Ttrain = [ytrain,xtrain];
Ttest = [ytest,xtest];


%%% the following two code segments perform an identical task
% evaluate greedysubset at Ttrain directly ...
B1 = greedysubset(Ttrain); 
% evaluate greedysubset at Ttrain indirectly ...
f = @greedysubset; B2 = feval(f,Ttrain); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% USING FUNCTION HANDLES ONE CAN AVOID UNNECESSARY REPETITION 
%%% in situations where the same task  needs to be performed for several 
%%% different functions whose input-output format is identical. 

%%% e.g., we will extract the regression candidate with minimal test error
%%% from each of our sparse linear estimators above, and display it.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MOST NAIVE, REPETITIVE WAY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = bestsubset(Ttrain); % run the sparse linear estimator
for j = 1:p
    rss_test(j) = mean((ytest-xtest*B(:,j)).^2); % test error
end
[a,b] = min(rss_test); % find the sparse solution with minimum error
beta_selected{1} = B(:,b);

B = greedysubset(Ttrain); % run the sparse linear estimator
for j = 1:p
    rss_test(j) = mean((ytest-xtest*B(:,j)).^2); % test error
end
[a,b] = min(rss_test); % find the sparse solution with minimum error
beta_selected{2} = B(:,b);
disp([beta_selected{1},beta_selected{2}])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% BETTER WAY: STORE REPEATED SCRIPT AS A FUNCTION, testerror.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = @bestsubset; 
beta_selected{1} = testerror(f,Ttrain,Ttest);
f = @greedysubset; 
beta_selected{2} = testerror(f,Ttrain,Ttest);
disp([beta_selected{1},beta_selected{2}])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% BEST WAY: EACH LINE OF CODE APPEARS ONCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h{1} = @bestsubset; 
h{2} = @greedysubset; 
for count = 1:size(h,2)
    beta_selected{count} = testerror(h{count},Ttrain,Ttest);
end
disp([beta_selected{1},beta_selected{2}])
%%% note that crossval.m will only take f and Ttrain as input