function betaselected = testerror(f,Ttrain,Ttest)

ytest = Ttest(:,1);
xtest = Ttest(:,[2:end]);
[ntest,p] = size(xtest);

B = feval(f,Ttrain); % run the sparse linear estimator
size(B)

for j = 1:p
    rss_test(j) = mean((ytest-xtest*B(:,j)).^2); % test error
end
[a,b] = min(rss_test); % find the sparse solution with minimum error
betaselected = B(:,b);