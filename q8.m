
clear
format long g
%programs for q8:
prostate = load('prostate.dat');
prostate = prostate(randperm(97), :);

p = size(prostate, 2)-1;
N = size(prostate, 1);


%first generate 4 sets of zero mean unit variance covariates
% Xtr = [0.835489204101061,-1.46580510397612,0.0212805473288732,-0.135257550419820,-0.708490069016039,-0.139643693780653,1.09834172334447,0.556425679015507;0.489248663098974,0.530720815903653,-0.976470496506693,1.19801793212329,-0.938710201892785,-1.33653041457421,0.417569499807183,-1.48559668860604;0.107412959889818,0.724990682412462,1.36918304882841,-1.23199728131778,1.18431512871572,0.509350206886977,-1.24287197207159,0.625385940122995;-1.43215082708985,0.210093605660008,-0.413993099650588,0.169236899614303,0.462885142193108,0.966823901467885,-0.273039251080060,0.303785069467533];
% Etr = [-0.353592909995806;-2.14076864373318;0.966209335370828;0.458054708305766];

Xtr = normrnd(0, 1, 97, 4); % 4 normally distributed covariates with zero mean and unit variance

% M = mean(Xtr, 1);
% S = std(Xtr, 0, 1);
% 
% for i = 1:p
%     for j = 1:4
%         Xtr(j, i) = (Xtr(j, i) - M(i))/(S(i));
%     end
% end

% beta = [0.276630772106680;0.374786031037808;0.396795234377106;0.132116169297616;0.966225713000461;0.961158982324749;0.0140102598713646;0.509365104852530];
% Ytr = Xtr*beta + Etr;

%now add onto actual dataset
%insert into 4 random positions so it wouldn't favor training or testing
%set.
RSStest = [];
RSStrain = [];
lcavolset = [];
lweightset = [];
ageset = [];
lbphset = [];
sviset = [];
lcpset = [];
gleasonset = [];
pgg45set = [];
random1 = [];
random2 = [];
random3 = [];
random4 = [];

for i = 1:500
    [tempRSStest,tempRSStrain, tempbeta] = crossvalmethod(Xtr, prostate);
    RSStest = [RSStest tempRSStest];
    RSStrain = [RSStrain tempRSStrain];
    lcavolset = [lcavolset tempbeta(1)];
    lweightset = [lweightset tempbeta(2)];
    ageset = [ageset tempbeta(3)];
    lbphset = [lbphset tempbeta(4)];
    sviset = [sviset tempbeta(5)];
    lcpset = [lcpset tempbeta(6)];
    gleasonset = [gleasonset tempbeta(7)];
    pgg45set = [pgg45set tempbeta(8)];
    random1 = [random1 tempbeta(9)];
    random2 = [random2 tempbeta(10)];
    random3 = [random3 tempbeta(11)];
    random4 = [random4 tempbeta(12)];

end
    
%want to plot a histogram for each of the covariates of betaCV as
%well to deduce which covariates have a larger impact on lpsa. The
%covariates being lcavol, lweight, age, lbph, svi, lcp, gleason,
%and pgg45 respectively
f1 = figure;
histogram(RSStest, 10);
xlabel('RSS for test set', 'interpreter', 'latex')
ylabel('Freq')
f2 = figure;
histogram(RSStrain, 10);
xlabel('RSS for training set', 'interpreter', 'latex')
ylabel('Freq')
f3 = figure;
histogram(lcavolset);
xlabel('lcavol $\hat{\beta}^{CV}$', 'interpreter', 'latex')
ylabel('Freq')
f4 = figure;
histogram(lweightset);
xlabel('lweight $\hat{\beta}^{CV}$', 'interpreter', 'latex')
ylabel('Freq')
f5 = figure;
histogram(ageset);
xlabel('age $\hat{\beta}^{CV}$', 'interpreter', 'latex')
ylabel('Freq')
f6 = figure;
histogram(lbphset);
xlabel('lbph $\hat{\beta}^{CV}$', 'interpreter', 'latex')
ylabel('Freq')
f7 = figure;
histogram(sviset);
xlabel('svi $\hat{\beta}^{CV}$', 'interpreter', 'latex')
ylabel('Freq')
f8 = figure;
histogram(lcpset);
xlabel('lcp $\hat{\beta}^{CV}$', 'interpreter', 'latex')
ylabel('Freq')
f9 = figure;
histogram(gleasonset);
xlabel('gleason $\hat{\beta}^{CV}$', 'interpreter', 'latex')
ylabel('Freq')
f10 = figure;
histogram(pgg45set);
xlabel('pgg45 $\hat{\beta}^{CV}$', 'interpreter', 'latex')
ylabel('Freq')
f11 = figure;
histogram(random1);
xlabel('random 1')
ylabel('Freq')
f12 = figure;
histogram(random2);
xlabel('random 2')
ylabel('Freq')
f13 = figure;
histogram(random3);
xlabel('random 3')
ylabel('Freq')
f14 = figure;
histogram(random4);
xlabel('random 4')
ylabel('Freq')
    
function [RSStest,RSStrain, betaCV] = crossvalmethod(Xtr, prostate)
%     temp = [Ytr Xtr];
%     k = randi([1, N], 1, 4);
%     k = sort(k);
%T = [prostate([1:k(1)],:); temp(1, :); prostate([k(1)+1:k(2)], :); temp(2, :); prostate([k(2)+1:k(3)], :); temp(3, :); prostate([k(3)+1:k(4)], :); temp(4, :); prostate([k(4)+1:end], :)];
    T = [prostate Xtr];
%    T = T(randperm(97), :);
    Train = T([1:70], :);
    Test = T([71:end], :);

    %now find betacv with crossval using monotonic lars as sparse estimator
    f = @monotonic_lars;
    betaCV = crossval(Train, f);

    RSSval = RSS(betaCV, Test, size(Test, 1));
    RSStrain = RSS(betaCV, Train, size(Train, 1));
    RSStest = RSSval;
    
end




