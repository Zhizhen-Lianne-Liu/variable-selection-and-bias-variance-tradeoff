function beta = monotonic_lars(T)
% LARSEN  The LARS algorithm for lasso, lars and elastic net regression, 
% 	  amended to take as input either the data or the estimated covariance matrix
%    BETA = LARSEN(X, Y,method,1) performs lars, lasso or elastic net regression on 
%    the variables in X to approximate the response Y.  Variables X are assumed to be
%    normalized (zero mean, unit length), the response Y is assumed to be
%    centered. 
%    BETA = LARSEN(Sigma_XX, Sigma_XY, method, 0) performs lars, lasso or elastic net
%    regression on the dataset X,y whose estimated statistics are Cov(X,y) = Sigma_XY
%    and Cov(X,X) = Sigma_XX. The covariance matrix Sigma_XX is assumed standardised. 
%    Method takes respectively values 'lars', 'lasso' or 'elnet'.
%    For method = 'elnet', the ridge term coefficient is given by varargin{1}.lambda2. 
%    The value lambda2 = 0 trivialises to the lasso solution. Conversly, when using 
%    'lars' or 'lasso' it is internally set to 0. If using 'elnet' and a value for 
%    lambda2 has not been provided, then it is set to 1e-6. This keeps the ridge 
%    influence low while making p > n possible.
%    varargin{1}.STOP with nonzero STOP will perform
%    elastic net regression with early stopping. If STOP is negative, its 
%    absolute value corresponds to the desired number of variables. If STOP
%    is positive, it corresponds to an upper bound on the L1-norm of the
%    BETA coefficients.
%    varargin{1}.TRACE with nonzero TRACE will print the adding and subtracting of 
%    variables as all elastic net solutions are found.
%    Returns BETA where each row contains the predictor coefficients of
%    one iteration. A suitable row is chosen using e.g. cross-validation,
%    possibly including interpolation to achieve sub-iteration accuracy.
%
% Primary Author: Karl Skoglund, IMM, DTU, kas@imm.dtu.dk
% Amended by: Christoforos Anagnostopoulos, IMS, ICL, christoforos.anagnostopoulos06@imperial.ac.uk
% References: 
%	'Regularization and Variable Selection via the Elastic Net' by Hui Zou and Trevor Hastie, 2005.
%  	'Least Angle Regression' by Bradley Efron et al, 2003.

%% Input checking
method = 'lasso';
X = T(:,[2:end]);
y = T(:,1); 

params = [];
if isfield(params,'trace') trace = params.trace; else trace = 0; end
if isfield(params,'stop') stop = params.stop; else stop = 0; end
if isequal(method,'lasso')
	lasso = 1;
	lambda2 = 0;
end
if isequal(method,'elnet')
	lasso = 1;
	if isfield(params,'lambda2') lambda2 = params.lambda2; else lambda2 = 1e-6; end	
end
if isequal(method,'lars')
	lasso = 0;
	lambda2 = 0;
end
%% Elastic net variable setups
[n p] = size(X);
maxk = 8*(n+p); % Maximum number of iterations


if lambda2 < eps % pure Lasso
	nvars = min(n-1,p);
else
	nvars = p; % Elastic net
end
if stop > 0,
	stop = stop/sqrt(1 + lambda2);
end
if stop == 0
	beta = zeros(2*nvars, p);
elseif stop < 0
	beta = zeros(2*round(-stop), p);
else
	beta = zeros(100, p);
end
mu = zeros(n, 1); % current "position" as LARS-EN travels towards lsq solution
I = 1:p; % inactive set
A = []; % active set
R = []; % Cholesky factorization R'R = X'X where R is upper triangular
lassocond = 0; % Set to 1 if LASSO condition is met
stopcond = 0; % Set to 1 if early stopping condition is met
k = 0; % Algorithm step count
vars = 0; % Current number of variables

d1 = sqrt(lambda2); % Convenience variables d1 and d2
d2 = 1/sqrt(1 + lambda2); 

if trace
	disp(sprintf('Step\tAdded\tDropped\t\tActive set size'));
end

%% Elastic net main loop
while vars < nvars && ~stopcond && k < maxk
	k = k + 1;
	c = X'*(y - mu)*d2;
	[C j] = max(abs(c(I)));
	j = I(j);
	
	if ~lassocond % if a variable has been dropped, do one iteration with this configuration (don't add new one right away)
		R = cholinsert(R,X(:,j),X(:,A),lambda2);
		A = [A j];
		I(I == j) = [];
		vars = vars + 1;
		if trace
			disp(sprintf('%d\t\t%d\t\t\t\t\t%d', k, j, vars));
		end
	end
	
	s = sign(c(A)); % get the signs of the correlations
	
	GA1 = R\(R'\s);
	AA = 1/sqrt(sum(GA1.*s));
	w = AA*GA1; % weights applied to each active variable to get equiangular direction
	u2 = zeros(p, 1); u2(A) = d1*d2*w; % part 2
	u1 = X(:,A)*w*d2; % equiangular direction (unit vector) part 1
	a = X'*u1*d2 + d1*u2*d2; % correlation between each variable and eqiangular vector
	
	if vars == nvars % if all variables active, go all the way to the lsq solution
		gamma = C/AA;
	else
		temp = [(C - c(I))./(AA - a(I)); (C + c(I))./(AA + a(I))];
		gamma = min([temp(temp > 0); C/AA]);
	end
	
	% LASSO modification
	if lasso
		lassocond = 0;
		temp = -beta(k,A)./w';
		[gamma_tilde] = min([temp(temp > 0) gamma]);
		j = find(temp == gamma_tilde);
		if gamma_tilde < gamma,
			gamma = gamma_tilde;
			lassocond = 1;
		end
	end
	mu = mu + gamma*u1;
	if size(beta,1) < k+1
		beta = [beta; zeros(size(beta,1), p)];
	end
	beta(k+1,A) = beta(k,A) + gamma*w';
	
	% Early stopping at specified bound on L1 norm of beta
	if stop > 0
		t2 = sum(abs(beta(k+1,:)));
		if t2 >= stop
			t1 = sum(abs(beta(k,:)));
			s = (stop - t1)/(t2 - t1); % interpolation factor 0 < s < 1
			beta(k+1,:) = beta(k,:) + s*(beta(k+1,:) - beta(k,:));
			stopcond = 1;
		end
	end
	
	% If LASSO condition satisfied, drop variable from active set
	if lassocond == 1
		R = choldelete(R,j);
		I = [I A(j)];
		A(j) = [];
		vars = vars - 1;
		if trace
			disp(sprintf('%d\t\t\t\t%d\t\t\t%d', k, j, vars));
		end
	end
	
	% Early stopping at specified number of variables
	if stop < 0
		stopcond = vars >= -stop;
	end
end

% trim beta
if size(beta,1) > k+1
	beta(k+2:end, :) = [];
end

% divide by d2 to avoid double shrinkage
beta = beta/d2;
beta = beta(2:end,:)';


%%%%%%%% turn output monotonic in model size
betaold = beta; clear beta;
for j = 1:size(betaold,2)
	nos(j) = size(find(betaold(:,j)),1);
	
end
for j = 1:p
	%% in the VERY UNLIKELY case where one model size is not represented, cheat
	if isempty(find(nos == j))
		beta(:,j) = betaold(:,find(nos==(j-1),1));	
	else
		beta(:,j) = betaold(:,find(nos==j,1,'last'));
	end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Here we extract a monotonic candidate set 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if k == maxk
	disp('LARS-EN warning: Forced exit. Maximum number of iteration reached.');
end


%% Fast Cholesky insert and remove functions
% Updates R in a Cholesky factorization R'R = X'X of a data matrix X. R is
% the current R matrix to be updated. x is a column vector representing the
% variable to be added and X is the data matrix containing the currently
% active variables (not including x).
function R = cholinsert(R, x, X, lambda)
diag_k = (x'*x + lambda)/(1 + lambda); % diagonal element k in X'X matrix
if isempty(R)
  R = sqrt(diag_k);
else
  col_k = x'*X/(1 + lambda); % elements of column k in X'X matrix
  R_k = R'\col_k'; % R'R_k = (X'X)_k, solve for R_k
  R_kk = sqrt(diag_k - R_k'*R_k); % norm(x'x) = norm(R'*R), find last element by exclusion
  R = [R R_k; [zeros(1,size(R,2)) R_kk]]; % update R
end

% Deletes a variable from the X'X matrix in a Cholesky factorisation R'R =
% X'X. Returns the downdated R. This function is just a stripped version of
% Matlab's qrdelete.
function R = choldelete(R,j)
R(:,j) = []; % remove column j
n = size(R,2);
for k = j:n
  p = k:k+1;
  [G,R(p,k)] = planerot(R(p,k)); % remove extra element in column
  if k < n
    R(p,k+1:n) = G*R(p,k+1:n); % adjust rest of row
  end
end
R(end,:) = []; % remove zero'ed out row
