function [B_train , B_test, trtime] = POSH_fun(data_our, pars)

gamma     = 1; 
eta       = pars.eta; 
lambda    = pars.lambda; 
Iter_num = pars.Iter_num;
nbits    = pars.nbits;

X  = data_our.X(:,data_our.indexTrain); %train
X2 = data_our.X(:,data_our.indexTest); %test
y  = data_our.label(data_our.indexTrain, :);

% label matrix G = c x N   (c is amount of classes, N is amount of instances)
if isvector(y)
    G = sparse(1:length(y), double(y), 1); G = full(G');
else
    G = y';
end
[c, ~] = size(G);
[d,N1] = size(X);
    
nn_num = 2;
n_anchors = 1000;
rand('seed',100);randn('seed',100);
anchor = X(:,randperm(N1, n_anchors));
[Z,M] = AGH_Graph(X', anchor', nn_num);
XZ = X*Z;
L = XZ*M*XZ';

% %%%%%% B - initialize %%%%%%
rand('seed',100);randn('seed',100);
B = randn(nbits, N1)>0; B = B*2-1;
rand('seed',100);randn('seed',100);
R = randn(c, nbits);  %G=Y'*B;
XXT = X*X';
D  = eye(d);
tmpG = (G*G'+lambda*eye(c))\G;

tic;
%------------------------training----------------------------
for iter = 1:Iter_num
    
    %fprintf('The %d-th iteration...\n',iter);
    B0 = B;
    
    % ----------------------- U-step -----------------------%
    U = (-eta*L+(eta+gamma)*XXT+lambda*D)\(gamma*X*B');
    
    % ----------------------- R-step -----------------------%
    R = tmpG*B';
    
    % ----------------------- B-step -----------------------%
    A = R'*G+gamma*U'*X; %mu = median(A,2); A = bsxfun(@minus, A, mu);
    B = sign(A);

    % ----------------------- D-step -----------------------%
    dii = 0.5./sqrt(sum(U.*U,2)+eps);
    D = spdiags(dii,0,d,d);
    f(iter) = norm(B-B0);
end
trtime = toc;

B_train = B'>0;
%-----------------------------------------------------Out-of-Sample-------------------------------------
NT = (X * X' + 1 * eye(size(X, 1))) \ X;
W = NT*B_train;
B_test = X2'*W >0;
end