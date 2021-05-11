function [ params ] = solve(params)
X = params.X;
P = params.P;
B = params.B;
Z = params.Z;

lambda = params.lambda;
alpha = params.alpha;
beta = params.beta;
b = params.b;
[n,dim] = size(X);
epchos = params.epchos;
label = params.train_label;
label = normalize(label);

for i=1:epchos
%% «ÛP
P = (lambda*(X'*X)+beta*eye(dim))\(lambda*X'*B);
%% «ÛZ
C = 2*b*label*(label'*B) - b*ones(n,1)*(ones(n,1)'*B) + alpha*B;
% J = eye(n) - 1/n*ones(n,n);
tmpcjc = C'*C - 1/n*C'*ones(n,1)*ones(n,1)'*C;
[V,D] = eig(tmpcjc);
[D,index] = sort(diag(D),'descend');
D = diag(D);
V = V(:,index);
for k=1:b
    if D(k,k)<1e-5
        k = k-1;
        break;
    end
end
D = D(1:k,1:k);
D = D.^0.5;
% U = J*C*V(:,1:k)/D;
U = (C*V(:,1:k)-1/n*ones(n,1)*(ones(n,1)'*C*V(:,1:k)))/D;
b_ = b-k;
if(b_>0)
    UY = rand(n,b_);
    UY = UY - repmat(mean(UY),n,1);
    U = [U,UY];
    U = Schmidt(U);
end
Z = sqrt(n)*U*V';
%% «ÛB
C = (2*b*(Z'*label)*label'-b*(Z'*ones(n,1))*ones(n,1)' + lambda*P'*X' + alpha*Z')';
B = sgn(C);

% print loss
% params.P = P;
% params.B = B;
% params.Z = Z;
% los = Loss(params);
% fprintf('epcho:%d, loss:%f\n', i,los);

end
%%
params.P = P;
params.B = B;
params.Z = Z;
end

