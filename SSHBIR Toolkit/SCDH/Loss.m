function [ los ] = Loss( params )
% Objective function loss
X = params.X;
P = params.P;
B = params.B;
Z = params.Z;
n = size(X,1);

label = params.train_label;
label = normalize(label);

lambda = params.lambda;
alpha = params.alpha;
beta = params.beta;
b = params.b;

los = 0;
% los = los + trace(Z'*Z*(B'*B)-2*b*Z'*s1*s2*B)+trace(b*b*(s2*s2')*(s1'*s1));
los = los + trace(Z'*Z*(B'*B)-4*b*Z'*label*label'*B+2*b*Z'*ones(n,1)*ones(n,1)'*B);
los = los + trace(b*b*(4*(label'*label)*label'*label-2*(label'*ones(n,1))*ones(n,1)'*label- ...
    2*(ones(n,1)'*label)*label'*ones(n,1)+(ones(n,1)'*ones(n,1))*ones(n,1)'*ones(n,1)));
los = los + lambda * sum(sum((X*P-B).^2));
los = los + alpha * sum(sum((B-Z).^2));
los = los + beta * sum(sum(P.^2));

end

